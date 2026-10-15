"""SCIM schema encoding, training datasets, and batch collation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from preprocessing.sampling import select_sentences

SCHEMA_ENCODING = "label-definition-v1"


def encode_entries(entries: list[dict[str, str]], tokenizer: Any) -> dict[str, torch.Tensor]:
    if not entries:
        raise ValueError("Every schema must contain at least one entry")
    rows = []
    for entry in entries:
        label, definition = entry["label"], entry["definition"]
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(definition, str)
            or not definition
        ):
            raise ValueError("Schema labels and definitions must be non-empty strings")
        # Tokenize each part separately so Eq. (3) averages exactly |r|+|d| tokens.
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        ids += tokenizer(definition, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Schema entry {label!r} has no tokens")
        rows.append(ids)
    length = max(map(len, rows))
    ids = torch.full((len(rows), length), int(tokenizer.pad_token_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def encode_schema_batch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_entries(row.get("schema_entries", []), tokenizer) for row in rows]
    roles = max(item["input_ids"].shape[0] for item in encoded)
    length = max(item["input_ids"].shape[1] for item in encoded)
    ids = torch.full((len(rows), roles, length), int(tokenizer.pad_token_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    role_mask = torch.zeros((len(rows), roles), dtype=torch.long)
    for index, item in enumerate(encoded):
        count, width = item["input_ids"].shape
        ids[index, :count, :width] = item["input_ids"]
        mask[index, :count, :width] = item["attention_mask"]
        role_mask[index, :count] = 1
    return ids, mask, role_mask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file into memory."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(item)
    return records


def select_sentence_subset(
    records: list[dict[str, Any]], maximum_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    """Sample complete source sentences, retaining every predicate instance."""
    return select_sentences(records, maximum_samples, seed)


class DynaDExDataset(Dataset):
    """Tokenize prompts, targets, and their synchronized schema definitions."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        maximum_length: int = 2048,
        maximum_samples: int | None = None,
        seed: int = 42,
        use_schema_mapper: bool = True,
        latent_length: int = 10,
    ) -> None:
        self.records = select_sentence_subset(read_jsonl(path), maximum_samples, seed)
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length
        self.use_schema_mapper = use_schema_mapper
        self.latent_length = latent_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.records[index]
        prompt = str(item["instruction"])
        output = str(item.get("output", ""))
        eos_token = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = self.tokenizer(output + eos_token, add_special_tokens=False)["input_ids"]
        prefix_length = self.latent_length if self.use_schema_mapper else 0
        if len(prompt_ids) + len(output_ids) + prefix_length > self.maximum_length:
            raise ValueError(
                f"Record {item.get('id', index)} has {len(prompt_ids) + len(output_ids)} "
                f"tokens, exceeding the configured limit of {self.maximum_length}"
            )

        input_ids = torch.tensor(prompt_ids + output_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + output_ids, dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }
        if not self.use_schema_mapper:
            return result

        entries = item.get("schema_entries", [])
        if not entries:
            raise ValueError(f"Record {item.get('id', index)} has no schema entries")
        encoded = encode_entries(entries, self.tokenizer)
        result["schema_input_ids"] = encoded["input_ids"]
        result["schema_attention_mask"] = encoded["attention_mask"]
        return result


class DynaDExCollator:
    """Dynamically pad text sequences and variable-size schemas."""

    def __init__(self, tokenizer: Any, use_schema_mapper: bool = True) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.use_schema_mapper = use_schema_mapper

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        result = {
            "input_ids": input_ids,
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": labels,
        }
        if not self.use_schema_mapper:
            return result

        maximum_roles = max(item["schema_input_ids"].shape[0] for item in batch)
        definition_length = max(item["schema_input_ids"].shape[1] for item in batch)
        schema_ids = torch.full(
            (len(batch), maximum_roles, definition_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        schema_mask = torch.zeros_like(schema_ids)
        role_mask = torch.zeros((len(batch), maximum_roles), dtype=torch.long)
        for row_index, item in enumerate(batch):
            role_count = item["schema_input_ids"].shape[0]
            width = item["schema_input_ids"].shape[1]
            schema_ids[row_index, :role_count, :width] = item["schema_input_ids"]
            schema_mask[row_index, :role_count, :width] = item["schema_attention_mask"]
            role_mask[row_index, :role_count] = 1
        result.update(
            {
                "schema_input_ids": schema_ids,
                "schema_attention_mask": schema_mask,
                "schema_role_mask": role_mask,
            }
        )
        return result
