"""Training dataset and collator for synchronized DynaDEx instructions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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
    """Sample complete records without replacement using a fixed seed."""
    if maximum_samples is None or maximum_samples <= 0 or maximum_samples >= len(records):
        return records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in sorted(indices[:maximum_samples])]


class DynaDExDataset(Dataset):
    """Tokenize prompts, targets, and their synchronized schema definitions."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        maximum_length: int = 2048,
        maximum_definition_length: int = 64,
        maximum_samples: int | None = None,
        seed: int = 42,
        use_schema_mapper: bool = True,
    ) -> None:
        self.records = select_sentence_subset(read_jsonl(path), maximum_samples, seed)
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length
        self.maximum_definition_length = maximum_definition_length
        self.use_schema_mapper = use_schema_mapper

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.records[index]
        prompt = str(item["instruction"])
        output = str(item.get("output", ""))
        eos_token = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = self.tokenizer(output + eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(output_ids) > self.maximum_length:
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
        definitions = [str(entry["definition"]) for entry in entries]
        encoded = self.tokenizer(
            definitions,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.maximum_definition_length,
            return_tensors="pt",
        )
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
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
            "labels": labels,
        }
        if not self.use_schema_mapper:
            return result

        maximum_roles = max(item["schema_input_ids"].shape[0] for item in batch)
        definition_length = batch[0]["schema_input_ids"].shape[1]
        schema_ids = torch.full(
            (len(batch), maximum_roles, definition_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        schema_mask = torch.zeros_like(schema_ids)
        role_mask = torch.zeros((len(batch), maximum_roles), dtype=torch.long)
        for row_index, item in enumerate(batch):
            role_count = item["schema_input_ids"].shape[0]
            schema_ids[row_index, :role_count] = item["schema_input_ids"]
            schema_mask[row_index, :role_count] = item["schema_attention_mask"]
            role_mask[row_index, :role_count] = 1
        result.update(
            {
                "schema_input_ids": schema_ids,
                "schema_attention_mask": schema_mask,
                "schema_role_mask": role_mask,
            }
        )
        return result
