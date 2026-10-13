"""Run zero-shot or adapted DynaDEx inference with a supplied schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer

from .checkpointing import load_phase1_model, load_phase2_model
from .data import read_jsonl
from .modeling_dynadex import DynaDExModel
from .training_common import configure_reproducibility, parse_dtype


PAIR_PATTERN = re.compile(r"\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)")
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def parse_response(
    response: str,
    allowed_labels: set[str],
    sentence: str,
) -> list[list[str]]:
    """Parse valid, unique label-span pairs in generation order."""
    cleaned = THINK_PATTERN.sub("", response)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for match in PAIR_PATTERN.finditer(cleaned):
        label = match.group(1).strip()
        span = match.group(2).strip()
        pair = (label, span)
        if label not in allowed_labels or not span or span not in sentence or pair in seen:
            continue
        pairs.append([label, span])
        seen.add(pair)
    return pairs


def encode_schema_batch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    maximum_definition_length: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize variable-size schemas and pad the role dimension."""
    maximum_roles = max(len(row.get("schema_entries", [])) for row in rows)
    if maximum_roles == 0:
        raise ValueError("Every row must contain at least one schema entry")
    schema_ids = torch.full(
        (len(rows), maximum_roles, maximum_definition_length),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
    )
    schema_mask = torch.zeros_like(schema_ids)
    role_mask = torch.zeros((len(rows), maximum_roles), dtype=torch.long)
    for row_index, row in enumerate(rows):
        definitions = [
            str(entry["definition"]) for entry in row.get("schema_entries", [])
        ]
        encoded = tokenizer(
            definitions,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=maximum_definition_length,
            return_tensors="pt",
        )
        role_count = len(definitions)
        schema_ids[row_index, :role_count] = encoded["input_ids"]
        schema_mask[row_index, :role_count] = encoded["attention_mask"]
        role_mask[row_index, :role_count] = 1
    return schema_ids, schema_mask, role_mask


def load_runtime(
    base_model: str,
    phase1_checkpoint: Path | None,
    phase2_checkpoint: Path | None,
    latent_length: int,
    use_schema_mapper: bool,
    dtype_name: str,
    device: str,
) -> tuple[Any, Any]:
    """Load the tokenizer and the requested execution-regime checkpoint."""
    dtype = parse_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if phase1_checkpoint is None:
        if phase2_checkpoint is not None:
            raise ValueError("A Phase 2 checkpoint requires its Phase 1 checkpoint")
        model = DynaDExModel.from_pretrained(
            base_model,
            latent_length=latent_length,
            use_schema_mapper=False,
            add_lora=False,
            torch_dtype=dtype,
        )
    elif phase2_checkpoint is None:
        model = load_phase1_model(
            base_model,
            phase1_checkpoint,
            latent_length,
            use_schema_mapper,
            dtype,
        )
    else:
        model = load_phase2_model(
            base_model,
            phase1_checkpoint,
            phase2_checkpoint,
            latent_length,
            use_schema_mapper,
            dtype,
        )
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def generate_batch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    maximum_prompt_length: int,
    maximum_new_tokens: int,
    minimum_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate and parse predictions for one batch."""
    encoded = tokenizer(
        [str(row["instruction"]) for row in rows],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    prompt_lengths = encoded["attention_mask"].sum(dim=1)
    if int(prompt_lengths.max()) > maximum_prompt_length:
        raise ValueError(
            f"A prompt has {int(prompt_lengths.max())} tokens, exceeding "
            f"the configured limit of {maximum_prompt_length}"
        )

    if model.schema_mapper is None:
        schema_ids = schema_mask = role_mask = None
    else:
        schema_ids, schema_mask, role_mask = encode_schema_batch(rows, tokenizer)
    generated = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        schema_input_ids=schema_ids,
        schema_attention_mask=schema_mask,
        schema_role_mask=role_mask,
        pad_token_id=int(tokenizer.pad_token_id),
        do_sample=False,
        max_new_tokens=maximum_new_tokens,
        min_new_tokens=minimum_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
    predictions: list[dict[str, Any]] = []
    for row, response in zip(rows, responses):
        labels = {str(entry["label"]) for entry in row.get("schema_entries", [])}
        predictions.append(
            {
                "id": row.get("id"),
                "sentence": row.get("sentence", ""),
                "predicted_roles": parse_response(
                    response,
                    labels,
                    str(row.get("sentence", "")),
                ),
                "gold_roles": row.get("roles", []),
                "raw_output": response,
            }
        )
    return predictions


def run_inference(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    maximum_prompt_length: int,
    maximum_new_tokens: int,
    minimum_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run deterministic batched inference over all rows."""
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        outputs.extend(
            generate_batch(
                rows[start : start + batch_size],
                tokenizer,
                model,
                maximum_prompt_length,
                maximum_new_tokens,
                minimum_new_tokens,
            )
        )
    return outputs


def write_predictions(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write prediction records as an indented JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--phase1-checkpoint", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--phase2-checkpoint", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=Path("data/processed/test_ins.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/predictions.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--maximum-prompt-length", type=int, default=2048)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--minimum-new-tokens", type=int, default=0)
    parser.add_argument("--latent-length", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--without-schema-mapper", action="store_true")
    parser.add_argument(
        "--instruction-only",
        action="store_true",
        help="Evaluate the untouched backbone with the visible instruction only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.seed)
    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    tokenizer, model = load_runtime(
        args.base_model,
        None if args.instruction_only else args.phase1_checkpoint,
        args.phase2_checkpoint,
        args.latent_length,
        not args.without_schema_mapper and not args.instruction_only,
        args.dtype,
        args.device,
    )
    predictions = run_inference(
        rows,
        tokenizer,
        model,
        args.batch_size,
        args.maximum_prompt_length,
        args.maximum_new_tokens,
        args.minimum_new_tokens,
    )
    write_predictions(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
