"""Measure fixed-length latency and CUDA memory for matched configurations."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from .data import read_jsonl
from .inference import generate_batch, load_runtime
from .training_common import configure_reproducibility


def measure_configuration(
    rows: list[dict[str, Any]],
    base_model: str,
    phase1_checkpoint: Path,
    phase2_checkpoint: Path | None,
    use_schema_mapper: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Measure one checkpoint after model loading and fixed warm-up."""
    tokenizer, model = load_runtime(
        base_model,
        phase1_checkpoint,
        phase2_checkpoint,
        args.latent_length,
        use_schema_mapper,
        args.dtype,
        args.device,
    )
    warmup_rows = rows[: args.warmup_sentences]
    measured_rows = rows[
        args.warmup_sentences : args.warmup_sentences + args.measured_sentences
    ]
    if len(measured_rows) < args.measured_sentences:
        raise ValueError("The input file does not contain enough test sentences")

    for row in warmup_rows:
        generate_batch(
            [row], tokenizer, model, args.maximum_prompt_length, 64, 64
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for row in measured_rows:
        generate_batch(
            [row], tokenizer, model, args.maximum_prompt_length, 64, 64
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    reserved = (
        torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "latency_ms_per_sentence": 1000.0 * elapsed / len(measured_rows),
        "peak_allocated_gb": allocated,
        "peak_reserved_gb": reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input", type=Path, default=Path("data/processed/test_ins.jsonl"))
    parser.add_argument("--full-phase1-checkpoint", type=Path, default=Path("outputs/full/phase1"))
    parser.add_argument("--full-phase2-checkpoint", type=Path, default=None)
    parser.add_argument("--ablated-phase1-checkpoint", type=Path, default=Path("outputs/ablated/phase1"))
    parser.add_argument("--ablated-phase2-checkpoint", type=Path, default=None)
    parser.add_argument("--ablated-with-schema-mapper", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/inference_efficiency.json"))
    parser.add_argument("--warmup-sentences", type=int, default=10)
    parser.add_argument("--measured-sentences", type=int, default=100)
    parser.add_argument("--maximum-prompt-length", type=int, default=2048)
    parser.add_argument("--latent-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_reproducibility(args.seed)
    rows = read_jsonl(args.input)
    full = measure_configuration(
        rows,
        args.base_model,
        args.full_phase1_checkpoint,
        args.full_phase2_checkpoint,
        True,
        args,
    )
    ablated = measure_configuration(
        rows,
        args.base_model,
        args.ablated_phase1_checkpoint,
        args.ablated_phase2_checkpoint,
        args.ablated_with_schema_mapper,
        args,
    )
    result = {
        "protocol": {
            "generated_tokens": 64,
            "warmup_sentences": args.warmup_sentences,
            "measured_sentences": args.measured_sentences,
        },
        "full": full,
        "ablated": ablated,
        "latency_ratio": full["latency_ms_per_sentence"]
        / ablated["latency_ms_per_sentence"],
        "allocated_overhead_gb": full["peak_allocated_gb"]
        - ablated["peak_allocated_gb"],
        "reserved_overhead_gb": full["peak_reserved_gb"]
        - ablated["peak_reserved_gb"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
