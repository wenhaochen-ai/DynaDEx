"""Create nested sentence-level annotation-budget subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_records, write_json, write_jsonl
from .sampling import select_sentences, subset_manifest

DEFAULT_BUDGETS = (100, 250, 500, 1000, 2000, 3000)


def parse_integer_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected one or more positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/train_ins.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/budgets"))
    parser.add_argument("--budgets", type=parse_integer_list, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=parse_integer_list, default=[42])
    args = parser.parse_args()

    records = read_records(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")

    for seed in args.seeds:
        for requested_budget in sorted(set(args.budgets)):
            selected = select_sentences(records, requested_budget, seed)
            manifest = subset_manifest(selected, seed, requested_budget)
            actual_budget = manifest["actual_sentences"]
            output = args.output_dir / f"seed_{seed}" / f"train_n{actual_budget}.jsonl"
            write_jsonl(output, selected)
            write_json(output.with_suffix(".manifest.json"), manifest)
            print(
                f"seed={seed} requested={requested_budget} actual={actual_budget} output={output}"
            )


if __name__ == "__main__":
    main()
