"""Evaluate canonical, syntactic, and lexical definition conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import read_records, score_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--canonical", type=Path, default=Path("results/canonical.json"))
    parser.add_argument("--syntactic", type=Path, default=Path("results/syntactic.json"))
    parser.add_argument("--lexical", type=Path, default=Path("results/lexical.json"))
    parser.add_argument("--mode", choices=("pair", "srl"), default="pair")
    parser.add_argument("--output", type=Path, default=Path("results/rephrasing_metrics.json"))
    args = parser.parse_args()

    gold = read_records(args.gold)
    conditions = {
        "canonical": score_records(gold, read_records(args.canonical), args.mode),
        "syntactic": score_records(gold, read_records(args.syntactic), args.mode),
        "lexical": score_records(gold, read_records(args.lexical), args.mode),
    }
    canonical_exact = set(conditions["canonical"]["exact_record_ids"])
    syntactic_exact = set(conditions["syntactic"]["exact_record_ids"])
    lexical_exact = set(conditions["lexical"]["exact_record_ids"])
    denominator = len(canonical_exact)
    result = {
        "conditions": conditions,
        "f1_change_points": {
            name: 100.0 * (metrics["f1"] - conditions["canonical"]["f1"])
            for name, metrics in conditions.items()
            if name != "canonical"
        },
        "canonical_exact_records": denominator,
        "syntactic_survival": len(canonical_exact & syntactic_exact),
        "lexical_survival": len(canonical_exact & lexical_exact),
        "joint_survival": len(canonical_exact & syntactic_exact & lexical_exact),
        "joint_survival_rate": (
            len(canonical_exact & syntactic_exact & lexical_exact) / denominator
            if denominator
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["f1_change_points"], indent=2))


if __name__ == "__main__":
    main()
