"""Build synchronized definition-rephrasing inputs without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_instructions import build_prompt
from .common import read_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-input", type=Path, default=Path("data/processed/test_ins.jsonl"))
    parser.add_argument("--variants", type=Path, default=Path("data/schemas/definition_variants.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/definition_variants"))
    args = parser.parse_args()

    canonical_rows = read_records(args.canonical_input)
    variant_payload = json.loads(args.variants.read_text(encoding="utf-8"))
    if not isinstance(variant_payload, dict):
        raise ValueError("The variants file must map condition names to label-definition maps")

    for condition, replacements in variant_payload.items():
        if not isinstance(replacements, dict):
            raise ValueError(f"Condition {condition} must contain a label-definition map")
        output_rows = []
        for row in canonical_rows:
            canonical_entries = row.get("schema_entries", [])
            labels = [entry["label"] for entry in canonical_entries]
            if set(labels) != set(replacements):
                raise ValueError(
                    f"Condition {condition} does not define exactly the canonical labels"
                )
            entries = [
                {"label": label, "definition": str(replacements[label])}
                for label in labels
            ]
            updated = dict(row)
            updated["schema_entries"] = entries
            updated["instruction"] = build_prompt(str(row["sentence"]), entries)
            output_rows.append(updated)
        output = args.output_dir / f"{condition}.jsonl"
        write_jsonl(output, output_rows)
        print(f"Wrote {len(output_rows)} synchronized records to {output}")


if __name__ == "__main__":
    main()
