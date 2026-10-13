"""Build synchronized visible and latent schema inputs for DynaDEx."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_records, write_jsonl


META_INSTRUCTION = (
    "You are DynaDEx, a schema-conditioned extraction system. Given an input "
    "sentence and a schema, extract every text span\n"
    "that satisfies one of the schema definitions."
)

TASK_INSTRUCTION = """Read the schema definitions and analyze the complete input sentence.
Execution Steps:
1. Read each schema label and its definition.
2. Identify every text span that satisfies a provided definition.
3. Return each valid label-span pair once.
Output Format: (Role Label, Text Span).
Demonstration:
Sample Input: Steve Jobs is the founder of Apple.
Sample Output: (Person, Steve Jobs) (Company, Apple)"""


def normalize_definitions(value: Any) -> list[str]:
    """Return a non-empty list of textual definitions."""
    values = value if isinstance(value, list) else [value]
    definitions = [str(item).strip() for item in values if item is not None and str(item).strip()]
    if not definitions:
        raise ValueError("Every schema label must have at least one definition")
    return definitions


class SchemaRepository:
    """Resolve flat schemas and predicate-indexed FrameNet schemas."""

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Schema must be a non-empty JSON array: {path}")
        self.flat: list[tuple[str, list[str]]] | None = None
        self.by_key: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}
        self.by_frame: dict[str, list[list[tuple[str, list[str]]]]] = defaultdict(list)
        self.by_lexical_unit: dict[str, list[list[tuple[str, list[str]]]]] = defaultdict(list)

        if "role" in payload[0]:
            self.flat = [
                (str(item["role"]), normalize_definitions(item.get("def")))
                for item in payload
            ]
            return

        for item in payload:
            meta = item.get("meta_info", {})
            lexical_unit = str(meta.get("lexical_unit", ""))
            frame = str(meta.get("frame", ""))
            if not lexical_unit or not frame:
                continue
            entries = [
                (str(role["role"]), normalize_definitions(role.get("def")))
                for role in item.get("schema", [])
                if role.get("role")
            ]
            self.by_key[(lexical_unit, frame)] = entries
            self.by_frame[frame].append(entries)
            self.by_lexical_unit[lexical_unit].append(entries)
        if not self.by_key:
            raise ValueError(f"Unsupported schema structure: {path}")

    def resolve(self, record: dict[str, Any]) -> list[tuple[str, list[str]]]:
        """Resolve the task-level or predicate-indexed schema for one record."""
        if self.flat is not None:
            return list(self.flat)
        meta = record.get("meta_info", {})
        lexical_unit = str(meta.get("lexical_unit", record.get("lexical_unit", "")))
        frame = str(
            meta.get(
                "frame",
                record.get("frame", meta.get("predicate_sense", record.get("predicate_sense", ""))),
            )
        )
        if (lexical_unit, frame) in self.by_key:
            return list(self.by_key[(lexical_unit, frame)])
        if frame and len(self.by_frame.get(frame, [])) == 1:
            return list(self.by_frame[frame][0])
        if lexical_unit and len(self.by_lexical_unit.get(lexical_unit, [])) == 1:
            return list(self.by_lexical_unit[lexical_unit][0])
        raise KeyError(
            f"No unique predicate-indexed schema for record {record.get('id', '<unknown>')}"
        )


def choose_entries(
    schema: list[tuple[str, list[str]]],
    rng: random.Random,
    augment_definitions: bool,
    shuffle_schema: bool,
) -> list[dict[str, str]]:
    """Choose one definition per label and optionally randomize label order."""
    selected = [
        {
            "label": label,
            "definition": rng.choice(definitions) if augment_definitions else definitions[0],
        }
        for label, definitions in schema
    ]
    if shuffle_schema:
        rng.shuffle(selected)
    return selected


def build_prompt(sentence: str, entries: list[dict[str, str]]) -> str:
    """Render the prompt specification shared by every execution regime."""
    schema_lines = "\n".join(
        f"- Label: {entry['label']} | Definition: {entry['definition']}"
        for entry in entries
    )
    return (
        f"[Meta Instruction]\n{META_INSTRUCTION}\n"
        f"[Task]\n{TASK_INSTRUCTION}\n"
        f"[Schema Definitions]\n<schema>\n{schema_lines}\n</schema>\n"
        f"[Input Sentence]\n{sentence}\n[Response]\n"
    )


def format_output(roles: list[list[str]]) -> str:
    """Serialize gold label-span pairs using the common output grammar."""
    return " ".join(f"({label}, {span})" for label, span in roles)


def build_rows(
    records: list[dict[str, Any]],
    repository: SchemaRepository,
    seed: int,
    augment_definitions: bool,
    shuffle_schema: bool,
) -> list[dict[str, Any]]:
    """Create instruction rows with synchronized textual and latent definitions."""
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rng = random.Random(seed + index)
        entries = choose_entries(
            repository.resolve(record), rng, augment_definitions, shuffle_schema
        )
        sentence = str(record["sentence"])
        roles = [[str(role[0]), str(role[1])] for role in record.get("roles", [])]
        rows.append(
            {
                "id": str(record.get("id", f"record-{index:08d}")),
                "instruction": build_prompt(sentence, entries),
                "output": format_output(roles),
                "sentence": sentence,
                "roles": roles,
                "schema_entries": entries,
                "meta_info": record.get("meta_info", {}),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset.json"))
    parser.add_argument("--schema", type=Path, default=Path("data/schemas/dataset_schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dataset_ins.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment-definitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select paraphrased definitions for Phase 1 training records.",
    )
    parser.add_argument("--shuffle-schema", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        read_records(args.input),
        SchemaRepository(args.schema),
        args.seed,
        args.augment_definitions,
        args.shuffle_schema,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} instruction records to {args.output}")


if __name__ == "__main__":
    main()
