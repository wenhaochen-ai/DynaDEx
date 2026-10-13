"""Convert source corpora to the common DynaDEx span-extraction format."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import normalized_record, read_records, write_json


def collapse_tagged_spans(
    tokens: list[str], labels: list[str], separator: str
) -> list[list[str]]:
    """Collapse BIOES labels into labeled text spans."""
    spans: list[list[str]] = []
    active_label: str | None = None
    active_tokens: list[str] = []

    def flush() -> None:
        nonlocal active_label, active_tokens
        if active_label and active_tokens:
            spans.append([active_label, separator.join(active_tokens)])
        active_label = None
        active_tokens = []

    for token, label in zip(tokens, labels):
        if label in {"O", "_", ""}:
            flush()
            continue
        if label == "rel":
            flush()
            spans.append(["PRED", token])
            continue
        prefix, role = label.split("-", 1) if "-" in label else ("S", label)
        if prefix == "B":
            flush()
            active_label, active_tokens = role, [token]
        elif prefix == "I" and active_label == role:
            active_tokens.append(token)
        elif prefix == "E" and active_label == role:
            active_tokens.append(token)
            flush()
        else:
            flush()
            if prefix == "S":
                spans.append([role, token])
            else:
                active_label, active_tokens = role, [token]
    flush()
    return spans


def convert_cpb_bioes(path: Path, language: str) -> list[dict[str, Any]]:
    """Convert CPB-style ``word/POS/label`` lines to normalized records."""
    separator = " " if language == "en" else ""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        words: list[str] = []
        labels: list[str] = []
        for field in line.split():
            parts = field.rsplit("/", 2)
            if len(parts) != 3:
                raise ValueError(f"Malformed token at line {line_number + 1}: {field}")
            word, _, label = parts
            words.append(word)
            labels.append(label)
        records.append(
            normalized_record(
                separator.join(words),
                collapse_tagged_spans(words, labels, separator),
                f"cpb-{line_number:08d}",
            )
        )
    return records


def sentence_blocks(path: Path) -> Iterable[list[list[str]]]:
    """Yield tab-separated sentence blocks from a CoNLL file."""
    block: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                block.append(line.rstrip("\n").split("\t"))
            elif block:
                yield block
                block = []
    if block:
        yield block


def dependency_subtree(root_id: int, children: dict[int, list[int]]) -> list[int]:
    """Return the dependency subtree rooted at ``root_id``."""
    stack = [root_id]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(children.get(node, []))
    return sorted(visited)


def convert_conll2009(
    path: Path,
    language: str,
    allowed_roles: set[str] | None,
) -> list[dict[str, Any]]:
    """Convert one predicate-indexed CoNLL-2009 instance per sentence."""
    separator = " " if language == "en" else ""
    records: list[dict[str, Any]] = []
    for sentence_index, columns in enumerate(sentence_blocks(path)):
        if any(len(row) < 14 for row in columns):
            raise ValueError(f"Sentence {sentence_index} has fewer than 14 columns")

        children: dict[int, list[int]] = defaultdict(list)
        for row in columns:
            children[int(row[8])].append(int(row[0]))

        predicate_rows = [
            (index, row[13])
            for index, row in enumerate(columns)
            if row[13] != "_"
        ]
        root_candidates = [
            index for index, row in enumerate(columns) if int(row[8]) == 0
        ]
        if not predicate_rows or not root_candidates:
            continue

        predicate_index = root_candidates[0]
        if columns[predicate_index][13] == "_" and language == "en":
            root_id = int(columns[predicate_index][0])
            dependents = [
                index
                for index, row in enumerate(columns)
                if int(row[8]) == root_id and row[13] != "_"
            ]
            verbal = [index for index in dependents if columns[index][4].startswith("V")]
            if verbal or dependents:
                predicate_index = (verbal or dependents)[0]

        predicate_rank = next(
            (rank for rank, (index, _) in enumerate(predicate_rows) if index == predicate_index),
            None,
        )
        if predicate_rank is None:
            continue

        predicate_sense = columns[predicate_index][13]
        predicate_span = predicate_sense.split(".", 1)[0]
        roles: list[list[str]] = [["PRED", predicate_span]]
        argument_column = 14 + predicate_rank
        reject_sentence = False
        for row in columns:
            if argument_column >= len(row) or row[argument_column] == "_":
                continue
            role = row[argument_column]
            if allowed_roles is not None and role not in allowed_roles:
                reject_sentence = True
                break
            token_ids = dependency_subtree(int(row[0]), children)
            span = separator.join(columns[token_id - 1][1] for token_id in token_ids)
            roles.append([role, span])
        if reject_sentence:
            continue

        sentence = separator.join(row[1] for row in columns)
        records.append(
            normalized_record(
                sentence,
                roles,
                f"conll2009-{sentence_index:08d}",
                {"predicate_sense": predicate_sense},
            )
        )
    return records


def convert_entity_json(
    path: Path,
    sentence_field: str,
    tokens_field: str,
    entities_field: str,
    label_field: str,
    text_field: str,
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert entity-style JSON used by ACE2005, FabNER, and FIRE."""
    records: list[dict[str, Any]] = []
    for index, item in enumerate(read_records(path)):
        sentence = str(item.get(sentence_field, "")).strip()
        if not sentence and tokens_field:
            sentence = " ".join(str(token) for token in item.get(tokens_field, []))
        roles: list[list[str]] = []
        for entity in item.get(entities_field, []):
            label = entity.get(label_field)
            span = entity.get(text_field)
            if label is not None and span is not None:
                normalized_label = label_map.get(str(label).lower(), str(label))
                roles.append([normalized_label, str(span)])
        records.append(normalized_record(sentence, roles, f"entity-{index:08d}"))
    return records


def convert_phee(path: Path) -> list[dict[str, Any]]:
    """Convert PHEE event mentions to trigger and argument spans."""
    records: list[dict[str, Any]] = []
    for index, item in enumerate(read_records(path)):
        roles: list[list[str]] = []
        for event in item.get("event_mentions", []):
            trigger = event.get("trigger", {})
            if trigger.get("text"):
                roles.append(["Trigger", str(trigger["text"])])
            for argument in event.get("arguments", []):
                if argument.get("role") and argument.get("text"):
                    roles.append([str(argument["role"]), str(argument["text"])])
        records.append(
            normalized_record(str(item.get("text", "")), roles, f"phee-{index:08d}")
        )
    return records


def convert_normalized(path: Path) -> list[dict[str, Any]]:
    """Validate already normalized records, including FrameNet and CoNLL-2012."""
    records: list[dict[str, Any]] = []
    for index, item in enumerate(read_records(path)):
        meta = item.get("meta_info") or item.get("aux_meta_info") or {}
        records.append(
            normalized_record(
                str(item.get("sentence", "")),
                item.get("roles", []),
                str(item.get("id", f"normalized-{index:08d}")),
                meta if isinstance(meta, dict) else {},
            )
        )
    return records


def load_schema_labels(path: Path) -> set[str]:
    """Load the released task inventory used to validate source annotations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a schema array in {path}")
    labels = {str(item["role"]) for item in payload if isinstance(item, dict) and item.get("role")}
    if not labels:
        raise ValueError(f"No role labels found in {path}")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/source/input.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/output.json"))
    parser.add_argument(
        "--format",
        choices=("normalized", "cpb-bioes", "conll2009", "entity-json", "phee"),
        default="normalized",
    )
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Optional released schema used to reject records outside the task inventory.",
    )
    parser.add_argument("--sentence-field", default="sentence")
    parser.add_argument("--tokens-field", default="tokens")
    parser.add_argument("--entities-field", default="entities")
    parser.add_argument("--label-field", default="type")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--label-map", type=Path, default=None,
        help="External JSON object mapping source labels to schema labels; labels are unchanged if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.format == "cpb-bioes":
        records = convert_cpb_bioes(args.input, args.language)
    elif args.format == "conll2009":
        allowed_roles = load_schema_labels(args.schema) if args.schema else None
        records = convert_conll2009(args.input, args.language, allowed_roles)
    elif args.format == "entity-json":
        label_map: dict[str, str] = {}
        if args.label_map:
            payload = json.loads(args.label_map.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload.items()
            ):
                raise ValueError("The label map must be a JSON object with string keys and values")
            label_map = {
                str(key).lower(): str(value)
                for key, value in payload.items()
            }
        records = convert_entity_json(
            args.input,
            args.sentence_field,
            args.tokens_field,
            args.entities_field,
            args.label_field,
            args.text_field,
            label_map,
        )
    elif args.format == "phee":
        records = convert_phee(args.input)
    else:
        records = convert_normalized(args.input)
    write_json(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
