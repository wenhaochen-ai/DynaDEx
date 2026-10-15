"""Convert native annotations while retaining sentence and span coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .common import (
    character_annotation,
    find_token_offsets,
    ground_pairs,
    make_record,
    read_records,
    token_annotation,
    token_layout,
    validate_annotation,
    write_json,
)


def sentence_blocks(path: Path) -> Iterable[list[list[str]]]:
    """Read all CoNLL sentences, including an unterminated final block."""
    block: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if line.strip():
                block.append(line.split())
            elif block:
                yield block
                block = []
    if block:
        yield block


def bioes_annotations(
    labels: list[str], sentence: str, offsets: list[list[int]]
) -> list[dict[str, Any]]:
    annotations, active_role, active_start = [], None, 0
    for index, label in enumerate(labels):
        if label in {"O", "_", ""}:
            if active_role is not None:
                raise ValueError(f"Unclosed BIOES span before token {index}")
            continue
        if label in {"rel", "PRED"}:
            if active_role is not None:
                raise ValueError("Predicate interrupts a BIOES argument")
            annotations.append(token_annotation("PRED", index, index + 1, sentence, offsets))
            continue
        prefix, role = label.split("-", 1) if "-" in label else ("S", label)
        if prefix in {"B", "S"} and active_role is None:
            if prefix == "S":
                annotations.append(token_annotation(role, index, index + 1, sentence, offsets))
            else:
                active_role, active_start = role, index
        elif prefix in {"I", "E"} and active_role == role:
            if prefix == "E":
                annotations.append(
                    token_annotation(role, active_start, index + 1, sentence, offsets)
                )
                active_role = None
        else:
            raise ValueError(f"Invalid BIOES transition at token {index}: {label}")
    if active_role is not None:
        raise ValueError("Unclosed BIOES argument at end of sentence")
    return annotations


def convert_cpb_bioes(path: Path, language: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        fields = [field.rsplit("/", 2) for field in line.split()]
        if any(len(field) != 3 for field in fields):
            raise ValueError(f"Malformed CPB token on line {line_number + 1}")
        words, labels = [field[0] for field in fields], [field[2] for field in fields]
        sentence, offsets = token_layout(words, " " if language == "en" else "")
        annotations = bioes_annotations(labels, sentence, offsets)
        predicates = [item for item in annotations if item["label"] == "PRED"]
        if len(predicates) != 1:
            raise ValueError(f"CPB line {line_number + 1} must describe exactly one predicate")
        # This source format repeats predicate instances without sentence IDs.
        digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:24]
        records.append(
            make_record(
                f"{path.stem}:{line_number}",
                f"{path.stem}:{digest}",
                sentence,
                annotations,
                tokens=words,
                token_offsets=offsets,
                protocol="span-srl",
                predicate=predicates[0],
            )
        )
    return records


def convert_conll2009(
    path: Path, language: str, allowed_roles: set[str] | None
) -> list[dict[str, Any]]:
    """Expand every FILLPRED target; arguments remain singleton head tokens."""
    records = []
    for sentence_index, columns in enumerate(sentence_blocks(path)):
        if any(len(row) < 14 for row in columns):
            raise ValueError(f"Sentence {sentence_index} has fewer than 14 columns")
        if [int(row[0]) for row in columns] != list(range(1, len(columns) + 1)):
            raise ValueError("CoNLL-2009 token IDs must be contiguous and one-based")
        predicates = [index for index, row in enumerate(columns) if row[12] == "Y"]
        if any(len(row) != 14 + len(predicates) for row in columns):
            raise ValueError(f"Sentence {sentence_index}: APRED columns disagree with FILLPRED")
        words = [row[1] for row in columns]
        sentence, offsets = token_layout(words, " " if language == "en" else "")
        sentence_id = f"{path.stem}:{sentence_index}"
        for rank, predicate_index in enumerate(predicates):
            predicate = token_annotation(
                "PRED", predicate_index, predicate_index + 1, sentence, offsets
            )
            annotations = [predicate]
            for token_index, row in enumerate(columns):
                role = row[14 + rank]
                if role == "_":
                    continue
                if allowed_roles is not None and role not in allowed_roles:
                    raise ValueError(f"Schema is missing role {role!r} in {sentence_id}")
                annotations.append(
                    token_annotation(role, token_index, token_index + 1, sentence, offsets)
                )
            records.append(
                make_record(
                    f"{sentence_id}:p{predicate_index}",
                    sentence_id,
                    sentence,
                    annotations,
                    tokens=words,
                    token_offsets=offsets,
                    protocol="dependency-srl",
                    predicate=predicate,
                    meta_info={
                        "predicate_lemma": columns[predicate_index][2],
                        "predicate_sense": columns[predicate_index][13],
                        "predicate_rank": rank,
                    },
                )
            )
    return records


def bracket_annotations(
    cells: list[str], sentence: str, offsets: list[list[int]]
) -> list[dict[str, Any]]:
    """Read CoNLL proposition brackets, preserving C-/R- role labels."""
    annotations, stack = [], []
    for index, cell in enumerate(cells):
        for match in re.finditer(r"\(([^()*]+)|\)", cell):
            if match.group(1) is not None:
                stack.append((match.group(1), index))
            elif stack:
                label, start = stack.pop()
                annotations.append(
                    token_annotation(
                        "PRED" if label == "V" else label,
                        start,
                        index + 1,
                        sentence,
                        offsets,
                    )
                )
            else:
                raise ValueError(f"Unmatched proposition bracket at token {index}")
    if stack:
        raise ValueError("Unclosed proposition bracket")
    return annotations


def convert_conll2012(path: Path, language: str) -> list[dict[str, Any]]:
    records = []
    for sentence_index, columns in enumerate(sentence_blocks(path)):
        if any(len(row) < 12 for row in columns):
            raise ValueError("Expected native CoNLL-2012 columns")
        predicates = [i for i, row in enumerate(columns) if row[6] not in {"-", "_"}]
        if any(len(row) != 12 + len(predicates) for row in columns):
            raise ValueError("CoNLL-2012 predicate and argument column counts disagree")
        words = [row[3] for row in columns]
        sentence, offsets = token_layout(words, " " if language == "en" else "")
        sentence_id = f"{path.stem}:{columns[0][0]}:{columns[0][1]}:{sentence_index}"
        for rank, predicate_index in enumerate(predicates):
            annotations = bracket_annotations(
                [row[11 + rank] for row in columns], sentence, offsets
            )
            predicate = token_annotation(
                "PRED", predicate_index, predicate_index + 1, sentence, offsets
            )
            supplied = [item for item in annotations if item["label"] == "PRED"]
            if supplied and any(
                (a["start"], a["end"]) != (predicate["start"], predicate["end"]) for a in supplied
            ):
                raise ValueError("Predicate column and V span disagree")
            if not supplied:
                annotations.insert(0, predicate)
            records.append(
                make_record(
                    f"{sentence_id}:p{predicate_index}",
                    sentence_id,
                    sentence,
                    annotations,
                    tokens=words,
                    token_offsets=offsets,
                    protocol="span-srl",
                    predicate=predicate,
                    meta_info={
                        "document_id": columns[0][0],
                        "part": columns[0][1],
                        "predicate_lemma": columns[predicate_index][6],
                        "predicate_sense": columns[predicate_index][7],
                    },
                )
            )
    return records


def native_annotation(
    entity: dict[str, Any],
    label: str,
    sentence: str,
    offsets: list[list[int]],
    offset_unit: str,
    text_field: str = "text",
    **metadata: Any,
) -> dict[str, Any]:
    if "pos" in entity:
        if not isinstance(entity["pos"], list) or len(entity["pos"]) != 2:
            raise ValueError("Entity pos must be a character interval [start, end)")
        entity = {**entity, "start": entity["pos"][0], "end": entity["pos"][1]}
        offset_unit = "characters"
    if "start" in entity and "end" in entity:
        start, end = int(entity["start"]), int(entity["end"])
        if offset_unit == "tokens":
            annotation = token_annotation(label, start, end, sentence, offsets, **metadata)
        else:
            annotation = character_annotation(label, start, end, sentence, **metadata)
        text = entity.get(text_field)
        if text is not None and str(text) != annotation["text"]:
            raise ValueError(f"Native text and offsets disagree for {label!r}")
        return annotation
    text = entity.get(text_field)
    if not isinstance(text, str) or not text:
        raise ValueError(f"Missing text or offsets for {label!r}")
    grounded = ground_pairs([[label, text]], sentence, token_offsets=offsets)
    if not grounded or grounded[0].get("grounding_candidates") != 1:
        raise ValueError(f"Gold mention {text!r} requires explicit occurrence offsets")
    return {**grounded[0], **metadata}


def convert_entity_json(
    path: Path,
    sentence_field: str,
    tokens_field: str,
    entities_field: str,
    label_field: str,
    text_field: str,
    label_map: dict[str, str],
    offset_unit: str = "tokens",
) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(read_records(path)):
        tokens = [str(token) for token in item.get(tokens_field, [])] if tokens_field else []
        sentence = str(item.get(sentence_field, "")) or " ".join(tokens)
        if not sentence or entities_field not in item:
            raise ValueError(f"Record {index} is missing its sentence or {entities_field!r}")
        offsets = find_token_offsets(sentence, tokens) if tokens else []
        annotations = []
        for entity in item[entities_field]:
            label = str(entity[label_field])
            label = label_map.get(label, label_map.get(label.lower(), label))
            annotations.append(
                native_annotation(entity, label, sentence, offsets, offset_unit, text_field)
            )
        record_id = str(item.get("id", item.get("wnd_id", f"{path.stem}:{index}")))
        records.append(
            make_record(
                record_id,
                str(item.get("sentence_id", record_id)),
                sentence,
                annotations,
                tokens=tokens or None,
                token_offsets=offsets or None,
            )
        )
    return records


def convert_phee(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(read_records(path)):
        sentence, tokens = str(item["text"]), list(item["tokens"])
        offsets = find_token_offsets(sentence, tokens)
        annotations, events = [], []
        for event_index, event in enumerate(item["event_mentions"]):
            event_id = str(event.get("id", event_index))
            trigger = native_annotation(
                event["trigger"], "Trigger", sentence, offsets, "tokens", event_id=event_id
            )
            arguments = [
                native_annotation(a, str(a["role"]), sentence, offsets, "tokens", event_id=event_id)
                for a in event.get("arguments", [])
            ]
            annotations.extend([trigger, *arguments])
            events.append(
                {
                    "id": event_id,
                    "event_type": event.get("event_type"),
                    "trigger": trigger,
                    "arguments": arguments,
                }
            )
        record_id = str(item.get("wnd_id", item.get("id", f"{path.stem}:{index}")))
        record = make_record(
            record_id,
            record_id,
            sentence,
            annotations,
            tokens=tokens,
            token_offsets=offsets,
            protocol="phee",
        )
        record["events"] = events
        records.append(record)
    return records


def convert_framenet(path: Path) -> list[dict[str, Any]]:
    """Read full-text XML with provided targets and semantic frames."""
    files = sorted(path.rglob("*.xml")) if path.is_dir() else [path]
    records = []
    for source in files:
        for element in ET.parse(source).getroot().iter():
            if element.tag.rsplit("}", 1)[-1] != "sentence":
                continue
            text = next(
                (child for child in element if child.tag.rsplit("}", 1)[-1] == "text"), None
            )
            if text is None or text.text is None:
                raise ValueError("FrameNet sentence is missing text")
            sentence, sentence_id = text.text, f"{source.name}:{element.attrib['ID']}"
            for annotation_set in element:
                frame = annotation_set.get("frameName")
                if not frame:
                    continue
                targets, arguments = [], []
                for layer in annotation_set:
                    name = layer.get("name")
                    if name not in {"Target", "FE"} or (
                        name == "FE" and layer.get("rank", "1") != "1"
                    ):
                        continue
                    for label in layer:
                        if "start" not in label.attrib or "end" not in label.attrib:
                            continue  # Null instantiations have no text interval.
                        annotation = character_annotation(
                            "PRED" if name == "Target" else str(label.attrib["name"]),
                            int(label.attrib["start"]),
                            int(label.attrib["end"]) + 1,
                            sentence,
                        )
                        (targets if name == "Target" else arguments).append(annotation)
                if not targets:
                    raise ValueError(
                        f"FrameNet annotation set {annotation_set.get('ID')} has no target"
                    )
                predicate = character_annotation(
                    "PRED",
                    min(a["start"] for a in targets),
                    max(a["end"] for a in targets),
                    sentence,
                    segments=[
                        [a["start"], a["end"]] for a in sorted(targets, key=lambda a: a["start"])
                    ],
                )
                records.append(
                    make_record(
                        f"{sentence_id}:a{annotation_set.attrib['ID']}",
                        sentence_id,
                        sentence,
                        [predicate, *arguments],
                        protocol="framenet",
                        predicate=predicate,
                        meta_info={
                            "frame": frame,
                            "lexical_unit": annotation_set.get("luName", ""),
                            "source_document": source.name,
                            "native_sentence_id": element.attrib["ID"],
                            "annotation_set_id": annotation_set.attrib["ID"],
                        },
                    )
                )
    return records


def convert_normalized(path: Path) -> list[dict[str, Any]]:
    records = read_records(path)
    for record in records:
        if not all(key in record for key in ("id", "sentence_id", "sentence", "annotations")):
            raise ValueError(
                "Normalized records require IDs, sentence_id, and coordinate-bearing annotations"
            )
        for annotation in record["annotations"]:
            validate_annotation(annotation, record)
        if record.get("predicate") is not None:
            validate_annotation(record["predicate"], record)
        record["roles"] = [[a["label"], a["text"]] for a in record["annotations"]]
    return records


def load_schema_labels(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a non-empty schema array")
    return {"PRED" if item["role"] == "rel" else str(item["role"]) for item in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=(
            "normalized",
            "cpb-bioes",
            "conll2009",
            "conll2012",
            "entity-json",
            "phee",
            "framenet",
        ),
        required=True,
    )
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--sentence-field", default="sentence")
    parser.add_argument("--tokens-field", default="tokens")
    parser.add_argument("--entities-field", default="entities")
    parser.add_argument("--label-field", default="type")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--offset-unit", choices=("tokens", "characters"), default="tokens")
    parser.add_argument("--label-map", type=Path)
    args = parser.parse_args()
    if args.format == "cpb-bioes":
        records = convert_cpb_bioes(args.input, args.language)
    elif args.format == "conll2009":
        records = convert_conll2009(
            args.input, args.language, load_schema_labels(args.schema) if args.schema else None
        )
    elif args.format == "conll2012":
        records = convert_conll2012(args.input, args.language)
    elif args.format == "framenet":
        records = convert_framenet(args.input)
    elif args.format == "phee":
        records = convert_phee(args.input)
    elif args.format == "entity-json":
        label_map = json.loads(args.label_map.read_text(encoding="utf-8")) if args.label_map else {}
        if not isinstance(label_map, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in label_map.items()
        ):
            raise ValueError("Label maps must contain string keys and values")
        records = convert_entity_json(
            args.input,
            args.sentence_field,
            args.tokens_field,
            args.entities_field,
            args.label_field,
            args.text_field,
            label_map,
            args.offset_unit,
        )
    else:
        records = convert_normalized(args.input)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Converted record IDs must be unique")
    write_json(args.output, records)
    report = {
        "format": args.format,
        "records": len(records),
        "sentences": len({r["sentence_id"] for r in records}),
        "arguments": sum(sum(a["label"] != "PRED" for a in r["annotations"]) for r in records),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()
        if args.input.is_file()
        else None,
    }
    write_json(args.output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
