"""Compute extraction metrics, export benchmark files, and run native scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from preprocessing.common import (
    PREDICATE_LABELS,
    ground_pairs,
    read_records,
    validate_annotation,
    write_json,
)
from preprocessing.prepare_datasets import sentence_blocks


def normalize_roles(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Role annotations must be an array")
    result = []
    for item in value:
        if isinstance(item, dict):
            label, text = item.get("label", item.get("role")), item.get("text", item.get("span"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            label, text = item
        else:
            raise ValueError("Expected label/text annotations")
        if not isinstance(label, str) or not isinstance(text, str):
            raise ValueError("Annotation labels and text must be strings")
        result.append((label.strip(), text.strip()))
    return result


def record_roles(record: dict[str, Any], prediction: bool) -> list[tuple[str, str]]:
    keys = ("predicted_roles", "roles") if prediction else ("roles", "gold_roles")
    for key in keys:
        if key in record:
            return normalize_roles(record[key])
    raise ValueError("Record is missing role annotations")


def align_records(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(gold) != len(predictions):
        raise ValueError("Gold and prediction files have different record counts")
    indices = []
    for name, records in (("Gold", gold), ("Prediction", predictions)):
        if any(record.get("id") is None for record in records):
            raise ValueError(f"{name} records require stable IDs")
        index = {str(record["id"]): record for record in records}
        if len(index) != len(records):
            raise ValueError(f"{name} record IDs are not unique")
        indices.append(index)
    if set(indices[0]) != set(indices[1]):
        raise ValueError("Gold and prediction record ID sets differ")
    return [(record, indices[1][str(record["id"])]) for record in gold]


def prediction_annotations(
    gold: dict[str, Any], prediction: dict[str, Any]
) -> list[dict[str, Any]]:
    if "predicted_annotations" in prediction:
        return prediction["predicted_annotations"]
    # Only input text and tokenization participate in grounding.
    return ground_pairs(
        [list(pair) for pair in record_roles(prediction, True)],
        gold["sentence"],
        token_offsets=gold.get("token_offsets"),
        dependency=gold.get("protocol") == "dependency-srl",
    )


def positional_items(
    record: dict[str, Any], annotations: list[dict[str, Any]], mode: str, *, gold: bool
) -> set[tuple[Any, ...]]:
    predicate = record.get("predicate")
    if mode == "srl":
        if predicate is None:
            raise ValueError("SRL records require an explicit target predicate")
        validate_annotation(predicate, record)
        target = (
            predicate["start"],
            predicate["end"],
            tuple(map(tuple, predicate.get("segments", []))),
        )
    else:
        target = ()
    items = set()
    for index, annotation in enumerate(annotations):
        label = annotation.get("label", "")
        if mode == "srl" and label in PREDICATE_LABELS:
            continue  # Given target locations are inputs, as in proposition scoring.
        try:
            validate_annotation(annotation, record)
            if record.get("protocol") == "dependency-srl":
                if annotation.get("token_end", -1) - annotation.get("token_start", -1) != 1:
                    raise ValueError("Dependency arguments must identify one head token")
            key = (label, target, annotation["start"], annotation["end"])
        except (ValueError, KeyError, TypeError):
            if gold:
                raise
            key = (label, "invalid-prediction", index, annotation.get("text", ""))
        items.add(key)
    return items


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_records(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], mode: str = "auto"
) -> dict[str, Any]:
    if mode not in {"auto", "pair", "srl", "text-pair", "text-srl"}:
        raise ValueError(f"Unknown matching mode: {mode}")
    totals: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    exact_ids = []
    for gold_record, prediction in align_records(gold, predictions):
        selected_mode = mode
        if mode == "auto":
            selected_mode = "srl" if gold_record.get("predicate") is not None else "pair"
        if prediction.get("sentence", gold_record["sentence"]) != gold_record["sentence"]:
            raise ValueError("Prediction and gold sentences disagree for the same ID")
        if prediction.get("predicate") is not None:
            predicted_target = prediction["predicate"]
            expected_target = gold_record.get("predicate", {})
            if any(
                predicted_target.get(key) != expected_target.get(key)
                for key in ("start", "end", "segments")
            ):
                raise ValueError("Prediction and input target predicate disagree")
        if mode.startswith("text-"):
            gold_roles, predicted_roles = (
                record_roles(gold_record, False),
                record_roles(prediction, True),
            )

            def text_items(roles: list[tuple[str, str]]) -> set[tuple[Any, ...]]:
                predicate = tuple(text for label, text in roles if label in PREDICATE_LABELS)
                return {
                    (label, predicate if mode == "text-srl" else (), text)
                    for label, text in roles
                    if mode != "text-srl" or label not in PREDICATE_LABELS
                }

            gold_items, predicted_items = text_items(gold_roles), text_items(predicted_roles)
        else:
            if "annotations" not in gold_record:
                raise ValueError("Positional scoring requires native annotation coordinates")
            if selected_mode == "pair" and gold_record.get("predicate") is not None:
                raise ValueError("Use srl or auto for predicate-indexed records")
            gold_items = positional_items(
                gold_record, gold_record["annotations"], selected_mode, gold=True
            )
            predicted_items = positional_items(
                gold_record,
                prediction_annotations(gold_record, prediction),
                selected_mode,
                gold=False,
            )
        for index, error in enumerate(prediction.get("parse_errors", [])):
            predicted_items.add(("<parse-error>", index, str(error)))
        common, excess, missed = (
            gold_items & predicted_items,
            predicted_items - gold_items,
            gold_items - predicted_items,
        )
        totals.update(tp=len(common), fp=len(excess), fn=len(missed))
        for key, items in (("tp", common), ("fp", excess), ("fn", missed)):
            for item in items:
                per_label[item[0]][key] += 1
        if gold_items == predicted_items:
            exact_ids.append(str(gold_record["id"]))
    return {
        **prf(totals["tp"], totals["fp"], totals["fn"]),
        "records": len(gold),
        "matching": mode,
        "exact_records": len(exact_ids),
        "exact_record_ids": exact_ids,
        "per_label": {
            label: prf(values["tp"], values["fp"], values["fn"])
            for label, values in sorted(per_label.items())
        },
    }


def checked_annotations(record: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    if prediction.get("parse_errors"):
        raise ValueError(f"Record {record['id']} has malformed generated pairs; inspect raw_output")
    annotations = [
        item
        for item in prediction_annotations(record, prediction)
        if item["label"] not in PREDICATE_LABELS
    ]
    for annotation in annotations:
        validate_annotation(annotation, record)
    return annotations


def export_conll2009(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    source: Path,
    output: Path,
    sense_mode: str,
) -> None:
    """Write every source sentence and APRED column in its original order.

    The sense setting is explicit: ``predicted`` requires an independently
    predicted sense; ``provided`` evaluates a declared given-sense setting.
    """
    if sense_mode not in {"predicted", "provided"}:
        raise ValueError("Choose an explicit predicate sense setting")
    by_target = {}
    for record, prediction in align_records(gold, predictions):
        if record.get("protocol") != "dependency-srl":
            raise ValueError("CoNLL-2009 export requires dependency-srl records")
        key = (record["sentence_id"], record["predicate"]["token_start"])
        if key in by_target:
            raise ValueError("Duplicate predicate instance")
        by_target[key] = (record, prediction)
    lines, covered = [], set()
    for sentence_index, columns in enumerate(sentence_blocks(source)):
        sentence_id = f"{source.stem}:{sentence_index}"
        predicate_indices = [index for index, row in enumerate(columns) if row[12] == "Y"]
        rows = [row[:14] + ["_"] * len(predicate_indices) for row in columns]
        for row in rows:
            row[8] = row[10] = row[13] = "_"  # SRL-only output has no predicted syntax.
        for rank, predicate_index in enumerate(predicate_indices):
            key = (sentence_id, predicate_index)
            if key not in by_target:
                raise ValueError(f"Missing source predicate {key}")
            record, prediction = by_target[key]
            if [row[1] for row in columns] != record.get("tokens"):
                raise ValueError("Native source tokens and normalized tokens differ")
            covered.add(key)
            if sense_mode == "provided":
                sense = columns[predicate_index][13]
            else:
                sense = prediction.get("predicted_sense")
                if not isinstance(sense, str) or not sense or sense == "_":
                    raise ValueError(
                        "Predicted-sense scoring requires predicted_sense for every predicate"
                    )
            if any(char.isspace() for char in sense):
                raise ValueError("A predicate sense must occupy one CoNLL column")
            rows[predicate_index][13] = sense
            for annotation in checked_annotations(record, prediction):
                start, end = annotation.get("token_start"), annotation.get("token_end")
                if start is None or end != start + 1:
                    raise ValueError("Dependency predictions must identify one head token")
                label = annotation["label"]
                if any(char.isspace() for char in label) or label == "_":
                    raise ValueError("Invalid CoNLL argument label")
                existing = rows[start][14 + rank]
                if existing not in {"_", label}:
                    raise ValueError("Conflicting argument labels at one dependency head")
                rows[start][14 + rank] = label
        lines.extend("\t".join(row) for row in rows)
        lines.append("")
    if covered != set(by_target):
        raise ValueError("Normalized records contain predicates absent from the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def proposition_column(
    annotations: list[dict[str, Any]], count: int, predicate: dict[str, Any]
) -> list[str]:
    spans = [(a["token_start"], a["token_end"], a["label"]) for a in annotations]
    spans.append((predicate["token_start"], predicate["token_end"], "V"))
    for start, end, label in spans:
        if not 0 <= start < end <= count or any(char in label for char in "()*\t\n "):
            raise ValueError("Invalid proposition span or label")
        for other_start, other_end, _ in spans:
            if start < other_start < end < other_end:
                raise ValueError("Crossing spans cannot be represented in proposition brackets")
    cells = []
    for index in range(count):
        opening = sorted(
            (span for span in spans if span[0] == index), key=lambda span: (-span[1], span[2])
        )
        closing = [span for span in spans if span[1] == index + 1]
        cells.append("".join(f"({label}" for _, _, label in opening) + "*" + ")" * len(closing))
    return cells


def export_props(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], output: Path
) -> None:
    sentences: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, prediction in align_records(gold, predictions):
        if record.get("protocol") != "span-srl":
            raise ValueError("Proposition export requires span-srl records")
        sentences[record["sentence_id"]].append((record, prediction))
    lines = []
    for instances in sentences.values():
        instances.sort(key=lambda pair: pair[0]["predicate"]["token_start"])
        words = instances[0][0]["tokens"]
        columns, predicate_forms = [], ["-"] * len(words)
        for record, prediction in instances:
            if record["tokens"] != words:
                raise ValueError("Predicate instances disagree on source tokenization")
            predicate = record["predicate"]
            if predicate_forms[predicate["token_start"]] != "-":
                raise ValueError("Duplicate predicate location")
            predicate_forms[predicate["token_start"]] = words[predicate["token_start"]]
            columns.append(
                proposition_column(checked_annotations(record, prediction), len(words), predicate)
            )
        for index in range(len(words)):
            lines.append(
                "\t".join([predicate_forms[index], *(column[index] for column in columns)])
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_framenet(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], source: Path, output: Path
) -> None:
    documents: dict[str, dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for record, prediction in align_records(gold, predictions):
        if record.get("protocol") != "framenet":
            raise ValueError("FrameNet XML export requires framenet records")
        meta = record["meta_info"]
        documents[meta["source_document"]][
            (meta["native_sentence_id"], meta["annotation_set_id"])
        ] = (record, prediction)
    output.mkdir(parents=True, exist_ok=True)
    for name, targets in documents.items():
        if Path(name).name != name:
            raise ValueError("FrameNet document references must be basenames")
        tree = ET.parse(source / name)
        covered = set()
        for element in tree.getroot().iter():
            if element.tag.rsplit("}", 1)[-1] != "sentence":
                continue
            for annotation_set in list(element):
                if annotation_set.tag.rsplit("}", 1)[
                    -1
                ] != "annotationSet" or not annotation_set.get("frameName"):
                    continue
                key = (element.attrib["ID"], annotation_set.attrib["ID"])
                if key not in targets:
                    raise ValueError(
                        "Source XML contains a frame instance missing from the evaluation split"
                    )
                record, prediction = targets[key]
                covered.add(key)
                # Targets and frames are provided; all predicted FE layers are rebuilt.
                for layer in list(annotation_set):
                    if layer.get("name") != "Target":
                        annotation_set.remove(layer)
                namespace = (
                    annotation_set.tag.rsplit("}", 1)[0] + "}" if "}" in annotation_set.tag else ""
                )
                layer = ET.SubElement(
                    annotation_set, namespace + "layer", {"name": "FE", "rank": "1"}
                )
                for annotation in checked_annotations(record, prediction):
                    ET.SubElement(
                        layer,
                        namespace + "label",
                        {
                            "name": annotation["label"],
                            "start": str(annotation["start"]),
                            "end": str(annotation["end"] - 1),
                        },
                    )
        if covered != set(targets):
            raise ValueError("FrameNet source XML is missing expected frame instances")
        tree.write(output / name, encoding="utf-8", xml_declaration=True)


def export_span_json(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], output: Path
) -> None:
    """Write positional label/span records for community evaluator adapters."""
    records = []
    for record, prediction in align_records(gold, predictions):
        annotations = checked_annotations(record, prediction)
        records.append(
            {
                "id": record["id"],
                "sentence_id": record["sentence_id"],
                "sentence": record["sentence"],
                "tokens": record.get("tokens"),
                "annotations": annotations,
            }
        )
    write_json(output, records)


def _score_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Native export: python -m src.evaluate export --help; benchmark scoring: python -m src.evaluate official --help.",
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("auto", "pair", "srl", "text-pair", "text-srl"), default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = score_records(read_records(args.gold), read_records(args.predictions), args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metrics[key] for key in ("precision", "recall", "f1")}, indent=2))


def _export_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate export",
        description="Export positional predictions to benchmark evaluator input formats.",
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--format", choices=("conll2009", "props", "framenet", "span-json"), required=True
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sense-mode", choices=("provided", "predicted"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gold = read_records(args.gold)
    predictions = (
        read_records(args.predictions)
        if args.predictions
        else [
            {"id": record["id"], "predicted_annotations": record["annotations"]} for record in gold
        ]
    )
    if args.format == "conll2009":
        if args.source is None or args.sense_mode is None:
            parser.error("CoNLL-2009 requires --source and an explicit --sense-mode")
        export_conll2009(gold, predictions, args.source, args.output, args.sense_mode)
    elif args.format == "props":
        export_props(gold, predictions, args.output)
    elif args.format == "framenet":
        if args.source is None:
            parser.error("FrameNet XML export requires --source")
        export_framenet(gold, predictions, args.source, args.output)
    else:
        export_span_json(gold, predictions, args.output)


def _official_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate official",
        description="Run a supplied benchmark scorer against native gold and system files.",
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--protocol", choices=("conll2009", "props", "custom"), required=True)
    parser.add_argument(
        "--arguments",
        type=Path,
        help="JSON argument array for custom scorers; supports {gold}, {predictions}, {scorer}",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.gold, args.predictions, args.scorer):
        if not path.exists():
            parser.error(f"Required scorer input does not exist: {path.name}")
    if args.protocol == "conll2009":
        command = ["perl", str(args.scorer), "-g", str(args.gold), "-s", str(args.predictions)]
    elif args.protocol == "props":
        command = ["perl", str(args.scorer), str(args.gold), str(args.predictions)]
    else:
        if args.arguments is None:
            parser.error("Custom scorers require --arguments")
        command = json.loads(args.arguments.read_text(encoding="utf-8"))
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) for item in command)
        ):
            parser.error("Custom scorer arguments must be a non-empty string array")
        replacements = {
            "{gold}": str(args.gold),
            "{predictions}": str(args.predictions),
            "{scorer}": str(args.scorer),
        }
        command = [item for item in command]
        for index, item in enumerate(command):
            for placeholder, value in replacements.items():
                item = item.replace(placeholder, value)
            command[index] = item
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        args.output.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "protocol": args.protocol,
        "scorer_sha256": hashlib.sha256(args.scorer.read_bytes()).hexdigest(),
        "returncode": completed.returncode,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(completed.stdout, end="")
    completed.check_returncode()


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "export":
        _export_main(arguments[1:])
    elif arguments and arguments[0] == "official":
        _official_main(arguments[1:])
    else:
        _score_main(arguments)


if __name__ == "__main__":
    main()
