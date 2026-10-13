"""Compute exact micro metrics under pair or predicate-role-span matching."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable


PREDICATE_LABELS = {"PRED", "rel"}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read prediction or gold records from JSON or JSON Lines."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_roles(value: Iterable[Any]) -> list[tuple[str, str]]:
    """Normalize dictionary or sequence role annotations."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("Role annotations must be an array")
    roles: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            label = item.get("label", item.get("role", item.get("type")))
            span = item.get("span", item.get("text", item.get("name")))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            label, span = item[0], item[1]
        else:
            raise ValueError("Each annotation must contain a label and a span")
        if not isinstance(label, str) or not isinstance(span, str):
            raise ValueError("Annotation labels and spans must be strings")
        label, span = label.strip(), span.strip()
        if not label or not span:
            raise ValueError("Annotation labels and spans must be non-empty")
        roles.append((label, span))
    return roles


def record_roles(record: dict[str, Any], prediction: bool) -> list[tuple[str, str]]:
    """Select the prediction or gold role field from a record."""
    keys = ("predicted_roles", "roles") if prediction else ("roles", "gold_roles")
    for key in keys:
        if key in record:
            return normalize_roles(record[key])
    raise ValueError(f"Record is missing an annotation field: {keys}")


def evaluation_items(
    roles: list[tuple[str, str]], mode: str
) -> list[tuple[Hashable, str]]:
    """Return comparable items paired with their reporting label."""
    if mode == "pair":
        return [((label, span), label) for label, span in roles]
    if mode != "srl":
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    predicates = [span for label, span in roles if label in PREDICATE_LABELS]
    if len(predicates) > 1:
        raise ValueError("SRL evaluation requires at most one predicate per record")
    predicate = predicates[0] if predicates else ""
    return [
        ((predicate, label, span), label)
        for label, span in roles
        if label not in PREDICATE_LABELS
    ]


def align_records(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate record coverage and align by IDs, or by order for files without IDs."""
    if len(gold) != len(predictions):
        raise ValueError("Gold and prediction files have different record counts")
    gold_ids = [None if item.get("id") is None else str(item["id"]) for item in gold]
    prediction_ids = [None if item.get("id") is None else str(item["id"]) for item in predictions]
    for name, ids in (("Gold", gold_ids), ("Prediction", prediction_ids)):
        present = [record_id for record_id in ids if record_id is not None]
        if present and len(present) != len(ids):
            raise ValueError(f"{name} records contain a mixture of present and missing IDs")
        if len(set(present)) != len(present):
            raise ValueError(f"{name} IDs are not unique")
    if gold_ids and all(item is not None for item in gold_ids) and all(
        item is not None for item in prediction_ids
    ):
        if set(gold_ids) != set(prediction_ids):
            raise ValueError("Gold and prediction files have different record ID sets")
        prediction_map = {str(item["id"]): item for item in predictions}
        return [(item, prediction_map[str(item["id"])]) for item in gold]
    return list(zip(gold, predictions))


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Compute precision, recall, and F1 from integer counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_records(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Compute exact multiset micro scores and per-label statistics."""
    totals = Counter({"tp": 0, "fp": 0, "fn": 0})
    label_totals: dict[str, Counter[str]] = defaultdict(Counter)
    exact_record_ids: list[str] = []
    for index, (gold_record, prediction_record) in enumerate(
        align_records(gold, predictions)
    ):
        gold_roles = record_roles(gold_record, False)
        if mode == "srl" and sum(label in PREDICATE_LABELS for label, _ in gold_roles) != 1:
            raise ValueError("SRL gold records must contain exactly one predicate")
        gold_items = evaluation_items(gold_roles, mode)
        predicted_items = evaluation_items(record_roles(prediction_record, True), mode)
        gold_counter = Counter(item for item, _ in gold_items)
        predicted_counter = Counter(item for item, _ in predicted_items)
        common = gold_counter & predicted_counter
        tp = sum(common.values())
        fp = sum((predicted_counter - gold_counter).values())
        fn = sum((gold_counter - predicted_counter).values())
        totals.update({"tp": tp, "fp": fp, "fn": fn})
        if gold_counter == predicted_counter:
            exact_record_ids.append(str(gold_record.get("id", index)))

        labels = {label for _, label in gold_items + predicted_items}
        for label in labels:
            gold_label = Counter(item for item, item_label in gold_items if item_label == label)
            predicted_label = Counter(
                item for item, item_label in predicted_items if item_label == label
            )
            label_totals[label]["tp"] += sum((gold_label & predicted_label).values())
            label_totals[label]["fp"] += sum((predicted_label - gold_label).values())
            label_totals[label]["fn"] += sum((gold_label - predicted_label).values())

    result = prf(totals["tp"], totals["fp"], totals["fn"])
    result["records"] = len(gold)
    result["exact_records"] = len(exact_record_ids)
    result["exact_record_ids"] = exact_record_ids
    result["per_label"] = {
        label: prf(counts["tp"], counts["fp"], counts["fn"])
        for label, counts in sorted(label_totals.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions.json"))
    parser.add_argument("--mode", choices=("pair", "srl"), default="pair")
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()

    metrics = score_records(
        read_records(args.gold),
        read_records(args.predictions),
        args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("precision", "recall", "f1")}, indent=2))


if __name__ == "__main__":
    main()
