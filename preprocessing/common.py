"""Shared I/O and normalization helpers for preprocessing scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, JSON Lines file, or concatenated JSON objects."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return payload

    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        item, position = decoder.raw_decode(text, position)
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON objects in {path}")
        records.append(item)
    return records


def write_json(path: Path, payload: Any) -> None:
    """Write indented UTF-8 JSON and create the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 JSON Lines and create the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_roles(raw_roles: Iterable[Any]) -> list[list[str]]:
    """Normalize role annotations to ordered ``[label, span]`` pairs."""
    roles: list[list[str]] = []
    for role in raw_roles:
        if isinstance(role, dict):
            label = role.get("label", role.get("role", role.get("type")))
            span = role.get("span", role.get("text", role.get("name")))
        elif isinstance(role, (list, tuple)) and len(role) >= 2:
            label, span = role[0], role[1]
        else:
            continue
        if label is None or span is None:
            continue
        label_text = str(label).strip()
        span_text = str(span).strip()
        if label_text and span_text:
            roles.append([label_text, span_text])
    return roles


def normalized_record(
    sentence: str,
    roles: Iterable[Any],
    record_id: str,
    meta_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the common intermediate representation used by all datasets."""
    record: dict[str, Any] = {
        "id": record_id,
        "sentence": sentence.strip(),
        "roles": normalize_roles(roles),
    }
    if meta_info:
        record["meta_info"] = meta_info
    return record
