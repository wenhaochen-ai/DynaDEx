"""Shared record I/O, span coordinates, and label/text serialization."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, JSON Lines file, or concatenated JSON objects."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
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


PREDICATE_LABELS = {"PRED", "rel"}


def token_layout(tokens: list[str], separator: str = " ") -> tuple[str, list[list[int]]]:
    offsets, cursor = [], 0
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("Tokens must be non-empty strings")
        offsets.append([cursor, cursor + len(token)])
        cursor += len(token) + len(separator)
    return separator.join(tokens), offsets


def find_token_offsets(sentence: str, tokens: list[str]) -> list[list[int]]:
    """Align native tokens to their surface occurrences, preserving whitespace."""
    offsets, cursor = [], 0
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("Tokens must be non-empty strings")
        start = sentence.find(token, cursor)
        if start < 0 or sentence[cursor:start].strip():
            raise ValueError(f"Cannot align token {token!r} at character {cursor}")
        offsets.append([start, start + len(token)])
        cursor = start + len(token)
    if sentence[cursor:].strip():
        raise ValueError("Native tokens do not cover the complete sentence")
    return offsets


def token_annotation(
    label: str,
    start: int,
    end: int,
    sentence: str,
    offsets: list[list[int]],
    **metadata: Any,
) -> dict[str, Any]:
    if not 0 <= start < end <= len(offsets):
        raise ValueError(f"Invalid token interval [{start}, {end})")
    char_start, char_end = offsets[start][0], offsets[end - 1][1]
    return {
        **metadata,
        "label": label,
        "text": sentence[char_start:char_end],
        "start": char_start,
        "end": char_end,
        "token_start": start,
        "token_end": end,
    }


def character_annotation(
    label: str,
    start: int,
    end: int,
    sentence: str,
    **metadata: Any,
) -> dict[str, Any]:
    if not 0 <= start < end <= len(sentence):
        raise ValueError(f"Invalid character interval [{start}, {end})")
    return {**metadata, "label": label, "text": sentence[start:end], "start": start, "end": end}


def validate_annotation(annotation: dict[str, Any], record: dict[str, Any]) -> None:
    sentence = record["sentence"]
    label = annotation.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("Every annotation requires a non-empty label")
    start, end = annotation.get("start"), annotation.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sentence):
        raise ValueError(f"Annotation {label!r} requires valid character coordinates")
    if sentence[start:end] != annotation.get("text"):
        raise ValueError(f"Text and coordinates disagree for annotation {label!r}")
    if "segments" in annotation:
        previous_end = start
        for segment in annotation["segments"]:
            if (
                not isinstance(segment, (list, tuple))
                or len(segment) != 2
                or any(type(value) is not int for value in segment)
                or not previous_end <= segment[0] < segment[1] <= end
            ):
                raise ValueError("Predicate segments must be ordered intervals within its bounds")
            previous_end = segment[1]
    if "token_start" in annotation or "token_end" in annotation:
        token_start, token_end = annotation.get("token_start"), annotation.get("token_end")
        offsets = record.get("token_offsets", [])
        if (
            type(token_start) is not int
            or type(token_end) is not int
            or not 0 <= token_start < token_end <= len(offsets)
        ):
            raise ValueError("Invalid token coordinates")
        if (offsets[token_start][0], offsets[token_end - 1][1]) != (start, end):
            raise ValueError("Character and token coordinates disagree")


def make_record(
    record_id: str,
    sentence_id: str,
    sentence: str,
    annotations: list[dict[str, Any]],
    *,
    tokens: list[str] | None = None,
    token_offsets: list[list[int]] | None = None,
    protocol: str = "span",
    predicate: dict[str, Any] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": record_id,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "protocol": protocol,
        "annotations": annotations,
        "roles": [[item["label"], item["text"]] for item in annotations],
    }
    if tokens is not None:
        record["tokens"] = tokens
        record["token_offsets"] = (
            token_offsets if token_offsets is not None else find_token_offsets(sentence, tokens)
        )
        if len(record["token_offsets"]) != len(tokens):
            raise ValueError("Token offsets must cover every source token")
        for token, (start, end) in zip(tokens, record["token_offsets"], strict=True):
            if sentence[start:end] != token:
                raise ValueError("Source tokens and character offsets disagree")
    if predicate is not None:
        record["predicate"] = predicate
    if meta_info:
        record["meta_info"] = meta_info
    for annotation in annotations:
        validate_annotation(annotation, record)
    if predicate is not None:
        validate_annotation(predicate, record)
    return record


def ground_pairs(
    pairs: list[list[str]],
    sentence: str,
    *,
    token_offsets: list[list[int]] | None = None,
    dependency: bool = False,
) -> list[dict[str, Any]]:
    """Ground repeated pairs left to right for each label/text, without gold spans.

    Unmatched spans remain predictions with null coordinates and count as false
    positives. Repeated text at different positions is retained. When a text has
    multiple occurrences, generation order selects successive occurrences; the
    number of alternatives is recorded for inspection.
    """
    used: Counter[tuple[str, str]] = Counter()
    grounded: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    offsets = token_offsets or []
    boundaries = {(start, end): index for index, (start, end) in enumerate(offsets)}
    token_starts = {start: index for index, (start, _) in enumerate(offsets)}
    token_ends = {end: index + 1 for index, (_, end) in enumerate(offsets)}
    for label, text in pairs:
        candidates, cursor = [], 0
        while text and (start := sentence.find(text, cursor)) >= 0:
            end = start + len(text)
            if (
                not dependency and (not offsets or (start in token_starts and end in token_ends))
            ) or (dependency and (start, end) in boundaries):
                candidates.append((start, end))
            cursor = start + 1
        occurrence = used[(label, text)]
        used[(label, text)] += 1
        annotation: dict[str, Any] = {"label": label, "text": text, "start": None, "end": None}
        if candidates:
            if len(candidates) == 1:
                occurrence = 0
            if occurrence < len(candidates):
                start, end = candidates[occurrence]
                annotation.update(start=start, end=end, grounding_candidates=len(candidates))
                if offsets:
                    annotation.update(token_start=token_starts[start], token_end=token_ends[end])
        key = (label, annotation["start"], annotation["end"], text)
        if key not in seen:
            grounded.append(annotation)
            seen.add(key)
    return grounded


def sentence_group(record: dict[str, Any]) -> str:
    """Return the stable source sentence identifier used for annotation budgets."""
    value = record.get("sentence_id")
    if value is None or str(value) == "":
        raise ValueError(f"Record {record.get('id')} is missing its source sentence_id")
    return str(value)


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def format_pairs(roles: list[list[str]]) -> str:
    return " ".join(f"({label}, {escape_text(text)})" for label, text in roles)


def parse_pairs(response: str) -> tuple[list[list[str]], list[str]]:
    """Read generated pairs; preserve invalid labels/text for downstream scoring."""
    cleaned = THINK_PATTERN.sub("", response)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    pairs, errors, cursor = [], [], 0
    while cursor < len(cleaned):
        opening = cleaned.find("(", cursor)
        if opening < 0:
            break
        comma = cleaned.find(",", opening + 1)
        if comma < 0 or any(char in cleaned[opening + 1 : comma] for char in "()\n"):
            errors.append(cleaned[opening:])
            cursor = opening + 1
            continue
        label, text, depth = cleaned[opening + 1 : comma].strip(), [], 1
        cursor = comma + 1
        while cursor < len(cleaned) and depth:
            char = cleaned[cursor]
            if char == "\\" and cursor + 1 < len(cleaned) and cleaned[cursor + 1] in "\\()":
                cursor += 1
                text.append(cleaned[cursor])
            elif char == "(":
                depth += 1
                text.append(char)
            elif char == ")":
                depth -= 1
                if depth:
                    text.append(char)
            else:
                text.append(char)
            cursor += 1
        span = "".join(text).strip()
        if depth or not label or not span:
            errors.append(cleaned[opening:cursor])
        else:
            pairs.append([label, span])
    return pairs, errors
