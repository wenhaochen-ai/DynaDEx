"""Sample nested budgets of complete source sentences."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .common import sentence_group


def select_sentences(
    records: list[dict[str, Any]],
    maximum_sentences: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    groups = sorted({sentence_group(record) for record in records})
    if maximum_sentences is not None and maximum_sentences > 0:
        random.Random(seed).shuffle(groups)
        groups = groups[:maximum_sentences]
    selected = set(groups)
    return [record for record in records if sentence_group(record) in selected]


def subset_manifest(
    records: list[dict[str, Any]], seed: int, requested: int | None
) -> dict[str, Any]:
    ids = sorted({sentence_group(record) for record in records})
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "sampling_seed": seed,
        "requested_sentences": requested,
        "actual_sentences": len(ids),
        "predicate_or_sentence_records": len(records),
        "sentence_ids": ids,
        "sentence_ids_sha256": hashlib.sha256(payload).hexdigest(),
    }
