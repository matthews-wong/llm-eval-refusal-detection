"""Load labelled response fixtures from JSONL."""

import json
from dataclasses import dataclass

from .detector import LABELS

BENIGN = "benign"
HARMFUL = "harmful"
PROMPT_KINDS = (BENIGN, HARMFUL)

REQUIRED_FIELDS = ("id", "prompt_kind", "response", "finish_reason", "label")


@dataclass(frozen=True)
class Record:
    id: str
    prompt_kind: str
    response: str
    finish_reason: str
    label: str


def parse_record(raw: dict, where: str) -> Record:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"{where}: missing field(s) {missing}")
    if raw["prompt_kind"] not in PROMPT_KINDS:
        raise ValueError(f"{where}: prompt_kind {raw['prompt_kind']!r} not in {PROMPT_KINDS}")
    if raw["label"] not in LABELS:
        raise ValueError(f"{where}: label {raw['label']!r} not in {LABELS}")
    return Record(**{f: raw[f] for f in REQUIRED_FIELDS})


def load(path: str) -> list[Record]:
    records: list[Record] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            where = f"{path}:{n}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
            record = parse_record(raw, where)
            if record.id in seen:
                raise ValueError(f"{where}: duplicate id {record.id!r}")
            seen.add(record.id)
            records.append(record)
    return records
