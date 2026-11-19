"""Append-only JSONL dataset for AI StrongARM sizing evaluations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def append_rows(path: str | Path, rows: Iterable[Mapping]) -> int:
    """Append mappings as JSON Lines and return the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def read_rows(path: str | Path) -> list[dict]:
    """Read JSON Lines records, returning an empty list when the file is absent."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows
