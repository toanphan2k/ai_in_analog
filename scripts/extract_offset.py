"""Summarize input-referred offset samples from a CSV file."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json


def extract_offset(csv_path: str | Path, column: str = "vos") -> dict:
    """Summarize valid offset samples with mean, sigma, and range statistics."""
    values = []
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"input CSV must contain a {column!r} column")
        for row in reader:
            if row.get("ok", "").strip().lower() in {"0", "false", "no"}:
                continue
            text = row.get(column, "").strip()
            if not text:
                continue
            value = float(text)
            if math.isfinite(value):
                values.append(value)
    if not values:
        raise ValueError(f"no valid offset samples found in {csv_path}")

    array = np.asarray(values, dtype=float)
    sigma = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "samples": int(array.size),
        "mean_v": float(array.mean()),
        "sigma_v": sigma,
        "abs_3sigma_v": float(3 * sigma),
        "min_v": float(array.min()),
        "max_v": float(array.max()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse offset CSV options and print the resulting summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--column", default="vos")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = extract_offset(args.csv_file, args.column)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
