"""Extract input-referred offset from ngspice scalar metrics or Monte Carlo CSV samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json
from scripts.waveform_io import load_scalar_metrics


def extract_offset_metric(
    metrics_path: str | Path,
    *,
    key: str = "input_referred_offset_v",
) -> dict[str, float]:
    """Read input-referred offset magnitude from a unified ngspice scalar metrics file."""
    scalars = load_scalar_metrics(metrics_path)
    if key not in scalars:
        raise ValueError(f"metrics file {metrics_path} does not contain {key!r}")
    offset_v = float(scalars[key])
    if not math.isfinite(offset_v) or offset_v < 0.0:
        raise ValueError(f"invalid input-referred offset value: {offset_v}")
    return {
        "input_referred_offset_v": offset_v,
        "offset_v": offset_v,
    }


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
    """Parse offset file options and print the resulting offset metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--column", default="vos")
    parser.add_argument("--key", default="input_referred_offset_v")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.input_file.suffix.lower() == ".csv":
        result = extract_offset(args.input_file, args.column)
    else:
        result = extract_offset_metric(args.input_file, key=args.key)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
