"""Propose the next normalized W/L candidate without launching a simulation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import SIZING_CONFIG, load_config
from src.selector import candidate_pool, select_next, variable_names


def _normalize(value: float, bound: dict) -> float:
    low, high = float(bound["min"]), float(bound["max"])
    if bound.get("scale", "linear") == "log":
        if value <= 0 or low <= 0 or high <= low:
            raise ValueError("log-scaled design values and bounds must be positive")
        return math.log(value / low) / math.log(high / low)
    return (value - low) / (high - low)


def _denormalize(value: float, bound: dict) -> float:
    low, high = float(bound["min"]), float(bound["max"])
    if bound.get("scale", "linear") == "log":
        return low * (high / low) ** float(value)
    return low + float(value) * (high - low)


def propose(
    sizing_config: dict,
    *,
    results_csv: str | Path | None = None,
    objective: str = "delay_s",
    pool_size: int = 256,
    seed: int = 0,
) -> dict:
    """Propose the best candidate from a sampled pool using recorded observations."""
    names = variable_names(sizing_config)
    bounds = sizing_config["optimization"].get("bounds", {})
    if any(name not in bounds for name in names):
        raise ValueError("optimization.bounds must define bounds for every optimization variable")

    observed_x, observed_y = [], []
    if results_csv and Path(results_csv).exists():
        with Path(results_csv).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("status", "evaluated").lower() != "evaluated":
                    continue
                try:
                    metric = float(row[objective])
                    design = json.loads(row["design_json"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not math.isfinite(metric) or not all(name in design for name in names):
                    continue
                observed_x.append(
                    [_normalize(float(design[name]), bounds[name]) for name in names]
                )
                observed_y.append(metric)

    pool = candidate_pool(pool_size, sizing_config, seed=seed)
    candidate = select_next(
        pool,
        np.asarray(observed_x, dtype=float) if observed_x else None,
        np.asarray(observed_y, dtype=float) if observed_y else None,
    )
    return {
        "objective": objective,
        "observations_used": len(observed_y),
        "x_norm": candidate.tolist(),
        "design": {
            name: _denormalize(float(value), bounds[name])
            for name, value in zip(names, candidate)
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse proposal options and print the next normalized sizing candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SIZING_CONFIG)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--objective", default="delay_s")
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = propose(
        load_config(args.config),
        results_csv=args.results,
        objective=args.objective,
        pool_size=args.pool_size,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
