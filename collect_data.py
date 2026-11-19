#!/usr/bin/env python3
"""Sample sizing space, evaluate it with SPICE, and export a training dataset."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from optimize_sizing import _configure_logging, optimize  # noqa: E402
from scripts.netlist_tools import load_sizing_config  # noqa: E402
from src.design_features import design_feature_values  # noqa: E402
from src.selector import variable_names  # noqa: E402


def _dataset_rows(results_path: Path, sizing_config: dict) -> list[dict]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = []
    for sample in results["samples"]:
        error = str(sample.get("error") or "")
        row = {
            "sample": sample["name"],
            "status": sample["status"],
            "quality_label": (
                "metastable"
                if "metastab" in error.lower()
                else sample["status"]
            ),
            "valid_for_regression": sample["status"] == "evaluated",
            "run_id": results["name"].removeprefix("run_"),
            "error": error,
            "failure_reason": error,
        }
        sizes = sample.get("size", {})
        row.update(
            {
                f"size__{name}": sizes.get(name)
                for name in variable_names(sizing_config)
            }
        )
        engineered = design_feature_values(sizes, sizing_config)
        row.update({f"feature__{name}": value for name, value in engineered.items()})
        metrics = sample.get("metrics", {})
        row.update(metrics)
        delay_s = metrics.get("delay_s")
        if delay_s is not None and float(delay_s) > 0:
            row["speed_hz"] = 1.0 / float(delay_s)
        rows.append(row)
    return rows


def _write_dataset(path: Path, rows: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pandas and a parquet engine such as pyarrow"
            ) from exc
        frames = [pd.DataFrame(rows)]
        if append and path.exists():
            frames.insert(0, pd.read_parquet(path))
        pd.concat(frames, ignore_index=True, sort=False).to_parquet(temporary, index=False)
    elif path.suffix.lower() == ".csv":
        existing = []
        if append and path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        all_rows = existing + rows
        columns = list(dict.fromkeys(key for row in all_rows for key in row))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        raise ValueError("output must end in .csv or .parquet")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect simulator or mock results and export them as a training dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="ngspice", choices=("ngspice", "mock"))
    parser.add_argument(
        "--netlist",
        required=True,
        help="directory with netlist.spice and sizing.yaml",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--algorithm", choices=("lhs", "sobol", "halton", "random"), default="lhs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="optional unique simulator-run name")
    parser.add_argument(
        "--from-results",
        help="import an existing results.json (for example an NRBO verification run)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append rows instead of replacing the dataset",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--mock-failure-rate", type=float, default=0.03)
    parser.add_argument(
        "--output",
        help="dataset path (.csv or .parquet); default: dataset/<netlist-name>/data.csv",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    netlist_dir = Path(args.netlist)
    if not netlist_dir.is_absolute():
        netlist_dir = PROJECT_DIR / netlist_dir
    sizing_config = load_sizing_config(netlist_dir / "sizing.yaml")
    if args.from_results:
        results_path = Path(args.from_results)
        if not results_path.is_absolute():
            results_path = PROJECT_DIR / results_path
        if not results_path.is_file():
            requested = Path(args.from_results)
            if len(requested.parts) >= 3 and requested.parts[0] == "run":
                run_id = requested.parts[1]
                matches = list((PROJECT_DIR / "run").glob(f"*/{run_id}/results.json"))
                if len(matches) == 1:
                    results_path = matches[0]
                elif len(matches) > 1:
                    raise ValueError(
                        f"ambiguous run id {run_id!r}; use run/<netlist>/<run-id>/results.json"
                    )
        rows = _dataset_rows(results_path, sizing_config)
        run_dir = results_path.parent
    else:
        run_dir = optimize(
            netlist=netlist_dir,
            runner_name=args.runner,
            samples=args.samples,
            seed=args.seed,
            algorithm=args.algorithm,
            run_id=args.run_id,
            timeout_s=args.timeout_s,
            mock_failure_rate=args.mock_failure_rate,
        )
        rows = _dataset_rows(run_dir / "results.json", sizing_config)
    output = Path(args.output) if args.output else Path("dataset") / netlist_dir.name / "data.csv"
    if not output.is_absolute():
        output = PROJECT_DIR / output
    if output.resolve() == (PROJECT_DIR / "dataset" / "data.csv").resolve():
        output = PROJECT_DIR / "dataset" / netlist_dir.name / "data.csv"
    _write_dataset(output, rows, append=args.append)
    print(f"Dataset: {output} ({len(rows)} rows; source run: {run_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
