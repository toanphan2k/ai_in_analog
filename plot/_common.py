"""Shared CSV loading, unit scaling, Pareto, and CLI helpers for plotting."""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@dataclass(frozen=True)
class RunData:
    """Parsed rows from a single results.csv file."""

    path: Path
    label: str
    algorithm: str
    seed: str
    eval_numbers: np.ndarray
    iterations: np.ndarray
    statuses: tuple[str, ...]
    feasible: np.ndarray
    objective_score: np.ndarray
    delay_s: np.ndarray
    power_w: np.ndarray
    offset_v: np.ndarray
    noise_v: np.ndarray
    kickback_v: np.ndarray


def _parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    try:
        parsed = float(text)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def _build_label(path: Path, algorithm: str, seed: str, include_run_name: bool) -> str:
    run_name = path.parent.name if path.parent.name else path.stem
    parts = []
    if algorithm:
        parts.append(algorithm)
    if seed:
        parts.append(f"seed={seed}")
    meta = ", ".join(parts)
    if include_run_name and meta:
        return f"{run_name} ({meta})"
    if meta:
        return meta
    return run_name


def resolve_csv_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve run directories (containing results.csv) or direct CSV paths."""
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_dir():
            csv_path = path / "results.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"results.csv not found in run directory: {path}")
            resolved.append(csv_path)
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Run directory or CSV file not found: {path}")
    return resolved


def load_results_csv(csv_paths: Sequence[str | Path]) -> list[RunData]:
    """Load one or more run directories or results.csv files into RunData records."""
    resolved_paths = resolve_csv_paths(csv_paths)
    runs: list[RunData] = []
    raw_meta: list[tuple[Path, str, str]] = []
    parsed_rows_list: list[list[dict[str, str]]] = []

    for path in resolved_paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"CSV file contains no data rows: {path}")
        algorithms = sorted({r.get("algorithm", "").strip() for r in rows if r.get("algorithm", "").strip()})
        seeds = sorted({r.get("seed", "").strip() for r in rows if r.get("seed", "").strip()})
        algo_str = "/".join(algorithms)
        seed_str = "/".join(seeds)
        raw_meta.append((path, algo_str, seed_str))
        parsed_rows_list.append(rows)

    meta_keys = [(algo, seed) for _, algo, seed in raw_meta]
    need_run_name = len(set(meta_keys)) < len(meta_keys) or any(not algo for algo, _ in meta_keys)

    for (path, algo_str, seed_str), rows in zip(raw_meta, parsed_rows_list):
        eval_numbers = np.arange(1, len(rows) + 1, dtype=int)
        iterations = np.array([_parse_float(r.get("iteration")) for r in rows], dtype=float)
        statuses = tuple(r.get("status", "").strip() for r in rows)
        feasible = np.array([_parse_bool(r.get("rl_feasible")) for r in rows], dtype=bool)
        objective_score = np.array([_parse_float(r.get("rl_objective_score")) for r in rows], dtype=float)
        delay_s = np.array([_parse_float(r.get("delay_s")) for r in rows], dtype=float)
        power_w = np.array([_parse_float(r.get("power_w")) for r in rows], dtype=float)
        offset_v = np.array(
            [
                _parse_float(r.get("input_referred_offset_v", r.get("offset_v")))
                for r in rows
            ],
            dtype=float,
        )
        noise_v = np.array([_parse_float(r.get("input_referred_noise_v")) for r in rows], dtype=float)
        kickback_v = np.array([_parse_float(r.get("kickback_noise_v")) for r in rows], dtype=float)

        runs.append(
            RunData(
                path=path,
                label=_build_label(path, algo_str, seed_str, include_run_name=need_run_name),
                algorithm=algo_str,
                seed=seed_str,
                eval_numbers=eval_numbers,
                iterations=iterations,
                statuses=statuses,
                feasible=feasible,
                objective_score=objective_score,
                delay_s=delay_s,
                power_w=power_w,
                offset_v=offset_v,
                noise_v=noise_v,
                kickback_v=kickback_v,
            )
        )
    return runs


def select_unit(values: np.ndarray, unit_table: Sequence[tuple[float, str]]) -> tuple[float, str]:
    """Select an engineering scale factor and unit label from finite positive values."""
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return unit_table[0]
    median = float(np.median(finite))
    for scale, unit in unit_table:
        scaled = median * scale
        if 1.0 <= scaled < 1000.0:
            return scale, unit
    if median * unit_table[0][0] < 1.0:
        return unit_table[0]
    return unit_table[-1]


TIME_UNITS = ((1e12, "ps"), (1e9, "ns"), (1e6, "µs"), (1e3, "ms"), (1.0, "s"))
POWER_UNITS = ((1e6, "µW"), (1e3, "mW"), (1.0, "W"))
VOLTAGE_UNITS = ((1e6, "µV"), (1e3, "mV"), (1.0, "V"))


def pareto_front_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points minimizing both x and y, sorted by x."""
    if x.size == 0:
        return np.array([], dtype=int)
    order = np.lexsort((y, x))
    front: list[int] = []
    best_y = math.inf
    for idx in order:
        if y[idx] < best_y:
            front.append(int(idx))
            best_y = float(y[idx])
    return np.asarray(front, dtype=int)


def configure_matplotlib(show: bool) -> "matplotlib.pyplot":
    """Configure matplotlib backend and clean publication-style defaults."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 9,
        }
    )
    return plt


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "unknown"


def _extract_run_time_id(path: Path) -> str:
    """Extract the run time/id suffix (e.g. '2026_11_23_05_49' from 'run_2026_11_23_05_49')."""
    parent_name = path.parent.name
    if parent_name.startswith("run_") and len(parent_name) > 4:
        return parent_name[4:]
    return parent_name or path.stem


def default_output_dir(runs: Sequence[RunData]) -> Path:
    """Build default output dir: <ws>/plot_<algorithm>_seed_<seed>_iteration_<iteration>_<run_time_id>."""
    algorithms: list[str] = []
    seeds: list[str] = []
    max_iterations: list[str] = []
    run_ids: list[str] = []
    for run in runs:
        for token in run.algorithm.split("/"):
            token = token.strip()
            if token and token not in algorithms:
                algorithms.append(token)
        for token in run.seed.split("/"):
            token = token.strip()
            if token and token not in seeds:
                seeds.append(token)
        finite_iters = run.iterations[np.isfinite(run.iterations)]
        if finite_iters.size > 0:
            iter_token = str(int(np.max(finite_iters)))
            if iter_token not in max_iterations:
                max_iterations.append(iter_token)
        run_id_token = _extract_run_time_id(run.path).strip()
        if run_id_token and run_id_token not in run_ids:
            run_ids.append(run_id_token)

    algo_part = _sanitize_token("_".join(algorithms)) if algorithms else "unknown"
    seed_part = _sanitize_token("_".join(seeds)) if seeds else "unknown"
    iter_part = _sanitize_token("_".join(max_iterations)) if max_iterations else "unknown"
    suffix = f"_{_sanitize_token('_'.join(run_ids))}" if run_ids else ""
    return PROJECT_DIR / f"plot_{algo_part}_seed_{seed_part}_iteration_{iter_part}{suffix}"


def default_output_path(runs: Sequence[RunData], default_stem: str) -> Path:
    """Choose default PNG output path under <ws>/plot_<algorithm>_seed_<seed>_iteration_<iteration>/."""
    return default_output_dir(runs) / f"{default_stem}.png"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create a shared CLI argument parser supporting --output-dir <run_dir>."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        "--csv",
        dest="output_dir",
        nargs="+",
        required=True,
        help="path(s) to run output directory(ies) (e.g. run/strongarm_5ghz/<run_id>) or results.csv file(s)",
    )
    parser.add_argument(
        "-o",
        "--save-path",
        dest="save_path",
        type=Path,
        default=None,
        help=(
            "custom output image path "
            "(default: <ws>/plot_<algorithm>_seed_<seed>_iteration_<iteration>_<run_id>/<plot_name>.png)"
        ),
    )
    parser.add_argument("--title", default=None, help="custom plot title")
    parser.add_argument("--dpi", type=int, default=180, help="output image DPI (default: 180)")
    parser.add_argument("--show", action="store_true", help="display plot interactively")
    return parser


def save_or_show(fig, output_path: Path, *, dpi: int, show: bool) -> Path:
    """Save figure to disk and optionally display it interactively."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
