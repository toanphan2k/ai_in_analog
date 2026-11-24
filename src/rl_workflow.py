"""Direct-feedback PPO/SAC control of NRBO using YAML-weighted scoring."""
from __future__ import annotations

import csv
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from runner.mock_runner import MockRunner
from runner.ngspice_runner import NgspiceRun
from scripts.deck_generation import render_sized_deck
from scripts.extract_delay import TransientFailureError, extract_delay
from scripts.extract_gain import extract_gain
from scripts.extract_kickback import extract_kickback
from scripts.extract_noise import extract_noise
from scripts.extract_offset import extract_offset_metric
from scripts.netlist_tools import load_sizing_config
from scripts.pdk_models import rewrite_deck_with_reduced_models, write_spiceinit
from src.rl_nrbo import optimize_with_rl_nrbo
from src.selector import variable_names

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("ai_in_analog.rl_nrbo")
CHECKPOINT_INTERVAL = 25
CHECKPOINT_MODEL_NAME = "model.zip"
EVALUATION_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    subprocess.SubprocessError,
)
RESULT_PREFIX_COLUMNS = (
    "sample",
    "algorithm",
    "iteration",
    "seed",
    "status",
)
RESULT_METRIC_COLUMNS = (
    "delay_s",
    "delay_mean_s",
    "reset_time_s",
    "reset_time_mean_s",
    "speed_hz",
    "cycles_measured",
    "power_w",
    "energy_j",
    "energy_per_comparison_j",
    "input_referred_offset_v",
    "input_referred_noise_v",
    "kickback_noise_v",
    "output_swing_v",
    "voh_v",
    "vol_v",
    "overshoot_v",
    "dc_gain_v_per_v",
    "peak_gain_v_per_v",
    "peak_gain_frequency_hz",
    "bandwidth_3db_hz",
    "ac_freq_points",
    "rl_objective_score",
    "rl_feasible",
    "constraint_violations",
    "error",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _physical_sizes(config: Mapping, vector: Sequence[float]) -> dict[str, float]:
    names = variable_names(config)
    bounds = config.get("optimization", {}).get("bounds", {})
    if len(vector) != len(names):
        raise ValueError(f"expected {len(names)} design values, received {len(vector)}")
    sizes = {}
    for name, normalized in zip(names, vector):
        bound = bounds.get(name)
        if bound is None:
            raise ValueError(f"optimization.bounds is missing {name!r}")
        low, high = float(bound["min"]), float(bound["max"])
        normalized = float(normalized)
        if not 0.0 <= normalized <= 1.0 or high <= low:
            raise ValueError(f"invalid bounds or normalized value for {name}")
        if bound.get("scale", "linear") == "log":
            if low <= 0:
                raise ValueError(f"log-scaled bound for {name} must be positive")
            value = low * (high / low) ** normalized
        else:
            value = low + normalized * (high - low)
        sizes[name] = float(value)
    return sizes


def _normalize_sizes(config: Mapping, sizes: Mapping[str, Any]) -> list[float] | None:
    """Invert _physical_sizes from physical micrometre columns when resuming from results.csv."""
    names = variable_names(config)
    bounds = config.get("optimization", {}).get("bounds", {})
    vector: list[float] = []
    for name in names:
        raw = sizes.get(name)
        bound = bounds.get(name)
        if raw is None or bound is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        low, high = float(bound["min"]), float(bound["max"])
        if high <= low:
            return None
        if bound.get("scale", "linear") == "log":
            if low <= 0.0 or val <= 0.0:
                return None
            norm = math.log(val / low) / math.log(high / low)
        else:
            norm = (val - low) / (high - low)
        vector.append(min(1.0, max(0.0, float(norm))))
    return vector


def _size_column_names(
    samples: Sequence[Mapping[str, Any]],
    design_names: Sequence[str] = (),
) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for name in design_names:
        if name not in seen:
            columns.append(name)
            seen.add(name)
    extra: set[str] = set()
    for sample in samples:
        size = sample.get("size")
        if isinstance(size, Mapping):
            extra.update(str(key) for key in size if str(key) not in seen)
    columns.extend(sorted(extra))
    return tuple(columns)


def _write_results_csv(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    design_names: Sequence[str] = (),
    algorithm: str | None = None,
    seed: int | None = None,
) -> None:
    size_columns = _size_column_names(samples, design_names)
    fieldnames = RESULT_PREFIX_COLUMNS + size_columns + RESULT_METRIC_COLUMNS
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            metrics = sample.get("metrics", {})
            size = sample.get("size", {})
            sample_algorithm = sample.get("algorithm")
            sample_iteration = sample.get("iteration")
            sample_seed = sample.get("seed")
            row: dict[str, Any] = {
                "sample": sample["name"],
                "algorithm": (
                    sample_algorithm
                    if sample_algorithm is not None
                    else (algorithm if algorithm is not None else "")
                ),
                "iteration": "" if sample_iteration is None else sample_iteration,
                "seed": (
                    sample_seed
                    if sample_seed is not None
                    else (seed if seed is not None else "")
                ),
                "status": sample["status"],
                "constraint_violations": sample.get(
                    "constraint_violations",
                    metrics.get("constraint_violations", ""),
                ),
                "error": sample.get("error", ""),
            }
            if isinstance(size, Mapping):
                for column in size_columns:
                    if column in size:
                        row[column] = size[column]
            for column in RESULT_METRIC_COLUMNS:
                if column in {"error", "constraint_violations"}:
                    continue
                if column in metrics:
                    row[column] = metrics[column]
            writer.writerow(row)
    tmp_path.replace(path)


def _write_candidates_csv(
    path: Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    design_names: Sequence[str] = (),
    algorithm: str,
    seed: int,
    records_by_name: Mapping[str, Mapping[str, Any]],
) -> None:
    size_columns = _size_column_names(candidates, design_names)
    fieldnames = ("rank",) + RESULT_PREFIX_COLUMNS + size_columns + RESULT_METRIC_COLUMNS
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, cand in enumerate(candidates, start=1):
            sample_name = str(cand.get("sample", ""))
            source_rec = records_by_name.get(sample_name, {})
            metrics = dict(source_rec.get("metrics", {}))
            metrics.update(cand.get("metrics") or {})
            size = cand.get("size") or source_rec.get("size") or {}
            row: dict[str, Any] = {
                "rank": rank,
                "sample": sample_name,
                "algorithm": source_rec.get("algorithm", algorithm),
                "iteration": source_rec.get("iteration", ""),
                "seed": source_rec.get("seed", seed),
                "status": cand.get("status", source_rec.get("status", "")),
                "rl_objective_score": cand.get("score", metrics.get("rl_objective_score", "")),
                "rl_feasible": cand.get("feasible", metrics.get("rl_feasible", "")),
                "constraint_violations": source_rec.get(
                    "constraint_violations",
                    metrics.get("constraint_violations", ""),
                ),
                "error": cand.get("error", source_rec.get("error", "")),
            }
            if isinstance(size, Mapping):
                for column in size_columns:
                    if column in size:
                        row[column] = size[column]
            for column in RESULT_METRIC_COLUMNS:
                if column in {"rl_objective_score", "rl_feasible", "constraint_violations", "error"}:
                    continue
                if column in metrics:
                    row[column] = metrics[column]
            writer.writerow(row)
    tmp_path.replace(path)


def _load_existing_results_csv(
    csv_path: Path,
    sizing_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None, int | None]:
    """Load existing sample records from results.csv for --resume."""
    design_names = variable_names(sizing_config)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    samples: list[dict[str, Any]] = []
    detected_algo: str | None = None
    detected_seed: int | None = None

    for row in rows:
        algo = (row.get("algorithm") or "").strip()
        if algo and detected_algo is None:
            detected_algo = algo.upper()
        seed_text = (row.get("seed") or "").strip()
        if seed_text and detected_seed is None:
            try:
                detected_seed = int(float(seed_text))
            except ValueError:
                pass

        size: dict[str, float] = {}
        for col in design_names:
            raw = (row.get(col) or "").strip()
            if raw:
                try:
                    size[col] = float(raw)
                except ValueError:
                    pass

        metrics: dict[str, Any] = {}
        for col in RESULT_METRIC_COLUMNS:
            if col in {"error", "constraint_violations"}:
                continue
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            if col == "rl_feasible":
                metrics[col] = raw.lower() in {"true", "1", "yes"}
            elif col in {"cycles_measured", "ac_freq_points"}:
                try:
                    metrics[col] = int(float(raw))
                except ValueError:
                    pass
            else:
                try:
                    val = float(raw)
                    if math.isfinite(val):
                        metrics[col] = val
                except ValueError:
                    pass

        constraint_violations = (row.get("constraint_violations") or "").strip()
        if constraint_violations:
            metrics["constraint_violations"] = constraint_violations

        iter_raw = (row.get("iteration") or "").strip()
        iteration_val: int | None = None
        if iter_raw:
            try:
                iteration_val = int(float(iter_raw))
            except ValueError:
                pass

        seed_val: int | None = None
        if seed_text:
            try:
                seed_val = int(float(seed_text))
            except ValueError:
                pass

        x_norm = _normalize_sizes(sizing_config, size)
        rec: dict[str, Any] = {
            "name": (row.get("sample") or "").strip(),
            "algorithm": algo.upper() if algo else detected_algo,
            "iteration": iteration_val,
            "seed": seed_val if seed_val is not None else detected_seed,
            "status": (row.get("status") or "failed").strip(),
            "x_norm": x_norm,
            "size": size,
            "metrics": metrics,
            "constraint_violations": constraint_violations,
            "error": (row.get("error") or "").strip(),
            "files": {},
        }
        rec.update(metrics)
        samples.append(rec)

    return samples, detected_algo, detected_seed


def _load_score_config(netlist_dir: Path) -> tuple[dict, Path]:
    path = netlist_dir / "scoring.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"score config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"score config must contain a YAML mapping: {path}")
    scoring = payload.get("scoring", {})
    weights = payload.get("weights")
    references = payload.get("references")
    constraints = payload.get("constraints", {})
    penalty = payload.get("penalty", {})
    if scoring is None:
        scoring = {}
    if constraints is None:
        constraints = {}
    if penalty is None:
        penalty = {}
    if not isinstance(scoring, dict):
        raise ValueError("score config 'scoring' must be a mapping")
    if not isinstance(weights, dict) or not isinstance(references, dict):
        raise ValueError("score config must define 'weights' and 'references'")
    if not isinstance(constraints, dict):
        raise ValueError("score config 'constraints' must be a mapping")
    if not isinstance(penalty, dict):
        raise ValueError("score config 'penalty' must be a mapping")
    payload["scoring"] = scoring
    payload["constraints"] = constraints
    payload["penalty"] = penalty
    return payload, path


def _parse_optional_positive(mapping: Mapping[str, Any], key: str) -> float | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{key} must be finite and positive")
    return value


def _parse_optional_nonnegative(mapping: Mapping[str, Any], key: str) -> float | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    value = float(raw)
    if value < 0.0 or not math.isfinite(value):
        raise ValueError(f"{key} must be finite and nonnegative")
    return value


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def resolve_score_config(
    score_config: Mapping[str, Any],
    sizing_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and resolve scoring, weights, references, constraints, and penalties."""
    scoring = score_config.get("scoring") or {}
    weights = score_config["weights"]
    references = score_config["references"]
    constraints = score_config.get("constraints") or {}
    penalty = score_config.get("penalty") or {}

    objective = str(scoring.get("objective", "minimize")).strip().lower()
    if objective != "minimize":
        raise ValueError(f"unsupported scoring.objective {objective!r}; only 'minimize' is supported")
    infeasible_score = float(scoring.get("infeasible_score", 1000.0))
    if infeasible_score <= 0.0 or not math.isfinite(infeasible_score):
        raise ValueError("scoring.infeasible_score must be finite and positive")

    delay_weight = float(weights.get("delay", 1.0))
    power_weight = float(weights.get("power", 0.0))
    energy_weight = float(weights.get("energy", 0.0))
    offset_weight = float(weights.get("offset", 0.0))
    noise_weight = float(weights.get("noise", 0.0))
    kickback_weight = float(weights.get("kickback", 0.0))

    default_delay_reference_s = float(
        sizing_config.get("analyses", {}).get("transient", {}).get("period_s", 1e-9)
    )
    delay_reference_s = float(references.get("delay_s", default_delay_reference_s))
    power_reference_w = float(references.get("power_w", 0.01))
    energy_reference_j = float(references.get("energy_j", 1e-15))
    offset_reference_v = float(references.get("offset_v", 1e-3))
    noise_reference_v = float(references.get("noise_v", 1e-3))
    kickback_reference_v = float(references.get("kickback_v", 1e-3))

    resolved_weights = (
        delay_weight,
        power_weight,
        energy_weight,
        offset_weight,
        noise_weight,
        kickback_weight,
    )
    resolved_references = (
        delay_reference_s,
        power_reference_w,
        energy_reference_j,
        offset_reference_v,
        noise_reference_v,
        kickback_reference_v,
    )
    if any(weight < 0.0 or not math.isfinite(weight) for weight in resolved_weights):
        raise ValueError("objective weights must be finite and nonnegative")
    if sum(resolved_weights) == 0.0:
        raise ValueError("at least one objective weight must be positive")
    if any(reference <= 0.0 or not math.isfinite(reference) for reference in resolved_references):
        raise ValueError("metric references must be finite and positive")

    max_delay_s = _parse_optional_positive(constraints, "max_delay_s")
    max_reset_time_s = _parse_optional_positive(constraints, "max_reset_time_s")
    max_power_w = _parse_optional_positive(constraints, "max_power_w")
    max_offset_v = _parse_optional_positive(constraints, "max_offset_v")
    max_noise_v = _parse_optional_positive(constraints, "max_noise_v")
    max_kickback_v = _parse_optional_positive(constraints, "max_kickback_v")
    min_output_swing_v = _parse_optional_positive(constraints, "min_output_swing_v")
    min_voh_v = _parse_optional_positive(constraints, "min_voh_v")
    max_vol_v = _parse_optional_nonnegative(constraints, "max_vol_v")
    max_overshoot_v = _parse_optional_nonnegative(constraints, "max_overshoot_v")

    penalty_lambda = float(penalty.get("lambda", 100.0))
    if penalty_lambda < 0.0 or not math.isfinite(penalty_lambda):
        raise ValueError("penalty.lambda must be finite and nonnegative")
    penalty_type = str(penalty.get("type", "quadratic")).strip().lower()
    if penalty_type not in {"quadratic", "linear"}:
        raise ValueError(
            f"unsupported penalty.type {penalty_type!r}; expected 'quadratic' or 'linear'"
        )

    return {
        "objective": objective,
        "infeasible_score": infeasible_score,
        "delay_weight": delay_weight,
        "power_weight": power_weight,
        "energy_weight": energy_weight,
        "offset_weight": offset_weight,
        "noise_weight": noise_weight,
        "kickback_weight": kickback_weight,
        "delay_reference_s": delay_reference_s,
        "power_reference_w": power_reference_w,
        "energy_reference_j": energy_reference_j,
        "offset_reference_v": offset_reference_v,
        "noise_reference_v": noise_reference_v,
        "kickback_reference_v": kickback_reference_v,
        "max_delay_s": max_delay_s,
        "max_reset_time_s": max_reset_time_s,
        "max_power_w": max_power_w,
        "max_offset_v": max_offset_v,
        "max_noise_v": max_noise_v,
        "max_kickback_v": max_kickback_v,
        "min_output_swing_v": min_output_swing_v,
        "min_voh_v": min_voh_v,
        "max_vol_v": max_vol_v,
        "max_overshoot_v": max_overshoot_v,
        "penalty_lambda": penalty_lambda,
        "penalty_type": penalty_type,
    }


def _extract_scored_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = record.get("metrics", {})
    return {
        "delay_s": _finite_metric(metrics.get("delay_s")),
        "reset_time_s": _finite_metric(metrics.get("reset_time_s")),
        "power_w": _finite_metric(metrics.get("power_w")),
        "energy_j": _finite_metric(metrics.get("energy_per_comparison_j", metrics.get("energy_j"))),
        "offset_v": _finite_metric(metrics.get("input_referred_offset_v", metrics.get("offset_v"))),
        "noise_v": _finite_metric(metrics.get("input_referred_noise_v")),
        "kickback_v": _finite_metric(metrics.get("kickback_noise_v")),
        "output_swing_v": _finite_metric(metrics.get("output_swing_v")),
        "voh_v": _finite_metric(metrics.get("voh_v")),
        "vol_v": _finite_metric(metrics.get("vol_v")),
        "overshoot_v": _finite_metric(metrics.get("overshoot_v")),
    }


def _graded_failure_score(record: Mapping[str, Any], infeasible_score: float) -> float:
    """Compute a smooth graded penalty in [0.5*infeasible_score, infeasible_score] for failed sims."""
    info = record.get("failure_info")
    if not isinstance(info, Mapping):
        return infeasible_score

    dec_th = _finite_metric(info.get("decision_threshold_v")) or 0.9
    dec_th = max(dec_th, 1e-9)
    peak_diff = max(0.0, _finite_metric(info.get("peak_diff_v")) or 0.0)
    decision_ratio = min(1.0, peak_diff / dec_th)

    total_cycles = max(int(info.get("total_cycles") or 1), 1)
    done_cycles = min(total_cycles, max(0, int(info.get("cycles_completed") or 0)))
    cycle_ratio = done_cycles / total_cycles

    rst_min = _finite_metric(info.get("reset_min_v"))
    rst_th = _finite_metric(info.get("reset_threshold_v"))
    if rst_min is not None and rst_th is not None and rst_th > 0.0:
        reset_ratio = min(1.0, max(0.0, rst_min / rst_th))
        progress = 0.55 * decision_ratio + 0.25 * reset_ratio + 0.20 * cycle_ratio
    else:
        progress = 0.75 * decision_ratio + 0.25 * cycle_ratio

    progress = min(0.99, max(0.0, progress))
    return float(0.5 * infeasible_score + 0.5 * infeasible_score * (1.0 - progress))


def compute_objective_score(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> float:
    """Compute S_total (lower is better) with graded failure penalty and constraint penalties."""
    infeasible_score = float(resolved["infeasible_score"])
    if record.get("status") == "failed":
        return _graded_failure_score(record, infeasible_score)

    m = _extract_scored_metrics(record)
    delay = m["delay_s"]
    reset_time = m["reset_time_s"]
    power = m["power_w"]
    energy = m["energy_j"]
    offset = m["offset_v"]
    noise = m["noise_v"]
    kickback = m["kickback_v"]
    output_swing = m["output_swing_v"]
    voh = m["voh_v"]
    vol = m["vol_v"]
    overshoot = m["overshoot_v"]

    if (
        (resolved["delay_weight"] > 0.0 and delay is None)
        or (resolved["power_weight"] > 0.0 and power is None)
        or (resolved["energy_weight"] > 0.0 and energy is None)
        or (resolved["offset_weight"] > 0.0 and offset is None)
        or (resolved["noise_weight"] > 0.0 and noise is None)
        or (resolved["kickback_weight"] > 0.0 and kickback is None)
        or (resolved["max_delay_s"] is not None and delay is None)
        or (resolved.get("max_reset_time_s") is not None and reset_time is None)
        or (resolved["max_power_w"] is not None and power is None)
        or (resolved["max_offset_v"] is not None and offset is None)
        or (resolved["max_noise_v"] is not None and noise is None)
        or (resolved["max_kickback_v"] is not None and kickback is None)
        or (resolved["min_output_swing_v"] is not None and output_swing is None)
        or (resolved.get("min_voh_v") is not None and voh is None)
        or (resolved.get("max_vol_v") is not None and vol is None)
        or (resolved.get("max_overshoot_v") is not None and overshoot is None)
    ):
        return _graded_failure_score(record, infeasible_score)

    value = 0.0
    if resolved["delay_weight"] > 0.0:
        value += resolved["delay_weight"] * (delay / resolved["delay_reference_s"])
    if resolved["power_weight"] > 0.0:
        value += resolved["power_weight"] * (power / resolved["power_reference_w"])
    if resolved["energy_weight"] > 0.0:
        value += resolved["energy_weight"] * (energy / resolved["energy_reference_j"])
    if resolved["offset_weight"] > 0.0:
        value += resolved["offset_weight"] * (abs(offset) / resolved["offset_reference_v"])
    if resolved["noise_weight"] > 0.0:
        value += resolved["noise_weight"] * (noise / resolved["noise_reference_v"])
    if resolved["kickback_weight"] > 0.0:
        value += resolved["kickback_weight"] * (kickback / resolved["kickback_reference_v"])

    violations: list[float] = []
    if resolved["max_power_w"] is not None:
        violations.append(max(0.0, power / resolved["max_power_w"] - 1.0))
    if resolved["max_delay_s"] is not None:
        violations.append(max(0.0, delay / resolved["max_delay_s"] - 1.0))
    if resolved.get("max_reset_time_s") is not None:
        violations.append(max(0.0, reset_time / resolved["max_reset_time_s"] - 1.0))
    if resolved["max_offset_v"] is not None:
        violations.append(max(0.0, abs(offset) / resolved["max_offset_v"] - 1.0))
    if resolved["max_noise_v"] is not None:
        violations.append(max(0.0, noise / resolved["max_noise_v"] - 1.0))
    if resolved["max_kickback_v"] is not None:
        violations.append(max(0.0, kickback / resolved["max_kickback_v"] - 1.0))
    if resolved["min_output_swing_v"] is not None:
        violations.append(
            max(0.0, (resolved["min_output_swing_v"] - output_swing) / resolved["min_output_swing_v"])
        )
    if resolved.get("min_voh_v") is not None:
        violations.append(max(0.0, (resolved["min_voh_v"] - voh) / resolved["min_voh_v"]))
    if resolved.get("max_vol_v") is not None:
        denom = max(resolved["max_vol_v"], 0.05)
        violations.append(max(0.0, (vol - resolved["max_vol_v"]) / denom))
    if resolved.get("max_overshoot_v") is not None:
        denom = max(resolved["max_overshoot_v"], 0.05)
        violations.append(max(0.0, (overshoot - resolved["max_overshoot_v"]) / denom))

    penalty_lambda = float(resolved["penalty_lambda"])
    if penalty_lambda > 0.0 and violations:
        if resolved["penalty_type"] == "quadratic":
            value += penalty_lambda * sum(g * g for g in violations)
        else:
            value += penalty_lambda * sum(violations)

    return float(value) if math.isfinite(value) else infeasible_score


def describe_constraint_violations(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> str:
    """Return human-readable infeasibility reasons when rl_feasible is False."""
    if record.get("status") == "failed":
        return "simulation_failed"

    m = _extract_scored_metrics(record)
    delay = m["delay_s"]
    reset_time = m["reset_time_s"]
    power = m["power_w"]
    energy = m["energy_j"]
    offset = m["offset_v"]
    noise = m["noise_v"]
    kickback = m["kickback_v"]
    output_swing = m["output_swing_v"]
    voh = m["voh_v"]
    vol = m["vol_v"]
    overshoot = m["overshoot_v"]

    reasons: list[str] = []
    for w_key, val, name in (
        ("delay_weight", delay, "delay_s"),
        ("power_weight", power, "power_w"),
        ("energy_weight", energy, "energy_j"),
        ("offset_weight", offset, "input_referred_offset_v"),
        ("noise_weight", noise, "input_referred_noise_v"),
        ("kickback_weight", kickback, "kickback_noise_v"),
    ):
        if resolved[w_key] > 0.0 and val is None:
            reasons.append(f"missing {name}")

    if resolved["max_delay_s"] is not None:
        if delay is None:
            reasons.append("missing delay_s")
        elif delay > resolved["max_delay_s"]:
            reasons.append(
                f"delay_s={delay * 1e12:.2f}ps > max {resolved['max_delay_s'] * 1e12:.2f}ps"
            )
    if resolved.get("max_reset_time_s") is not None:
        if reset_time is None:
            reasons.append("missing reset_time_s")
        elif reset_time > resolved["max_reset_time_s"]:
            reasons.append(
                f"reset_time_s={reset_time * 1e12:.2f}ps > max {resolved['max_reset_time_s'] * 1e12:.2f}ps"
            )
    if resolved["max_power_w"] is not None:
        if power is None:
            reasons.append("missing power_w")
        elif power > resolved["max_power_w"]:
            reasons.append(
                f"power_w={power * 1e3:.3f}mW > max {resolved['max_power_w'] * 1e3:.3f}mW"
            )
    if resolved["max_offset_v"] is not None:
        if offset is None:
            reasons.append("missing input_referred_offset_v")
        elif abs(offset) > resolved["max_offset_v"]:
            reasons.append(
                f"|offset_v|={abs(offset) * 1e3:.2f}mV > max {resolved['max_offset_v'] * 1e3:.2f}mV"
            )
    if resolved["max_noise_v"] is not None:
        if noise is None:
            reasons.append("missing input_referred_noise_v")
        elif noise > resolved["max_noise_v"]:
            reasons.append(
                f"noise_v={noise * 1e6:.1f}uV > max {resolved['max_noise_v'] * 1e6:.1f}uV"
            )
    if resolved["max_kickback_v"] is not None:
        if kickback is None:
            reasons.append("missing kickback_noise_v")
        elif kickback > resolved["max_kickback_v"]:
            reasons.append(
                f"kickback_v={kickback * 1e3:.2f}mV > max {resolved['max_kickback_v'] * 1e3:.2f}mV"
            )
    if resolved["min_output_swing_v"] is not None:
        if output_swing is None:
            reasons.append("missing output_swing_v")
        elif output_swing < resolved["min_output_swing_v"]:
            reasons.append(
                f"output_swing_v={output_swing:.3f}V < min {resolved['min_output_swing_v']:.3f}V"
            )
    if resolved.get("min_voh_v") is not None:
        if voh is None:
            reasons.append("missing voh_v")
        elif voh < resolved["min_voh_v"]:
            reasons.append(f"voh_v={voh:.3f}V < min {resolved['min_voh_v']:.3f}V")
    if resolved.get("max_vol_v") is not None:
        if vol is None:
            reasons.append("missing vol_v")
        elif vol > resolved["max_vol_v"]:
            reasons.append(f"vol_v={vol:.3f}V > max {resolved['max_vol_v']:.3f}V")
    if resolved.get("max_overshoot_v") is not None:
        if overshoot is None:
            reasons.append("missing overshoot_v")
        elif overshoot > resolved["max_overshoot_v"]:
            reasons.append(
                f"overshoot_v={overshoot:.3f}V > max {resolved['max_overshoot_v']:.3f}V"
            )
    return "; ".join(dict.fromkeys(reasons))


def is_record_feasible(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> bool:
    """Return True iff the candidate succeeded, has all required metrics, and meets all constraints."""
    return describe_constraint_violations(record, resolved) == ""


def _run_ngspice(
    runner: NgspiceRun,
    deck: Path,
    workdir: Path,
    primary_outputs: Sequence[Path],
) -> None:
    write_spiceinit(workdir)
    result = runner.run(deck.resolve(), workdir.resolve(), execute=True)
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"ngspice returned {result.get('returncode')}: {result.get('stdout', '')}"
        )
    log_path = workdir / "ngspice.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if "fatal error" in log_text.lower():
        raise RuntimeError("ngspice reported a fatal error; see ngspice.log")
    missing = [
        output.name
        for output in primary_outputs
        if not output.is_file() or output.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"ngspice did not produce expected output(s) ({', '.join(missing)}); see ngspice.log"
        )


def _expected_evaluations(
    algorithm: str,
    population_size: int,
    timesteps: int,
    episode_length: int,
) -> int:
    """Compute exact max unique evaluations for one training + final rollout session."""
    if algorithm.upper() == "PPO":
        rollout_steps = min(8, max(2, timesteps))
        effective_steps = math.ceil(timesteps / rollout_steps) * rollout_steps
    else:
        effective_steps = timesteps
    training_resets = 1 + (effective_steps // episode_length)
    total_resets = training_resets + 1  # +1 for final deterministic rollout
    total_steps = effective_steps + episode_length
    return total_resets * population_size + total_steps * (population_size - 1)


def _sample_dir(run_dir: Path, index: int) -> Path:
    return run_dir / "data" / f"sample_{index:06d}"


def _evaluate_sample(
    index: int,
    vector: Sequence[float],
    sizes: Mapping[str, float],
    *,
    run_name: str,
    template: str,
    sizing_config: Mapping,
    runner: NgspiceRun | MockRunner,
    run_dir: Path,
) -> dict:
    sample_name = f"{run_name}_{index:06d}"
    sample_dir = _sample_dir(run_dir, index)
    sample_dir.mkdir(parents=True)
    record: dict[str, Any] = {
        "name": sample_name,
        "status": "failed",
        "x_norm": [float(value) for value in vector],
        "size": dict(sizes),
        "metrics": {},
        "failure_info": None,
        "files": {},
        "error": "",
    }
    _atomic_json(sample_dir / "size.json", {"name": sample_name, "size": sizes})
    analyses = sizing_config.get("analyses", {})
    transient_config = analyses.get("transient", {})

    if isinstance(runner, MockRunner):
        record.update(
            runner.evaluate(
                sizes,
                np.asarray(vector, dtype=float),
                sample_index=index,
                vdd_v=float(transient_config.get("vdd_v", 1.8)),
            )
        )
        record.update(record["metrics"])
        _atomic_json(sample_dir / "result.json", record)
        return record

    deck_path = sample_dir / "netlist.spice"
    deck_text = render_sized_deck(template, sizes, sizing_config)
    deck_text = rewrite_deck_with_reduced_models(deck_text)
    deck_path.write_text(deck_text, encoding="utf-8")

    transient_waveform = sample_dir / str(transient_config.get("waveform", "transient.txt"))
    gain_config = analyses.get("gain", {})
    gain_waveform = sample_dir / str(gain_config.get("waveform", "ac.txt"))
    offset_config = analyses.get("offset", {})
    offset_metrics_file = sample_dir / str(offset_config.get("metrics_file", "metrics.txt"))
    noise_config = analyses.get("noise", {})
    noise_waveform = sample_dir / str(noise_config.get("waveform", "noise.txt"))
    noise_metrics_file = sample_dir / str(noise_config.get("metrics_file", "metrics.txt"))
    kickback_config = analyses.get("kickback", {})
    kickback_waveform = sample_dir / str(kickback_config.get("waveform", "kickback.txt"))

    transient_enabled = bool(transient_config.get("enabled", True))
    primary_outputs: list[Path] = [transient_waveform] if transient_enabled else []

    secondary_outputs: list[Path] = []
    if gain_config.get("enabled", False):
        secondary_outputs.append(gain_waveform)
    if offset_config.get("enabled", False):
        secondary_outputs.append(offset_metrics_file)
    if noise_config.get("enabled", False):
        secondary_outputs.append(noise_waveform)
    if kickback_config.get("enabled", False):
        secondary_outputs.append(kickback_waveform)
    secondary_outputs = list(dict.fromkeys(secondary_outputs))

    errors: list[str] = []
    try:
        _run_ngspice(
            runner,
            deck_path,
            sample_dir,
            primary_outputs if primary_outputs else secondary_outputs,
        )
    except EVALUATION_ERRORS as exc:
        record["error"] = f"ngspice: {type(exc).__name__}: {exc}"
        _atomic_json(sample_dir / "result.json", record)
        return record

    if transient_enabled:
        try:
            power_window = transient_config.get("power_window_s")
            record["metrics"].update(
                extract_delay(
                    transient_waveform,
                    float(transient_config["vdd_v"]),
                    float(transient_config["period_s"]),
                    output_positive=str(transient_config.get("output_positive", "v(X)")),
                    output_negative=str(transient_config.get("output_negative", "v(Y)")),
                    clock_vector=str(transient_config.get("clock_vector", "v(ck)")),
                    input_positive=transient_config.get("input_positive", "v(inp)"),
                    input_negative=transient_config.get("input_negative", "v(inn)"),
                    power_vector=str(transient_config.get("power_vector", "p_inst")),
                    power_from_s=float(power_window[0]) if power_window else None,
                    power_to_s=float(power_window[1]) if power_window else None,
                    reset_threshold_ratio=float(transient_config.get("reset_threshold_ratio", 0.9)),
                    inverting=bool(transient_config.get("inverting", True)),
                    check_polarity=bool(transient_config.get("check_polarity", True)),
                )
            )
        except TransientFailureError as exc:
            record["failure_info"] = exc.as_dict()
            record["error"] = f"transient: {type(exc).__name__}: {exc}"
            _atomic_json(sample_dir / "result.json", record)
            return record
        except EVALUATION_ERRORS as exc:
            record["error"] = f"transient: {type(exc).__name__}: {exc}"
            _atomic_json(sample_dir / "result.json", record)
            return record

    missing_secondary = [
        out.name for out in secondary_outputs if not out.is_file() or out.stat().st_size == 0
    ]
    if missing_secondary:
        record["error"] = (
            f"ngspice: RuntimeError: missing secondary output(s) ({', '.join(missing_secondary)})"
        )
        _atomic_json(sample_dir / "result.json", record)
        return record

    record["files"].update(
        deck=str(deck_path.relative_to(run_dir)),
        log=str((sample_dir / "ngspice.log").relative_to(run_dir)),
        **{
            key: str(path.relative_to(run_dir))
            for key, enabled, path in (
                ("transient", transient_enabled, transient_waveform),
                ("ac", gain_config.get("enabled", False), gain_waveform),
                ("noise", noise_config.get("enabled", False), noise_waveform),
                ("kickback", kickback_config.get("enabled", False), kickback_waveform),
                ("metrics", offset_config.get("enabled", False), offset_metrics_file),
            )
            if enabled
        },
    )

    if gain_config.get("enabled", False):
        try:
            record["metrics"].update(
                extract_gain(
                    gain_waveform,
                    output_positive=str(gain_config.get("output_positive", "v(X)")),
                    output_negative=str(gain_config.get("output_negative", "v(Y)")),
                    input_positive=str(gain_config.get("input_positive", "v(inp)")),
                    input_negative=str(gain_config.get("input_negative", "v(inn)")),
                )
            )
        except EVALUATION_ERRORS as exc:
            errors.append(f"gain: {type(exc).__name__}: {exc}")

    if offset_config.get("enabled", False):
        try:
            record["metrics"].update(
                extract_offset_metric(
                    offset_metrics_file,
                    key=str(offset_config.get("key", "input_referred_offset_v")),
                )
            )
        except EVALUATION_ERRORS as exc:
            errors.append(f"offset: {type(exc).__name__}: {exc}")

    if noise_config.get("enabled", False):
        try:
            record["metrics"].update(
                extract_noise(
                    noise_waveform,
                    dc_gain_v_per_v=record["metrics"].get("dc_gain_v_per_v"),
                    metrics_path=noise_metrics_file,
                    output_spectrum=str(noise_config.get("output_spectrum", "onoise_spectrum")),
                )
            )
        except EVALUATION_ERRORS as exc:
            errors.append(f"noise: {type(exc).__name__}: {exc}")

    if kickback_config.get("enabled", False):
        try:
            kb_window = kickback_config.get("window_s", (1.0e-9, 3.0e-9))
            record["metrics"].update(
                extract_kickback(
                    kickback_waveform,
                    from_s=float(kb_window[0]),
                    to_s=float(kb_window[1]),
                    differential_vector=str(
                        kickback_config.get("differential_vector", "vkb_diff")
                    ),
                )
            )
        except EVALUATION_ERRORS as exc:
            errors.append(f"kickback: {type(exc).__name__}: {exc}")

    record["error"] = "; ".join(errors)
    delay = record["metrics"].get("delay_s")
    reset_time = record["metrics"].get("reset_time_s")
    period_s = float(transient_config.get("period_s", 0.0))
    power = record["metrics"].get("power_w")
    if delay is not None and float(delay) > 0.0:
        half_period_min_s = float(delay)
        if reset_time is not None and float(reset_time) > 0.0:
            half_period_min_s = max(half_period_min_s, float(reset_time))
        record["metrics"]["speed_hz"] = 1.0 / (2.0 * half_period_min_s)
        record["status"] = "evaluated" if not record["error"] else "partial"
    if power is not None and period_s > 0.0:
        energy = float(power) * period_s
        record["metrics"]["energy_j"] = energy
        record["metrics"]["energy_per_comparison_j"] = energy
    record.update(record["metrics"])
    _atomic_json(sample_dir / "result.json", record)
    return record


def run_rl_sizing(
    *,
    netlist: str | Path,
    runner_name: str,
    algorithm: str,
    samples: int,
    seed: int,
    population_size: int,
    timesteps: int,
    episode_length: int,
    mock_failure_rate: float,
    run_id: str | None,
    timeout_s: float,
    workers: int = 1,
    resume: str | None = None,
    keep: int | None = None,
    verbose: int = 0,
) -> Path:
    """Run PPO/SAC-controlled NRBO with weighted measured scores."""
    algorithm = algorithm.upper()
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if runner_name not in {"ngspice", "mock"}:
        raise ValueError("runner must be ngspice or mock")
    if samples <= 0 or population_size < 4:
        raise ValueError("samples and population size must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if keep is not None and keep < 0:
        raise ValueError("keep must be nonnegative")

    netlist_dir = Path(netlist)
    if not netlist_dir.is_absolute():
        netlist_dir = PROJECT_DIR / netlist_dir
    netlist_dir = netlist_dir.resolve()
    template_path = netlist_dir / "netlist.spice"
    config_path = netlist_dir / "sizing.yaml"
    if not template_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("netlist directory must contain netlist.spice and sizing.yaml")

    sizing_config = load_sizing_config(config_path)
    score_config, score_path = _load_score_config(netlist_dir)
    resolved = resolve_score_config(score_config, sizing_config)

    design_names = variable_names(sizing_config)
    template = template_path.read_text(encoding="utf-8")
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            candidate = resume_path
            if candidate.is_dir():
                resume_path = candidate.resolve()
            elif resume_path.name.startswith("run_"):
                resume_path = (PROJECT_DIR / "run" / netlist_dir.name / resume).resolve()
            else:
                resume_path = (PROJECT_DIR / "run" / netlist_dir.name / f"run_{resume}").resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {resume_path}")
        run_dir = resume_path
        run_id = run_dir.name
        csv_path = run_dir / "results.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"resume run is missing results.csv: {csv_path}")
        existing_samples, prev_algo, _ = _load_existing_results_csv(csv_path, sizing_config)
        if prev_algo is not None and prev_algo != algorithm:
            raise ValueError(
                f"resume algorithm {algorithm!r} does not match run algorithm {prev_algo!r}"
            )
        checkpoint_dir = run_dir / "checkpoints" / algorithm
        checkpoint_model = checkpoint_dir / "model.zip"
        if not checkpoint_model.is_file():
            raise FileNotFoundError(
                f"resume checkpoint model does not exist: {checkpoint_model}"
            )
    else:
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("run_%Y_%m_%d_%H_%M")
        elif not run_id.startswith("run_"):
            run_id = f"run_{run_id}"
        if not re.fullmatch(r"run_[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError(
                "run-id may contain only run_, letters, digits, dot, underscore, and hyphen"
            )
        run_dir = (PROJECT_DIR / "run" / netlist_dir.name / run_id).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints" / algorithm
        existing_samples = []

    (run_dir / "sizing.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "scoring.yaml").write_text(score_path.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)

    if verbose >= 2:
        print(f"Resolved score: {resolved}", file=sys.stderr)
        print(f"Score config: {score_path}", file=sys.stderr)

    all_samples: list[dict[str, Any]] = list(existing_samples)
    records_by_name: dict[str, dict[str, Any]] = {
        str(s["name"]): s for s in all_samples if s.get("name")
    }

    if verbose >= 1:
        keep_label = "all" if keep is None else str(keep)
        print(
            f"Starting {run_id}: algorithm={algorithm}, runner={runner_name}, "
            f"population={population_size}, workers={workers}, "
            f"timesteps={timesteps}, episodes={episode_length}, keep={keep_label}",
            file=sys.stderr,
        )
    runner = (
        MockRunner(seed=seed, failure_rate=mock_failure_rate)
        if runner_name == "mock"
        else NgspiceRun(timeout_s=timeout_s)
    )
    evaluated_by_vector: dict[tuple[float, ...], dict[str, Any]] = {}

    for existing_sample in existing_samples:
        existing_vector = existing_sample.get("x_norm")
        if existing_vector:
            key = tuple(np.round(np.clip(np.asarray(existing_vector, dtype=float), 0.0, 1.0), 10))
            evaluated_by_vector[key] = existing_sample

    def objective_score(record: Mapping) -> float:
        return compute_objective_score(record, resolved)

    def is_feasible(record: Mapping) -> bool:
        return is_record_feasible(record, resolved)

    initial_iteration = max(
        (
            int(existing["iteration"])
            for existing in existing_samples
            if existing.get("iteration") is not None
        ),
        default=0,
    )

    existing_indexes = []
    for existing in existing_samples:
        name = str(existing.get("name", ""))
        suffix = name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            existing_indexes.append(int(suffix))
    next_index = max(existing_indexes, default=0) + 1
    total_evaluations = (next_index - 1) + _expected_evaluations(
        algorithm, population_size, timesteps, episode_length
    )

    kept_records: dict[int, dict[str, Any]] = {}

    def _sample_rank_key(index: int, record: Mapping[str, Any]) -> tuple[bool, float, int]:
        metrics = record.get("metrics") or {}
        feasible = bool(metrics.get("rl_feasible", record.get("rl_feasible", False)))
        raw_score = metrics.get("rl_objective_score", record.get("rl_objective_score"))
        try:
            score = float(raw_score) if raw_score is not None else math.inf
        except (TypeError, ValueError):
            score = math.inf
        if not math.isfinite(score):
            score = math.inf
        return (not feasible, score, index)

    def _prune_sample_dirs() -> None:
        if keep is None or len(kept_records) <= keep:
            return
        ranked_kept = sorted(
            kept_records.items(),
            key=lambda item: _sample_rank_key(item[0], item[1]),
        )
        for evict_index, evict_record in ranked_kept[keep:]:
            sample_dir = _sample_dir(run_dir, evict_index)
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            evict_record["files"] = {}
            kept_records.pop(evict_index, None)

    if keep is not None:
        for existing in existing_samples:
            name = str(existing.get("name", ""))
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                idx = int(suffix)
                if _sample_dir(run_dir, idx).exists():
                    kept_records[idx] = existing
        _prune_sample_dirs()

    def _run_candidate_sample(
        index: int,
        vector: np.ndarray,
        sizes: dict[str, float],
    ) -> dict[str, Any]:
        try:
            return _evaluate_sample(
                index,
                vector,
                sizes,
                run_name=run_id,
                template=template,
                sizing_config=sizing_config,
                runner=runner,
                run_dir=run_dir,
            )
        except EVALUATION_ERRORS as exc:
            sample_dir = _sample_dir(run_dir, index)
            sample_dir.mkdir(parents=True, exist_ok=True)
            return {
                "name": f"{run_id}_{index:06d}",
                "status": "failed",
                "x_norm": [float(value) for value in vector],
                "size": sizes,
                "metrics": {},
                "failure_info": None,
                "files": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _finalize_record(
        key: tuple[float, ...],
        index: int,
        record: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        record["algorithm"] = algorithm
        record["iteration"] = int(iteration)
        record["seed"] = seed
        score_val = objective_score(record)
        violations_str = describe_constraint_violations(record, resolved)
        feasible_flag = violations_str == ""
        record["metrics"]["rl_objective_score"] = score_val
        record["metrics"]["rl_feasible"] = feasible_flag
        record["metrics"]["constraint_violations"] = violations_str
        record["constraint_violations"] = violations_str
        record.update(record["metrics"])
        if verbose >= 1:
            print(
                f"Evaluated {index}/{total_evaluations}: "
                f"score={score_val:.6g}, "
                f"feasible={feasible_flag}, status={record['status']}",
                file=sys.stderr,
            )
        _atomic_json(_sample_dir(run_dir, index) / "result.json", record)
        all_samples.append(record)
        records_by_name[record["name"]] = record
        evaluated_by_vector[key] = record
        if keep is not None:
            kept_records[index] = record
            _prune_sample_dirs()
        if index % CHECKPOINT_INTERVAL == 0:
            if verbose >= 1:
                print(
                    f"Checkpoint: {index}/{total_evaluations} candidates evaluated",
                    file=sys.stderr,
                )
            _write_results_csv(
                run_dir / "results.csv",
                all_samples,
                design_names=design_names,
                algorithm=algorithm,
                seed=seed,
            )
        return record

    def evaluate_candidates(
        vectors: Sequence[np.ndarray],
        iteration: int,
    ) -> list[Mapping]:
        nonlocal next_index
        keys: list[tuple[float, ...]] = []
        pending: list[tuple[tuple[float, ...], int, np.ndarray, dict[str, float]]] = []
        queued_keys: set[tuple[float, ...]] = set()
        for vector in vectors:
            key = tuple(np.round(np.clip(vector, 0.0, 1.0), 10))
            keys.append(key)
            if key in evaluated_by_vector or key in queued_keys:
                continue
            queued_keys.add(key)
            index = next_index
            next_index += 1
            sizes = _physical_sizes(sizing_config, vector)
            pending.append((key, index, vector.copy(), sizes))

        if pending:
            if workers == 1 or len(pending) == 1:
                for key, index, vector, sizes in pending:
                    record = _run_candidate_sample(index, vector, sizes)
                    _finalize_record(key, index, record, iteration)
            else:
                max_workers = min(workers, len(pending))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        (key, index, executor.submit(_run_candidate_sample, index, vector, sizes))
                        for key, index, vector, sizes in pending
                    ]
                    for key, index, future in futures:
                        record = future.result()
                        _finalize_record(key, index, record, iteration)

        return [evaluated_by_vector[key] for key in keys]

    def evaluate_candidate(vector: np.ndarray, iteration: int) -> Mapping:
        return evaluate_candidates([vector], iteration)[0]

    candidates = optimize_with_rl_nrbo(
        algorithm=algorithm,
        dimension=len(design_names),
        population_size=population_size,
        top_k=samples,
        timesteps=timesteps,
        episode_length=episode_length,
        seed=seed,
        evaluate_candidate=evaluate_candidate,
        evaluate_candidates=evaluate_candidates,
        score_candidate=objective_score,
        is_feasible=is_feasible,
        checkpoint_dir=run_dir / "checkpoints" / algorithm,
        resume_model=(
            run_dir / "checkpoints" / algorithm / "model.zip"
            if resume is not None
            else None
        ),
        initial_iteration=initial_iteration,
    )
    if verbose >= 1:
        print(f"Ranking {len(candidates)} measured candidates", file=sys.stderr)
    _write_candidates_csv(
        run_dir / "candidates.csv",
        candidates,
        design_names=design_names,
        algorithm=algorithm,
        seed=seed,
        records_by_name=records_by_name,
    )
    _write_results_csv(
        run_dir / "results.csv",
        all_samples,
        design_names=design_names,
        algorithm=algorithm,
        seed=seed,
    )
    if verbose >= 1:
        print(
            f"Completed {run_id}: {len(all_samples)} measured evaluations, "
            f"{len(candidates)} returned candidates",
            file=sys.stderr,
        )
    return run_dir
