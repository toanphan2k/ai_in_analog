"""Direct-feedback PPO/SAC control of NRBO using YAML-weighted scoring."""
from __future__ import annotations

import csv
import json
import logging
import math
import re
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
from scripts.extract_delay import extract_delay
from scripts.extract_gain import extract_gain
from scripts.extract_kickback import extract_kickback
from scripts.extract_noise import extract_noise
from scripts.extract_offset import extract_offset_metric
from scripts.netlist_tools import load_sizing_config
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
    "dc_gain_v_per_v",
    "peak_gain_v_per_v",
    "peak_gain_frequency_hz",
    "bandwidth_3db_hz",
    "samples",
    "rl_objective_score",
    "rl_feasible",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
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
                "error": sample.get("error", ""),
            }
            if isinstance(size, Mapping):
                for column in size_columns:
                    if column in size:
                        row[column] = size[column]
            for column in RESULT_METRIC_COLUMNS:
                if column in metrics:
                    row[column] = metrics[column]
            writer.writerow(row)


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
    max_power_w = _parse_optional_positive(constraints, "max_power_w")
    max_offset_v = _parse_optional_positive(constraints, "max_offset_v")
    max_noise_v = _parse_optional_positive(constraints, "max_noise_v")
    max_kickback_v = _parse_optional_positive(constraints, "max_kickback_v")
    min_output_swing_v = _parse_optional_positive(constraints, "min_output_swing_v")

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
        "max_power_w": max_power_w,
        "max_offset_v": max_offset_v,
        "max_noise_v": max_noise_v,
        "max_kickback_v": max_kickback_v,
        "min_output_swing_v": min_output_swing_v,
        "penalty_lambda": penalty_lambda,
        "penalty_type": penalty_type,
    }


def _extract_scored_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = record.get("metrics", {})
    delay = _finite_metric(metrics.get("delay_s"))
    power = _finite_metric(metrics.get("power_w"))
    energy = _finite_metric(metrics.get("energy_per_comparison_j", metrics.get("energy_j")))
    offset = _finite_metric(metrics.get("input_referred_offset_v", metrics.get("offset_v")))
    noise = _finite_metric(metrics.get("input_referred_noise_v"))
    kickback = _finite_metric(metrics.get("kickback_noise_v"))
    output_swing = _finite_metric(metrics.get("output_swing_v"))
    return {
        "delay_s": delay,
        "power_w": power,
        "energy_j": energy,
        "offset_v": offset,
        "noise_v": noise,
        "kickback_v": kickback,
        "output_swing_v": output_swing,
    }


def compute_objective_score(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> float:
    """Compute S_total (lower is better) with normalized terms and constraint penalties."""
    infeasible_score = float(resolved["infeasible_score"])
    if record.get("status") == "failed":
        return infeasible_score

    m = _extract_scored_metrics(record)
    delay = m["delay_s"]
    power = m["power_w"]
    energy = m["energy_j"]
    offset = m["offset_v"]
    noise = m["noise_v"]
    kickback = m["kickback_v"]
    output_swing = m["output_swing_v"]

    if (
        (resolved["delay_weight"] > 0.0 and delay is None)
        or (resolved["power_weight"] > 0.0 and power is None)
        or (resolved["energy_weight"] > 0.0 and energy is None)
        or (resolved["offset_weight"] > 0.0 and offset is None)
        or (resolved["noise_weight"] > 0.0 and noise is None)
        or (resolved["kickback_weight"] > 0.0 and kickback is None)
        or (resolved["max_delay_s"] is not None and delay is None)
        or (resolved["max_power_w"] is not None and power is None)
        or (resolved["max_offset_v"] is not None and offset is None)
        or (resolved["max_noise_v"] is not None and noise is None)
        or (resolved["max_kickback_v"] is not None and kickback is None)
        or (resolved["min_output_swing_v"] is not None and output_swing is None)
    ):
        return infeasible_score

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

    penalty_lambda = float(resolved["penalty_lambda"])
    if penalty_lambda > 0.0 and violations:
        if resolved["penalty_type"] == "quadratic":
            value += penalty_lambda * sum(g * g for g in violations)
        else:
            value += penalty_lambda * sum(violations)

    return float(value) if math.isfinite(value) else infeasible_score


def is_record_feasible(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> bool:
    """Return True iff the candidate succeeded, has all required metrics, and meets all constraints."""
    if record.get("status") == "failed":
        return False

    m = _extract_scored_metrics(record)
    delay = m["delay_s"]
    power = m["power_w"]
    energy = m["energy_j"]
    offset = m["offset_v"]
    noise = m["noise_v"]
    kickback = m["kickback_v"]
    output_swing = m["output_swing_v"]

    if (
        (resolved["delay_weight"] > 0.0 and delay is None)
        or (resolved["power_weight"] > 0.0 and power is None)
        or (resolved["energy_weight"] > 0.0 and energy is None)
        or (resolved["offset_weight"] > 0.0 and offset is None)
        or (resolved["noise_weight"] > 0.0 and noise is None)
        or (resolved["kickback_weight"] > 0.0 and kickback is None)
    ):
        return False

    if resolved["max_delay_s"] is not None and (delay is None or delay > resolved["max_delay_s"]):
        return False
    if resolved["max_power_w"] is not None and (power is None or power > resolved["max_power_w"]):
        return False
    if resolved["max_offset_v"] is not None and (
        offset is None or abs(offset) > resolved["max_offset_v"]
    ):
        return False
    if resolved["max_noise_v"] is not None and (noise is None or noise > resolved["max_noise_v"]):
        return False
    if resolved["max_kickback_v"] is not None and (
        kickback is None or kickback > resolved["max_kickback_v"]
    ):
        return False
    if resolved["min_output_swing_v"] is not None and (
        output_swing is None or output_swing < resolved["min_output_swing_v"]
    ):
        return False
    return True


def _run_ngspice(
    runner: NgspiceRun,
    deck: Path,
    workdir: Path,
    expected_outputs: Sequence[Path],
) -> None:
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
        for output in expected_outputs
        if not output.is_file() or output.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"ngspice did not produce expected output(s) ({', '.join(missing)}); see ngspice.log"
        )


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
    record = {
        "name": sample_name,
        "status": "failed",
        "x_norm": [float(value) for value in vector],
        "size": dict(sizes),
        "metrics": {},
        "files": {},
        "error": "",
    }
    _atomic_json(sample_dir / "size.json", {"name": sample_name, "size": sizes})
    analyses = sizing_config.get("analyses", {})
    transient_config = analyses.get("transient", {})

    if isinstance(runner, MockRunner):
        record.update(runner.evaluate(
            sizes,
            np.asarray(vector, dtype=float),
            sample_index=index,
            vdd_v=float(transient_config.get("vdd_v", 1.8)),
        ))
        record.update(record["metrics"])
        _atomic_json(sample_dir / "result.json", record)
        return record

    deck_path = sample_dir / "netlist.spice"
    deck_text = render_sized_deck(template, sizes, sizing_config)
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

    expected_outputs: list[Path] = []
    if transient_config.get("enabled", True):
        expected_outputs.append(transient_waveform)
    if gain_config.get("enabled", False):
        expected_outputs.append(gain_waveform)
    if offset_config.get("enabled", False):
        expected_outputs.append(offset_metrics_file)
    if noise_config.get("enabled", False):
        expected_outputs.append(noise_waveform)
    if kickback_config.get("enabled", False):
        expected_outputs.append(kickback_waveform)
    deduped_outputs = list(dict.fromkeys(expected_outputs))

    record["files"].update(
        deck=str(deck_path.relative_to(run_dir)),
        log=str((sample_dir / "ngspice.log").relative_to(run_dir)),
        **{
            key: str(path.relative_to(run_dir))
            for key, enabled, path in (
                ("transient", transient_config.get("enabled", True), transient_waveform),
                ("ac", gain_config.get("enabled", False), gain_waveform),
                ("noise", noise_config.get("enabled", False), noise_waveform),
                ("kickback", kickback_config.get("enabled", False), kickback_waveform),
                ("metrics", offset_config.get("enabled", False), offset_metrics_file),
            )
            if enabled
        },
    )

    errors: list[str] = []
    try:
        _run_ngspice(runner, deck_path, sample_dir, deduped_outputs)
    except EVALUATION_ERRORS as exc:
        record["error"] = f"ngspice: {type(exc).__name__}: {exc}"
        _atomic_json(sample_dir / "result.json", record)
        return record

    if transient_config.get("enabled", True):
        try:
            power_window = transient_config.get("power_window_s")
            record["metrics"].update(extract_delay(
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
            ))
        except EVALUATION_ERRORS as exc:
            errors.append(f"transient: {type(exc).__name__}: {exc}")

    if gain_config.get("enabled", False):
        try:
            record["metrics"].update(extract_gain(
                gain_waveform,
                output_positive=str(gain_config.get("output_positive", "v(X)")),
                output_negative=str(gain_config.get("output_negative", "v(Y)")),
                input_positive=str(gain_config.get("input_positive", "v(inp)")),
                input_negative=str(gain_config.get("input_negative", "v(inn)")),
            ))
        except EVALUATION_ERRORS as exc:
            errors.append(f"gain: {type(exc).__name__}: {exc}")

    if offset_config.get("enabled", False):
        try:
            record["metrics"].update(extract_offset_metric(
                offset_metrics_file,
                key=str(offset_config.get("key", "input_referred_offset_v")),
            ))
        except EVALUATION_ERRORS as exc:
            errors.append(f"offset: {type(exc).__name__}: {exc}")

    if noise_config.get("enabled", False):
        try:
            record["metrics"].update(extract_noise(
                noise_waveform,
                dc_gain_v_per_v=record["metrics"].get("dc_gain_v_per_v"),
                metrics_path=noise_metrics_file,
                output_spectrum=str(noise_config.get("output_spectrum", "onoise_spectrum")),
            ))
        except EVALUATION_ERRORS as exc:
            errors.append(f"noise: {type(exc).__name__}: {exc}")

    if kickback_config.get("enabled", False):
        try:
            kb_window = kickback_config.get("window_s", (1.0e-9, 3.0e-9))
            record["metrics"].update(extract_kickback(
                kickback_waveform,
                from_s=float(kb_window[0]),
                to_s=float(kb_window[1]),
                differential_vector=str(
                    kickback_config.get("differential_vector", "vkb_diff")
                ),
            ))
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
        results_path = run_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"resume run is missing results.json: {results_path}")
        previous_metadata = json.loads(results_path.read_text(encoding="utf-8"))
        if previous_metadata.get("algorithm") != algorithm:
            raise ValueError(
                f"resume algorithm {algorithm!r} does not match "
                f"run algorithm {previous_metadata.get('algorithm')!r}"
            )
        checkpoint_dir = run_dir / "checkpoints" / algorithm
        checkpoint_model = checkpoint_dir / "model.zip"
        if not checkpoint_model.is_file():
            raise FileNotFoundError(
                f"resume checkpoint model does not exist: {checkpoint_model}"
            )
        existing_samples = previous_metadata.get("samples", [])
        previous_algorithm = previous_metadata.get("algorithm")
        previous_seed = previous_metadata.get("seed")
        for existing_sample in existing_samples:
            if "algorithm" not in existing_sample and previous_algorithm is not None:
                existing_sample["algorithm"] = previous_algorithm
            if "seed" not in existing_sample and previous_seed is not None:
                existing_sample["seed"] = previous_seed
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
        checkpoint_model = checkpoint_dir / "model.zip"
        existing_samples = []
    (run_dir / "sizing.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "scoring.yaml").write_text(score_path.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)

    if verbose >= 2:
        print(f"Resolved score: {resolved}", file=sys.stderr)
        print(f"Score config: {score_path}", file=sys.stderr)
    metadata = {
        "name": run_id,
        "runner": runner_name,
        "algorithm": algorithm,
        "metaheuristic": "NRBO",
        "netlist": str(netlist_dir),
        "seed": seed,
        "population_size": population_size,
        "workers": workers,
        "rl_timesteps": timesteps,
        "episode_length": episode_length,
        "samples_requested": samples,
        "resumed_from": str(run_dir) if resume is not None else None,
        "checkpoint_dir": str(checkpoint_dir) if resume is not None or checkpoint_dir else None,
        "score_config": str(score_path),
        "resolved_score": resolved,
        "status": "running",
        "samples": existing_samples,
    }
    _atomic_json(run_dir / "results.json", metadata)

    if verbose >= 1:
        print(
            f"Starting {run_id}: algorithm={algorithm}, runner={runner_name}, "
            f"population={population_size}, workers={workers}, "
            f"timesteps={timesteps}, episodes={episode_length}",
            file=sys.stderr,
        )
    runner = (
        MockRunner(seed=seed, failure_rate=mock_failure_rate)
        if runner_name == "mock"
        else NgspiceRun(timeout_s=timeout_s)
    )
    evaluated_by_vector = {}

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
        record["metrics"]["rl_objective_score"] = objective_score(record)
        record["metrics"]["rl_feasible"] = is_feasible(record)
        record.update(record["metrics"])
        if verbose >= 1:
            print(
                f"Evaluated {index}/{run_id}: "
                f"score={record['metrics']['rl_objective_score']:.6g}, "
                f"feasible={record['metrics']['rl_feasible']}, status={record['status']}",
                file=sys.stderr,
            )
        _atomic_json(_sample_dir(run_dir, index) / "result.json", record)
        metadata["samples"].append(record)
        metadata["completed_samples"] = len(metadata["samples"])
        metadata["status"] = (
            "complete"
            if all(item.get("status") == "evaluated" for item in metadata["samples"])
            else "partial"
        )
        evaluated_by_vector[key] = record
        if index % CHECKPOINT_INTERVAL == 0:
            if verbose >= 1:
                print(f"Checkpoint: {index} candidates evaluated", file=sys.stderr)
            _write_results_csv(
                run_dir / "results.csv",
                metadata["samples"],
                design_names=design_names,
                algorithm=algorithm,
                seed=seed,
            )
            _atomic_json(run_dir / "results.json", metadata)
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
    candidate_payload = {
        "algorithm": algorithm,
        "metaheuristic": "NRBO",
        "score_config": str(score_path),
        "resolved_score": resolved,
        "candidates": candidates,
        "note": "Candidates are ranked from direct measurements; no surrogate model is used.",
    }
    _atomic_json(run_dir / "candidates.json", candidate_payload)
    _write_results_csv(
        run_dir / "results.csv",
        metadata["samples"],
        design_names=design_names,
        algorithm=algorithm,
        seed=seed,
    )
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["candidate_count"] = len(candidates)
    _atomic_json(run_dir / "results.json", metadata)
    if verbose >= 1:
        print(
            f"Completed {run_id}: {len(metadata['samples'])} measured evaluations, "
            f"{len(candidates)} returned candidates",
            file=sys.stderr,
        )
    return run_dir
