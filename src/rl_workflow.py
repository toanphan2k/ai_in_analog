"""Run direct-feedback PPO/SAC control of the NRBO sizing search."""
from __future__ import annotations

import csv
import json
import logging
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from runner.mock_runner import MockRunner
from runner.ngspice_runner import NgspiceRun
from scripts.deck_generation import render_gain_deck, render_transient_deck
from scripts.extract_delay import extract_delay
from scripts.extract_gain import extract_gain
from scripts.netlist_tools import load_sizing_config
from src.rl_nrbo import optimize_with_rl_nrbo
from src.selector import variable_names

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("ai_in_analog.rl_nrbo")
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
CHECKPOINT_INTERVAL = 25
EVALUATION_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    subprocess.SubprocessError,
)
RESULT_COLUMNS = (
    "sample",
    "status",
    "size_json",
    "delay_s",
    "delay_mean_s",
    "speed_hz",
    "cycles_measured",
    "power_w",
    "output_swing_v",
    "dc_gain_v_per_v",
    "peak_gain_v_per_v",
    "peak_gain_frequency_hz",
    "bandwidth_3db_hz",
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
            if low <= 0.0:
                raise ValueError(f"log-scaled bound for {name} must be positive")
            value = low * (high / low) ** normalized
        else:
            value = low + normalized * (high - low)
        sizes[name] = float(value)
    return sizes


def _write_results_csv(path: Path, samples: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample": sample["name"],
                    "status": sample["status"],
                    "size_json": json.dumps(sample["size"], sort_keys=True),
                    **sample.get("metrics", {}),
                    "error": sample.get("error", ""),
                }
            )


def _run_ngspice(
    runner: NgspiceRun,
    deck_path: Path,
    workdir: Path,
    waveform: Path,
) -> None:
    result = runner.run(deck_path.resolve(), workdir.resolve(), execute=True)
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"ngspice returned {result.get('returncode')}: {result.get('stdout', '')}"
        )
    log_path = workdir / "ngspice.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if "fatal error" in log_text.lower():
        raise RuntimeError("ngspice reported a fatal error; see ngspice.log")
    if not waveform.is_file() or waveform.stat().st_size == 0:
        raise RuntimeError("ngspice did not produce the expected waveform; see ngspice.log")


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
    sample_dir = run_dir / f"sample_{index:06d}"
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

    delay_dir = sample_dir / "delay"
    delay_dir.mkdir()
    transient_name = "transient.txt"
    transient_deck = delay_dir / "transient.spice"
    transient_waveform = delay_dir / transient_name
    deck_text = render_transient_deck(template, sizes, sizing_config, transient_name)
    transient_deck.write_text(deck_text, encoding="utf-8")
    record["files"].update(
        delay_deck=str(transient_deck.relative_to(run_dir)),
        transient=str(transient_waveform.relative_to(run_dir)),
        delay_log=str((delay_dir / "ngspice.log").relative_to(run_dir)),
    )
    try:
        _run_ngspice(runner, transient_deck, delay_dir, transient_waveform)
        power_window = transient_config.get("power_window_s")
        record["metrics"].update(
            extract_delay(
                transient_waveform,
                float(transient_config["vdd_v"]),
                float(transient_config["period_s"]),
                output_positive=str(transient_config.get("output_positive", "v(x)")),
                output_negative=str(transient_config.get("output_negative", "v(y)")),
                clock_vector=str(transient_config.get("clock_vector", "v(ck)")),
                power_vector=str(transient_config.get("power_vector", "p_inst")),
                power_from_s=float(power_window[0]) if power_window else None,
                power_to_s=float(power_window[1]) if power_window else None,
            )
        )
    except EVALUATION_ERRORS as exc:
        record["error"] = f"transient: {type(exc).__name__}: {exc}"

    gain_config = analyses.get("gain", {})
    if gain_config.get("enabled", False):
        gain_dir = sample_dir / "gain"
        gain_dir.mkdir()
        gain_name = "ac.txt"
        gain_deck = gain_dir / "gain.spice"
        gain_waveform = gain_dir / gain_name
        try:
            gain_text = render_gain_deck(template, sizes, sizing_config, gain_name)
            gain_deck.write_text(gain_text, encoding="utf-8")
            record["files"].update(
                gain_deck=str(gain_deck.relative_to(run_dir)),
                ac_waveform=str(gain_waveform.relative_to(run_dir)),
                gain_log=str((gain_dir / "ngspice.log").relative_to(run_dir)),
            )
            _run_ngspice(runner, gain_deck, gain_dir, gain_waveform)
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
            reason = f"gain: {type(exc).__name__}: {exc}"
            record["error"] = f"{record['error']}; {reason}".strip("; ")

    delay = record["metrics"].get("delay_s")
    if delay is not None and float(delay) > 0.0:
        record["metrics"]["speed_hz"] = 1.0 / float(delay)
        record["status"] = "evaluated" if not record["error"] else "partial"
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
    objective: str | None,
    maximize_objective: bool,
    delay_weight: float,
    speed_weight: float,
    power_weight: float,
    power_reference_w: float,
    max_power_w: float | None,
    population_size: int,
    timesteps: int,
    episode_length: int,
    mock_failure_rate: float,
    run_id: str | None,
    timeout_s: float,
) -> Path:
    """Run PPO/SAC-controlled NRBO using measured SPICE/mock feedback only."""
    algorithm = algorithm.upper()
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if runner_name not in {"ngspice", "mock"}:
        raise ValueError("runner must be ngspice or mock")
    if samples <= 0 or population_size < 4:
        raise ValueError("samples must be positive and population_size must be at least 4")
    if min(timesteps, episode_length) <= 0:
        raise ValueError("RL timesteps and episode length must be positive")
    if power_reference_w <= 0.0 or not math.isfinite(power_reference_w):
        raise ValueError("power_reference_w must be finite and positive")
    if max_power_w is not None and (max_power_w <= 0.0 or not math.isfinite(max_power_w)):
        raise ValueError("max_power_w must be finite and positive")
    weights = (delay_weight, speed_weight, power_weight)
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("objective weights must be finite and nonnegative")
    if objective is None and delay_weight + speed_weight + power_weight == 0.0:
        raise ValueError("at least one objective weight must be positive")
    if objective not in {None, "delay_s", "speed_hz", "power_w"}:
        raise ValueError("objective must be delay_s, speed_hz, or power_w")
    if not 0.0 <= mock_failure_rate < 1.0:
        raise ValueError("mock_failure_rate must be in [0, 1)")

    netlist_dir = Path(netlist)
    if not netlist_dir.is_absolute():
        netlist_dir = PROJECT_DIR / netlist_dir
    netlist_dir = netlist_dir.resolve()
    template_path = netlist_dir / "netlist.spice"
    config_path = netlist_dir / "sizing.yaml"
    if not template_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"netlist directory must contain netlist.spice and sizing.yaml: {netlist_dir}"
        )
    sizing_config = load_sizing_config(config_path)
    design_names = variable_names(sizing_config)
    template = template_path.read_text(encoding="utf-8")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and hyphen")
    run_name = f"run_{run_id}"
    run_dir = (PROJECT_DIR / "run" / netlist_dir.name / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; choose another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sizing.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    metadata = {
        "name": run_name,
        "runner": runner_name,
        "algorithm": algorithm,
        "metaheuristic": "NRBO",
        "netlist": str(netlist_dir),
        "seed": seed,
        "population_size": population_size,
        "rl_timesteps": timesteps,
        "episode_length": episode_length,
        "samples_requested": samples,
        "status": "running",
        "samples": [],
    }
    _atomic_json(run_dir / "results.json", metadata)
    runner = (
        MockRunner(seed=seed, failure_rate=mock_failure_rate)
        if runner_name == "mock"
        else NgspiceRun(timeout_s=timeout_s)
    )
    transient_config = sizing_config.get("analyses", {}).get("transient", {})
    delay_reference_s = float(transient_config.get("period_s", 1e-9))
    if delay_reference_s <= 0.0:
        raise ValueError("analyses.transient.period_s must be positive")
    evaluated_by_vector = {}

    def objective_score(record: Mapping) -> float:
        metrics = record.get("metrics", {})
        if record.get("status") == "failed":
            return 1e3
        delay = metrics.get("delay_s")
        power = metrics.get("power_w")
        speed = metrics.get("speed_hz")
        if speed is None and delay is not None and float(delay) > 0.0:
            speed = 1.0 / float(delay)
        if objective == "delay_s":
            if delay is None:
                return 1e3
            value = float(delay) / delay_reference_s
            return -value if maximize_objective else value
        if objective == "speed_hz":
            if speed is None:
                return 1e3
            value = float(speed) * delay_reference_s
            return -value if not maximize_objective else value
        if objective == "power_w":
            if power is None:
                return 1e3
            value = float(power) / power_reference_w
            value = -value if maximize_objective else value
        else:
            if delay_weight and delay is None:
                return 1e3
            if speed_weight and speed is None:
                return 1e3
            if power_weight and power is None:
                return 1e3
            value = delay_weight * (float(delay or 0.0) / delay_reference_s)
            value -= speed_weight * (float(speed or 0.0) * delay_reference_s)
            value += power_weight * (float(power or 0.0) / power_reference_w)
        if max_power_w is not None:
            if power is None:
                return 1e3
            violation = max(0.0, float(power) / max_power_w - 1.0)
            value += 100.0 * violation**2
        return float(value)

    def is_feasible(record: Mapping) -> bool:
        if record.get("status") == "failed":
            return False
        power = record.get("metrics", {}).get("power_w")
        return max_power_w is None or (power is not None and float(power) <= max_power_w)

    def evaluate_candidate(vector: np.ndarray) -> Mapping:
        key = tuple(np.round(np.clip(vector, 0.0, 1.0), 10))
        if key in evaluated_by_vector:
            return evaluated_by_vector[key]
        index = len(metadata["samples"]) + 1
        sizes = _physical_sizes(sizing_config, vector)
        try:
            record = _evaluate_sample(
                index,
                vector,
                sizes,
                run_name=run_name,
                template=template,
                sizing_config=sizing_config,
                runner=runner,
                run_dir=run_dir,
            )
        except EVALUATION_ERRORS as exc:
            sample_dir = run_dir / f"sample_{index:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "name": f"{run_name}_{index:06d}",
                "status": "failed",
                "x_norm": [float(value) for value in vector],
                "size": sizes,
                "metrics": {},
                "files": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        delay = record.get("metrics", {}).get("delay_s")
        if delay is not None and float(delay) > 0.0:
            record["metrics"]["speed_hz"] = 1.0 / float(delay)
        record["metrics"]["rl_objective_score"] = objective_score(record)
        record["metrics"]["rl_feasible"] = is_feasible(record)
        record.update(record["metrics"])
        _atomic_json(run_dir / f"sample_{index:06d}" / "result.json", record)
        metadata["samples"].append(record)
        metadata["completed_samples"] = len(metadata["samples"])
        metadata["status"] = (
            "complete"
            if all(item.get("status") == "evaluated" for item in metadata["samples"])
            else "partial"
        )
        evaluated_by_vector[key] = record
        if index % CHECKPOINT_INTERVAL == 0:
            _write_results_csv(run_dir / "results.csv", metadata["samples"])
            _atomic_json(run_dir / "results.json", metadata)
        return record

    candidates = optimize_with_rl_nrbo(
        algorithm=algorithm,
        dimension=len(design_names),
        population_size=population_size,
        top_k=samples,
        timesteps=timesteps,
        episode_length=episode_length,
        seed=seed,
        evaluate_candidate=evaluate_candidate,
        score_candidate=objective_score,
        is_feasible=is_feasible,
        policy_path=run_dir / "rl_controller",
    )
    candidate_payload = {
        "algorithm": algorithm,
        "metaheuristic": "NRBO",
        "objective": objective,
        "delay_weight": delay_weight,
        "speed_weight": speed_weight,
        "power_weight": power_weight,
        "max_power_w": max_power_w,
        "rl_timesteps": timesteps,
        "episode_length": episode_length,
        "population_size": population_size,
        "candidates": candidates,
        "note": "Candidates are ranked from direct measurements; no surrogate model is used.",
    }
    _atomic_json(run_dir / "candidates.json", candidate_payload)
    with (run_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "sample",
                "status",
                "score",
                "feasible",
                "size_json",
                "metrics_json",
            ),
        )
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "sample": candidate["sample"],
                    "status": candidate["status"],
                    "score": candidate["score"],
                    "feasible": candidate["feasible"],
                    "size_json": json.dumps(candidate["size"], sort_keys=True),
                    "metrics_json": json.dumps(candidate["metrics"], sort_keys=True),
                }
            )
    _write_results_csv(run_dir / "results.csv", metadata["samples"])
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["candidate_count"] = len(candidates)
    _atomic_json(run_dir / "results.json", metadata)
    LOGGER.info("RL-NRBO search run completed.")
    return run_dir
