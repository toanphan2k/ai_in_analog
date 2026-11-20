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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
RESULT_COLUMNS = (
    "sample",
    "status",
    "size_json",
    "delay_s",
    "delay_mean_s",
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


def _write_results_csv(path: Path, samples: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            metrics = sample.get("metrics", {})
            row = {
                "sample": sample["name"],
                "status": sample["status"],
                "size_json": json.dumps(sample["size"], sort_keys=True),
                "error": sample.get("error", ""),
            }
            for column in RESULT_COLUMNS:
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
    weights = payload.get("weights")
    references = payload.get("references")
    constraints = payload.get("constraints", {})
    if not isinstance(weights, dict) or not isinstance(references, dict):
        raise ValueError("score config must define 'weights' and 'references'")
    if not isinstance(constraints, dict):
        raise ValueError("score config 'constraints' must be a mapping")
    return payload, path


def _run_ngspice(runner: NgspiceRun, deck: Path, workdir: Path, waveform: Path) -> None:
    result = runner.run(deck.resolve(), workdir.resolve(), execute=True)
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
        record.update(runner.evaluate(
            sizes,
            np.asarray(vector, dtype=float),
            sample_index=index,
            vdd_v=float(transient_config.get("vdd_v", 1.8)),
        ))
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
    )
    try:
        _run_ngspice(runner, transient_deck, delay_dir, transient_waveform)
        power_window = transient_config.get("power_window_s")
        metrics = extract_delay(
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
        record["metrics"].update(metrics)
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
            _run_ngspice(runner, gain_deck, gain_dir, gain_waveform)
            record["metrics"].update(extract_gain(
                gain_waveform,
                output_positive=str(gain_config.get("output_positive", "v(X)")),
                output_negative=str(gain_config.get("output_negative", "v(Y)")),
                input_positive=str(gain_config.get("input_positive", "v(inp)")),
                input_negative=str(gain_config.get("input_negative", "v(inn)")),
            ))
        except EVALUATION_ERRORS as exc:
            reason = f"gain: {type(exc).__name__}: {exc}"
            record["error"] = f"{record['error']}; {reason}".strip("; ")

    delay = record["metrics"].get("delay_s")
    period_s = float(transient_config.get("period_s", 0.0))
    power = record["metrics"].get("power_w")
    if delay is not None and float(delay) > 0.0:
        record["metrics"]["speed_hz"] = 1.0 / float(delay)
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
    weights = score_config["weights"]
    references = score_config["references"]
    constraints = score_config["constraints"]

    delay_weight = float(weights.get("delay", 1.0))
    speed_weight = float(weights.get("speed", 0.0))
    power_weight = float(weights.get("power", 0.0))
    energy_weight = float(weights.get("energy", 0.0))
    offset_weight = float(weights.get("offset", 0.0))
    noise_weight = float(weights.get("noise", 0.0))
    kickback_weight = float(weights.get("kickback", 0.0))

    delay_reference_s = float(
        sizing_config.get("analyses", {}).get("transient", {}).get("period_s", 1e-9)
    )
    power_reference_w = float(references.get("power_w", 0.01))
    energy_reference_j = float(references.get("energy_j", 1e-15))
    offset_reference_v = float(references.get("offset_v", 1e-3))
    noise_reference_v = float(references.get("noise_v", 1e-3))
    kickback_reference_v = float(references.get("kickback_v", 1e-3))

    resolved_weights = (
        delay_weight, speed_weight, power_weight, energy_weight,
        offset_weight, noise_weight, kickback_weight,
    )
    resolved_references = (
        delay_reference_s, power_reference_w, energy_reference_j,
        offset_reference_v, noise_reference_v, kickback_reference_v,
    )
    if any(weight < 0.0 or not math.isfinite(weight) for weight in resolved_weights):
        raise ValueError("objective weights must be finite and nonnegative")
    if sum(resolved_weights) == 0.0:
        raise ValueError("at least one objective weight must be positive")
    if any(reference <= 0.0 or not math.isfinite(reference) for reference in resolved_references):
        raise ValueError("metric references must be finite and positive")

    max_power_w = constraints.get("max_power_w")
    if max_power_w is not None:
        max_power_w = float(max_power_w)
        if max_power_w <= 0.0 or not math.isfinite(max_power_w):
            raise ValueError("max_power_w must be finite and positive")

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

    resolved = {
        "delay_weight": delay_weight,
        "speed_weight": speed_weight,
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
        "max_power_w": max_power_w,
    }
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
            f"population={population_size}, timesteps={timesteps}, episodes={episode_length}",
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
        metrics = record.get("metrics", {})
        if record.get("status") == "failed":
            return 1e3
        delay = metrics.get("delay_s")
        power = metrics.get("power_w")
        speed = metrics.get("speed_hz")
        if speed is None and delay is not None and float(delay) > 0.0:
            speed = 1.0 / float(delay)
        energy = metrics.get("energy_per_comparison_j", metrics.get("energy_j"))
        offset = metrics.get("input_referred_offset_v", metrics.get("offset_v"))
        noise = metrics.get("input_referred_noise_v")
        kickback = metrics.get("kickback_noise_v")
        if delay_weight and delay is None:
            return 1e3
        if speed_weight and speed is None:
            return 1e3
        if power_weight and power is None:
            return 1e3
        if energy_weight and energy is None:
            return 1e3
        if offset_weight and offset is None:
            return 1e3
        if noise_weight and noise is None:
            return 1e3
        if kickback_weight and kickback is None:
            return 1e3
        value = delay_weight * (float(delay or 0.0) / delay_reference_s)
        value -= speed_weight * (float(speed or 0.0) * delay_reference_s)
        value += power_weight * (float(power or 0.0) / power_reference_w)
        value += energy_weight * (float(energy or 0.0) / energy_reference_j)
        value += offset_weight * abs(float(offset or 0.0)) / offset_reference_v
        value += noise_weight * (float(noise or 0.0) / noise_reference_v)
        value += kickback_weight * (float(kickback or 0.0) / kickback_reference_v)
        if max_power_w is not None:
            violation = max(0.0, float(power or 0.0) / max_power_w - 1.0)
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
        existing_indexes = []
        for existing in existing_samples:
            name = str(existing.get("name", ""))
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing_indexes.append(int(suffix))
        index = max(existing_indexes, default=0) + 1
        sizes = _physical_sizes(sizing_config, vector)
        try:
            record = _evaluate_sample(
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
            sample_dir = run_dir / f"sample_{index:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "name": f"{run_id}_{index:06d}",
                "status": "failed",
                "x_norm": [float(value) for value in vector],
                "size": sizes,
                "metrics": {},
                "files": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
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
            if verbose >= 1:
                print(f"Checkpoint: {index} candidates evaluated", file=sys.stderr)
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
        checkpoint_dir=run_dir / "checkpoints" / algorithm,
        resume_model=(
            run_dir / "checkpoints" / algorithm / "model.zip"
            if resume is not None
            else None
        ),
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
    _write_results_csv(run_dir / "results.csv", metadata["samples"])
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
