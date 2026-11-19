#!/usr/bin/env python3
"""Generate, simulate, and record a batch of sized SPICE netlists."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import qmc

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from runner.ngspice_runner import NgspiceRun  # noqa: E402
from runner.mock_runner import MockRunner  # noqa: E402
from scripts.deck_generation import (  # noqa: E402
    render_gain_deck,
    render_transient_deck,
)
from scripts.extract_delay import extract_delay  # noqa: E402
from scripts.extract_gain import extract_gain  # noqa: E402
from scripts.netlist_tools import load_sizing_config  # noqa: E402
from optimize_nrbo import optimize_candidates  # noqa: E402
from src.selector import candidate_pool, variable_names  # noqa: E402


TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
LOGGER = logging.getLogger("ai_in_analog.optimize")

CSV_COLUMNS = (
    "sample",
    "status",
    "size_json",
    "delay_s",
    "delay_mean_s",
    "cycles_measured",
    "power_w",
    "output_swing_v",
    "dc_gain_v_per_v",
    "peak_gain_v_per_v",
    "peak_gain_frequency_hz",
    "bandwidth_3db_hz",
    "surrogate_delay_s",
    "surrogate_power_w",
    "delay_abs_error_s",
    "power_abs_error_w",
    "delay_ape_percent",
    "power_ape_percent",
    "samples",
    "error",
)
RUN_CHECKPOINT_INTERVAL = 25


def _configure_logging(verbosity: int) -> None:
    levels = (logging.WARNING, logging.INFO, logging.DEBUG, TRACE_LEVEL)
    level = levels[min(verbosity, len(levels) - 1)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _resolve_netlist_dir(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"netlist path is not a directory: {path}")
    return path


def _default_run_id() -> str:
    time_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return time_id


def _physical_sizes(config: Mapping, vector: Sequence[float]) -> dict[str, float]:
    names = variable_names(config)
    bounds = config.get("optimization", {}).get("bounds", {})
    if len(vector) != len(names):
        raise ValueError(f"expected {len(names)} design values, received {len(vector)}")
    sizes = {}
    for name, normalized in zip(names, vector):
        if name not in bounds:
            raise ValueError(f"optimization.bounds is missing {name!r}")
        bound = bounds[name]
        low, high = float(bound["min"]), float(bound["max"])
        if not 0 <= float(normalized) <= 1 or high <= low:
            raise ValueError(f"invalid bounds or normalized value for {name}")
        if bound.get("scale", "linear") == "log":
            if low <= 0:
                raise ValueError(f"log-scaled bound for {name} must be positive")
            value = low * (high / low) ** float(normalized)
        else:
            value = low + float(normalized) * (high - low)
        sizes[name] = float(value)
    return sizes


def _normalized_vector(config: Mapping, sizes: Mapping[str, float]) -> np.ndarray:
    names = variable_names(config)
    bounds = config.get("optimization", {}).get("bounds", {})
    if set(sizes) != set(names):
        raise ValueError("candidate sizing keys must exactly match optimization.variables")
    values = []
    for name in names:
        low, high = float(bounds[name]["min"]), float(bounds[name]["max"])
        value = float(sizes[name])
        if not low <= value <= high:
            raise ValueError(f"candidate {name}={value} is outside [{low}, {high}]")
        if bounds[name].get("scale", "linear") == "log":
            values.append(math.log(value / low) / math.log(high / low))
        else:
            values.append((value - low) / (high - low))
    return np.asarray(values, dtype=float)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, samples: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
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


def _sample_vectors(
    algorithm: str,
    count: int,
    sizing_config: Mapping,
    seed: int,
) -> np.ndarray:
    dimension = len(variable_names(sizing_config))
    if algorithm == "halton":
        return candidate_pool(count, sizing_config, seed)
    if algorithm == "random":
        return np.random.default_rng(seed).random((count, dimension))
    if algorithm == "lhs":
        rng = np.random.default_rng(seed)
        strata = (np.arange(count)[:, None] + rng.random((count, dimension))) / count
        return np.column_stack([rng.permutation(strata[:, axis]) for axis in range(dimension)])
    if algorithm == "sobol":
        power = int(math.ceil(math.log2(count)))
        return qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)[:count]
    raise ValueError(f"unsupported sampling algorithm {algorithm!r}")


def _run_deck(runner: NgspiceRun, deck_path: Path, workdir: Path, waveform: Path) -> None:
    LOGGER.debug("Running ngspice deck %s (cwd: %s)", deck_path, workdir)
    command = runner.command(deck_path.resolve(), workdir.resolve())
    LOGGER.debug("ngspice command: %s", " ".join(command))
    result = runner.run(deck_path.resolve(), workdir.resolve(), execute=True)
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"ngspice returned {result.get('returncode')}: {result.get('stdout', '')}"
        )
    log_path = workdir / "ngspice.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    LOGGER.log(TRACE_LEVEL, "ngspice log %s:\n%s", log_path, log_text)
    if "fatal error" in log_text.lower():
        raise RuntimeError("ngspice reported a fatal error; see ngspice.log")
    if not waveform.is_file() or waveform.stat().st_size == 0:
        raise RuntimeError("ngspice did not produce the expected waveform; see ngspice.log")
    LOGGER.debug("ngspice produced %s (%d bytes)", waveform, waveform.stat().st_size)


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
    sample_name = f"{run_name}_{index:03d}"
    LOGGER.info("Evaluating sample %d: %s", index, sample_name)
    LOGGER.debug("Sample %s sizing: %s", sample_name, dict(sizes))
    LOGGER.log(TRACE_LEVEL, "Sample %s normalized vector: %s", sample_name, list(vector))
    sample_dir = run_dir / f"sample_{index:03d}"
    sample_dir.mkdir(parents=True)
    record = {
        "name": sample_name,
        "status": "failed",
        "x_norm": [float(x) for x in vector],
        "size": dict(sizes),
        "metrics": {},
        "files": {},
        "error": "",
    }
    _atomic_json(sample_dir / "size.json", {"name": sample_name, "size": sizes})

    analysis_config = sizing_config.get("analyses", {})
    transient_config = analysis_config.get("transient", {})
    if isinstance(runner, MockRunner):
        mock_result = runner.evaluate(
            sizes,
            np.asarray(vector, dtype=float),
            sample_index=index,
            vdd_v=float(transient_config.get("vdd_v", 1.8)),
        )
        record.update(mock_result)
        record.update(record["metrics"])
        _atomic_json(sample_dir / "result.json", record)
        LOGGER.info("Mock sample %s status: %s", sample_name, record["status"])
        return record

    delay_dir = sample_dir / "delay"
    delay_dir.mkdir()
    transient_file = "transient.txt"
    delay_deck = delay_dir / "strongarm_delay.spice"
    delay_waveform = delay_dir / transient_file
    transient_deck = render_transient_deck(template, sizes, sizing_config, transient_file)
    delay_deck.write_text(transient_deck, encoding="utf-8")
    LOGGER.log(TRACE_LEVEL, "Generated transient deck %s:\n%s", delay_deck, transient_deck)
    record["files"].update(
        delay_deck=str(delay_deck.relative_to(run_dir)),
        transient=str(delay_waveform.relative_to(run_dir)),
        delay_log=str((delay_dir / "ngspice.log").relative_to(run_dir)),
    )
    try:
        _run_deck(runner, delay_deck, delay_dir, delay_waveform)
        power_window = transient_config.get("power_window_s")
        delay_metrics = extract_delay(
            delay_waveform,
            float(transient_config["vdd_v"]),
            float(transient_config["period_s"]),
            output_positive=str(transient_config.get("output_positive", "v(x)")),
            output_negative=str(transient_config.get("output_negative", "v(y)")),
            clock_vector=str(transient_config.get("clock_vector", "v(ck)")),
            power_vector=str(transient_config.get("power_vector", "p_inst")),
            power_from_s=float(power_window[0]) if power_window else None,
            power_to_s=float(power_window[1]) if power_window else None,
        )
        record["metrics"].update(delay_metrics)
        record["status"] = "partial"
        LOGGER.info("%s transient metrics: %s", sample_name, delay_metrics)
    except Exception as exc:
        record["error"] = f"transient: {type(exc).__name__}: {exc}"
        LOGGER.warning("%s transient analysis failed: %s", sample_name, exc)
        LOGGER.debug("Transient analysis exception", exc_info=True)

    gain_config = analysis_config.get("gain", {})
    if gain_config.get("enabled", False):
        gain_dir = sample_dir / "gain"
        gain_dir.mkdir()
        gain_file = "ac.txt"
        gain_deck = gain_dir / "strongarm_gain.spice"
        gain_waveform = gain_dir / gain_file
        try:
            ac_deck = render_gain_deck(template, sizes, sizing_config, gain_file)
            gain_deck.write_text(ac_deck, encoding="utf-8")
            LOGGER.log(TRACE_LEVEL, "Generated gain deck %s:\n%s", gain_deck, ac_deck)
            record["files"].update(
                gain_deck=str(gain_deck.relative_to(run_dir)),
                ac_waveform=str(gain_waveform.relative_to(run_dir)),
                gain_log=str((gain_dir / "ngspice.log").relative_to(run_dir)),
            )
            _run_deck(runner, gain_deck, gain_dir, gain_waveform)
            gain_metrics = extract_gain(
                gain_waveform,
                output_positive=str(gain_config.get("output_positive", "v(X)")),
                output_negative=str(gain_config.get("output_negative", "v(Y)")),
                input_positive=str(gain_config.get("input_positive", "v(inp)")),
                input_negative=str(gain_config.get("input_negative", "v(inn)")),
            )
            record["metrics"].update(gain_metrics)
            LOGGER.info("%s gain metrics: %s", sample_name, gain_metrics)
        except Exception as exc:
            reason = f"gain: {type(exc).__name__}: {exc}"
            record["error"] = f"{record['error']}; {reason}".strip("; ")
            LOGGER.warning("%s gain analysis failed: %s", sample_name, exc)
            LOGGER.debug("Gain analysis exception", exc_info=True)

    if record["metrics"].get("delay_s") is not None:
        record["status"] = "evaluated" if not record["error"] else "partial"
    LOGGER.info("Sample %s status: %s", sample_name, record["status"])
    record.update(record["metrics"])
    _atomic_json(sample_dir / "result.json", record)
    return record


def optimize(
    *,
    netlist: str | Path,
    runner_name: str = "ngspice",
    samples: int = 10,
    seed: int = 0,
    algorithm: str = "halton",
    model_path: str | Path | None = None,
    objective: str | None = None,
    maximize_objective: bool = False,
    delay_weight: float = 1.0,
    speed_weight: float = 0.0,
    power_weight: float = 0.0,
    max_power_w: float | None = None,
    candidates_path: str | Path | None = None,
    min_valid_probability: float = 0.5,
    neighbors_per_candidate: int = 0,
    neighbor_radius: float = 0.02,
    iterations: int = 60,
    population_size: int = 24,
    mock_failure_rate: float = 0.03,
    run_id: str | None = None,
    timeout_s: float = 120.0,
) -> Path:
    """Sample, optimize against a surrogate, or verify selected designs with SPICE."""
    if runner_name not in {"ngspice", "mock"}:
        raise ValueError(f"unsupported runner {runner_name!r}; available: ngspice, mock")
    if not 0.0 <= mock_failure_rate < 1.0:
        raise ValueError("mock_failure_rate must be in [0, 1)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if algorithm not in {"halton", "random", "lhs", "sobol", "nrbo", "verify"}:
        raise ValueError("algorithm must be halton, random, lhs, sobol, nrbo, or verify")
    if algorithm == "nrbo" and model_path is None:
        raise ValueError("--model is required when --algorithm nrbo")
    if algorithm != "nrbo" and model_path is not None:
        raise ValueError("--model can only be used with --algorithm nrbo")
    if algorithm == "verify" and candidates_path is None:
        raise ValueError("--candidates is required when --algorithm verify")
    if algorithm != "verify" and candidates_path is not None:
        raise ValueError("--candidates can only be used with --algorithm verify")
    if neighbors_per_candidate < 0:
        raise ValueError("neighbors_per_candidate must be nonnegative")
    if neighbors_per_candidate and neighbor_radius <= 0:
        raise ValueError("neighbor_radius must be positive when generating neighbors")
    if max_power_w is not None and (not math.isfinite(max_power_w) or max_power_w <= 0):
        raise ValueError("max_power_w must be positive")
    if not math.isfinite(delay_weight) or not math.isfinite(power_weight):
        raise ValueError("objective weights must be finite")
    if not math.isfinite(neighbor_radius) or neighbor_radius < 0:
        raise ValueError("neighbor_radius must be finite and nonnegative")

    netlist_dir = _resolve_netlist_dir(netlist)
    template_path = netlist_dir / "netlist.spice"
    config_path = netlist_dir / "sizing.yaml"
    if not template_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"netlist directory must contain netlist.spice and sizing.yaml: {netlist_dir}"
        )
    sizing_config = load_sizing_config(config_path)
    template = template_path.read_text(encoding="utf-8")
    variable_names(sizing_config)
    run_id = run_id if run_id is not None else _default_run_id()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and hyphen")
    run_name = f"run_{run_id}"
    run_dir = (PROJECT_DIR / "run" / netlist_dir.name / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; choose another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "nrbo":
        import joblib

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = PROJECT_DIR / model_file
        if not model_file.is_file():
            raise FileNotFoundError(f"model file does not exist: {model_file}")
        model = joblib.load(model_file)
        candidates = optimize_candidates(
            model,
            sizing_config,
            objective=objective,
            maximize_objective=maximize_objective,
            delay_weight=delay_weight,
            speed_weight=speed_weight,
            power_weight=power_weight,
            top_k=samples,
            seed=seed,
            iterations=iterations,
            population_size=population_size,
            max_power_w=max_power_w,
            min_valid_probability=min_valid_probability,
        )
        payload = {
            "algorithm": "nrbo",
            "model": str(model_file.resolve()),
            "objective": objective,
            "maximize_objective": maximize_objective,
            "delay_weight": delay_weight,
            "speed_weight": speed_weight,
            "power_weight": power_weight,
            "max_power_w": max_power_w,
            "candidates": candidates,
            "note": "Surrogate predictions only; candidates require final SPICE verification.",
        }
        _atomic_json(run_dir / "candidates.json", payload)
        with (run_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "rank",
                    "objective",
                    "score",
                    "feasible",
                    "size_json",
                    "predictions_json",
                ),
            )
            writer.writeheader()
            for rank, candidate in enumerate(candidates, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "objective": candidate["objective"],
                        "score": candidate["score"],
                        "feasible": candidate["feasible"],
                        "size_json": json.dumps(candidate["size"], sort_keys=True),
                        "predictions_json": json.dumps(candidate["predictions"], sort_keys=True),
                    }
                )
        LOGGER.info("NRBO proposed %d candidates; results=%s", len(candidates), run_dir)
        return run_dir

    verification_predictions: dict[tuple[float, ...], Mapping] = {}
    if algorithm == "verify":
        candidate_file = Path(candidates_path)
        if not candidate_file.is_absolute():
            candidate_file = PROJECT_DIR / candidate_file
        candidate_payload = json.loads(candidate_file.read_text(encoding="utf-8"))
        raw_candidates = candidate_payload.get("candidates", [])
        if not raw_candidates:
            raise ValueError(f"candidate file contains no candidates: {candidate_file}")
        designs = []
        seen_sizes = set()
        neighbor_random = np.random.default_rng(seed)
        for candidate in raw_candidates:
            sizes = candidate.get("size")
            if not isinstance(sizes, Mapping):
                raise ValueError("each candidate must contain a 'size' mapping")
            vector = _normalized_vector(sizing_config, sizes)
            signature = tuple(round(float(value), 12) for value in vector)
            if isinstance(candidate.get("predictions"), Mapping):
                verification_predictions[signature] = candidate["predictions"]
            if signature not in seen_sizes:
                designs.append((vector, dict(sizes)))
                seen_sizes.add(signature)
            for _ in range(neighbors_per_candidate):
                neighbor = np.clip(
                    vector + neighbor_random.normal(0.0, neighbor_radius, len(vector)),
                    0.0,
                    1.0,
                )
                signature = tuple(round(float(value), 12) for value in neighbor)
                if signature not in seen_sizes:
                    designs.append((neighbor, _physical_sizes(sizing_config, neighbor)))
                    seen_sizes.add(signature)
    else:
        vectors = _sample_vectors(algorithm, samples, sizing_config, seed)
        designs = [(vector, _physical_sizes(sizing_config, vector)) for vector in vectors]
    sample_count = len(designs)

    LOGGER.info(
        "Starting %s: runner=%s, netlist=%s, samples=%d, algorithm=%s, seed=%d",
        run_name,
        runner_name,
        netlist_dir,
        sample_count,
        algorithm,
        seed,
    )
    LOGGER.debug("Sizing config: %s", config_path)
    LOGGER.debug("Output directory: %s", run_dir)
    LOGGER.log(TRACE_LEVEL, "Sizing configuration:\n%s", json.dumps(sizing_config, indent=2))

    metadata = {
        "name": run_name,
        "runner": runner_name,
        "netlist": str(netlist_dir),
        "algorithm": algorithm,
        "seed": seed,
        "requested_samples": sample_count,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "samples": [],
    }
    _atomic_json(run_dir / "results.json", metadata)
    (run_dir / "sizing.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    runner = (
        MockRunner(seed=seed, failure_rate=mock_failure_rate)
        if runner_name == "mock"
        else NgspiceRun(timeout_s=timeout_s)
    )

    for index, (vector, sizes) in enumerate(designs, start=1):
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
        except Exception as exc:
            LOGGER.error("Sample %d failed unexpectedly: %s", index, exc)
            LOGGER.debug("Unexpected sample exception", exc_info=True)
            sample_dir = run_dir / f"sample_{index:03d}"
            sample_dir.mkdir(exist_ok=True)
            record = {
                "name": f"{run_name}_{index:03d}",
                "status": "failed",
                "x_norm": [float(x) for x in vector],
                "size": sizes,
                "metrics": {},
                "files": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
            record.update(record["metrics"])
            _atomic_json(sample_dir / "result.json", record)
        signature = tuple(round(float(value), 12) for value in vector)
        predicted_metrics = verification_predictions.get(signature, {})
        for target, predicted_key, absolute_key, percent_key in (
            ("delay_s", "surrogate_delay_s", "delay_abs_error_s", "delay_ape_percent"),
            ("power_w", "surrogate_power_w", "power_abs_error_w", "power_ape_percent"),
        ):
            predicted_value = predicted_metrics.get(target)
            measured_value = record.get("metrics", {}).get(target)
            if predicted_value is not None:
                record["metrics"][predicted_key] = float(predicted_value)
            if predicted_value is not None and measured_value is not None:
                absolute_error = abs(float(predicted_value) - float(measured_value))
                record["metrics"][absolute_key] = absolute_error
                if float(measured_value) != 0:
                    record["metrics"][percent_key] = (
                        100.0 * absolute_error / abs(float(measured_value))
                    )
        record.update(record["metrics"])
        _atomic_json(run_dir / f"sample_{index:03d}" / "result.json", record)
        metadata["samples"].append(record)
        metadata["completed_samples"] = len(metadata["samples"])
        metadata["status"] = (
            "complete"
            if len(metadata["samples"]) == sample_count
            and all(sample["status"] == "evaluated" for sample in metadata["samples"])
            else "partial"
        )
        if index % RUN_CHECKPOINT_INTERVAL == 0:
            _write_csv(run_dir / "results.csv", metadata["samples"])
            _atomic_json(run_dir / "results.json", metadata)
        LOGGER.info(
            "Progress: %d/%d samples; latest status=%s",
            index,
            sample_count,
            record["status"],
        )
    _write_csv(run_dir / "results.csv", metadata["samples"])
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / "results.json", metadata)
    evaluated = sum(sample["status"] == "evaluated" for sample in metadata["samples"])
    LOGGER.info(
        "Finished %s: %d/%d fully evaluated; status=%s; results=%s",
        run_name,
        evaluated,
        sample_count,
        metadata["status"],
        run_dir,
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line options and run the selected sizing workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="ngspice", choices=("ngspice", "mock"))
    parser.add_argument(
        "--netlist",
        required=True,
        help="directory with netlist.spice and sizing.yaml",
    )
    parser.add_argument("--samples", "--combinations", type=int, default=10)
    parser.add_argument(
        "--algorithm",
        choices=("halton", "random", "lhs", "sobol", "nrbo", "verify"),
        default="halton",
    )
    parser.add_argument(
        "--model",
        help="trained joblib model artifact; required for --algorithm nrbo",
    )
    parser.add_argument("--candidates", help="candidate JSON required for --algorithm verify")
    parser.add_argument("--objective", help="optional single predicted metric to minimize")
    parser.add_argument("--maximize-objective", action="store_true")
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=0.0)
    parser.add_argument("--power-weight", type=float, default=0.0)
    parser.add_argument(
        "--max-power-w",
        type=float,
        help="optional predicted power constraint for NRBO mode",
    )
    parser.add_argument("--min-valid-probability", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=60, help="NRBO search iterations")
    parser.add_argument(
        "--population-size", type=int, default=24, help="NRBO search population size"
    )
    parser.add_argument("--neighbors-per-candidate", type=int, default=0)
    parser.add_argument("--neighbor-radius", type=float, default=0.02)
    parser.add_argument("--mock-failure-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-id",
        help="output under run/<netlist-name>/<run-id>; default run-id is a UTC timestamp",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log detail (-v info, -vv debug, -vvv trace)",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    start = time.perf_counter()
    run_dir = optimize(
        netlist=args.netlist,
        runner_name=args.runner,
        samples=args.samples,
        seed=args.seed,
        algorithm=args.algorithm,
        model_path=args.model,
        objective=args.objective,
        maximize_objective=args.maximize_objective,
        delay_weight=args.delay_weight,
        speed_weight=args.speed_weight,
        power_weight=args.power_weight,
        max_power_w=args.max_power_w,
        candidates_path=args.candidates,
        min_valid_probability=args.min_valid_probability,
        neighbors_per_candidate=args.neighbors_per_candidate,
        neighbor_radius=args.neighbor_radius,
        iterations=args.iterations,
        population_size=args.population_size,
        mock_failure_rate=args.mock_failure_rate,
        run_id=args.run_id,
        timeout_s=args.timeout_s,
    )
    print(f"Results: {run_dir}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
