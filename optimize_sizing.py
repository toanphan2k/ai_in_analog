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

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from runner.ngspice_runner import NgspiceRun  # noqa: E402
from scripts._common import load_config  # noqa: E402
from scripts.deck_generation import (  # noqa: E402
    render_gain_deck,
    render_transient_deck,
)
from scripts.extract_delay import extract_delay  # noqa: E402
from scripts.extract_gain import extract_gain  # noqa: E402
from scripts.netlist_tools import load_sizing_config  # noqa: E402
from scripts.waveform_io import load_transient  # noqa: E402
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
    "samples",
    "error",
)


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


def _default_run_id(netlist_dir: Path) -> str:
    time_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{netlist_dir.name}_{time_id}"


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
    runner: NgspiceRun,
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
    run_id: str | None = None,
    timeout_s: float = 120.0,
) -> Path:
    if runner_name != "ngspice":
        raise ValueError(f"unsupported runner {runner_name!r}; available: ngspice")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if algorithm not in {"halton", "random"}:
        raise ValueError("algorithm must be 'halton' or 'random'")

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
    run_id = run_id if run_id is not None else _default_run_id(netlist_dir)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and hyphen")
    run_name = f"run_{run_id}"
    run_dir = (PROJECT_DIR / "run" / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; choose another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting %s: runner=%s, netlist=%s, samples=%d, algorithm=%s, seed=%d",
        run_name,
        runner_name,
        netlist_dir,
        samples,
        algorithm,
        seed,
    )
    LOGGER.debug("Sizing config: %s", config_path)
    LOGGER.debug("Output directory: %s", run_dir)
    LOGGER.log(TRACE_LEVEL, "Sizing configuration:\n%s", json.dumps(sizing_config, indent=2))

    if algorithm == "halton":
        vectors = candidate_pool(samples, sizing_config, seed=seed)
    else:
        vectors = np.random.default_rng(seed).random(
            (samples, len(variable_names(sizing_config)))
        )

    metadata = {
        "name": run_name,
        "runner": runner_name,
        "netlist": str(netlist_dir),
        "algorithm": algorithm,
        "seed": seed,
        "requested_samples": samples,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "samples": [],
    }
    _atomic_json(run_dir / "results.json", metadata)
    (run_dir / "sizing.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    runner = NgspiceRun(timeout_s=timeout_s)

    for index, vector in enumerate(vectors, start=1):
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
        metadata["samples"].append(record)
        _write_csv(run_dir / "results.csv", metadata["samples"])
        metadata["completed_samples"] = len(metadata["samples"])
        metadata["status"] = (
            "complete"
            if len(metadata["samples"]) == samples
            and all(sample["status"] == "evaluated" for sample in metadata["samples"])
            else "partial"
        )
        LOGGER.info("Progress: %d/%d samples; latest status=%s", index, samples, record["status"])
        _atomic_json(run_dir / "results.json", metadata)
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / "results.json", metadata)
    evaluated = sum(sample["status"] == "evaluated" for sample in metadata["samples"])
    LOGGER.info(
        "Finished %s: %d/%d fully evaluated; status=%s; results=%s",
        run_name,
        evaluated,
        samples,
        metadata["status"],
        run_dir,
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="ngspice", choices=("ngspice",))
    parser.add_argument("--netlist", required=True, help="directory with netlist.spice and sizing.yaml")
    parser.add_argument("--samples", "--combinations", type=int, default=10)
    parser.add_argument("--algorithm", choices=("halton", "random"), default="halton")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-id",
        help="output under run/<run-id>; default: <netlist-name>_<UTC-time-id>",
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
        run_id=args.run_id,
        timeout_s=args.timeout_s,
    )
    print(f"Results: {run_dir}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
