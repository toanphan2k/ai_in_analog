"""Extract worst-case transient decision delay from an ngspice wrdata file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json
from scripts.waveform_io import load_transient


def _read_waveform(path: str | Path):
    vectors = load_transient(
        path,
        default_names=("v(x)", "v(y)", "v(ck)", "i(vdd)"),
    )
    time = vectors.pop("time", vectors.pop("t", None))
    if time is None:
        raise ValueError("transient waveform must have a time column")
    return vectors, time


def _rising_crossings(time, signal, threshold: float):
    indices = np.flatnonzero((signal[:-1] < threshold) & (signal[1:] >= threshold))
    crossings = []
    for index in indices:
        fraction = (threshold - signal[index]) / (signal[index + 1] - signal[index])
        crossings.append(time[index] + fraction * (time[index + 1] - time[index]))
    return crossings


def _falling_crossing(t, values, threshold: float, start: float, stop: float):
    indices = np.flatnonzero(
        (values[:-1] >= threshold) & (values[1:] < threshold)
    )
    for i in indices:
        fraction = (threshold - values[i]) / (values[i + 1] - values[i])
        crossing = float(t[i] + fraction * (t[i + 1] - t[i]))
        if start <= crossing <= stop:
            return crossing
    return None


def extract_delay(
    waveform: str | Path,
    vdd: float,
    period_s: float,
    *,
    output_positive: str = "v(x)",
    output_negative: str = "v(y)",
    clock_vector: str = "v(ck)",
    power_vector: str = "p_inst",
    power_from_s: float | None = None,
    power_to_s: float | None = None,
) -> dict:
    if vdd <= 0 or period_s <= 0:
        raise ValueError("vdd and period_s must be positive")
    vectors, time = _read_waveform(waveform)
    output_positive = output_positive.lower()
    output_negative = output_negative.lower()
    clock_vector = clock_vector.lower()
    power_vector = power_vector.lower()
    required = (output_positive, output_negative, clock_vector)
    missing = [name for name in required if name not in vectors]
    if missing:
        raise ValueError(f"transient output is missing vector(s): {', '.join(missing)}")
    x, y, clock = (vectors[name] for name in required)
    edges = _rising_crossings(time, clock, vdd / 2)
    if not edges:
        raise ValueError(f"clock has no rising crossing at {vdd / 2:.3f} V")

    decision_output = np.minimum(x, y)
    delays = []
    window_minima = []
    for edge in edges:
        window_stop = float(edge + period_s / 2)
        in_window = (time >= edge) & (time <= window_stop)
        if np.any(in_window):
            window_minima.append(float(np.min(decision_output[in_window])))
        crossing = _falling_crossing(
            time,
            decision_output,
            vdd / 2,
            float(edge),
            window_stop,
        )
        if crossing is not None:
            delays.append(crossing - float(edge))
    if not delays:
        lowest = min(window_minima, default=float(np.min(decision_output)))
        raise ValueError(
            f"no output decision crossing below {vdd / 2:.3f} V within half a clock "
            f"period after {len(edges)} rising edge(s); lowest observed min(output) "
            f"in those windows was {lowest:.3f} V"
        )
    result = {
        "delay_s": float(max(delays)),
        "delay_mean_s": float(np.mean(delays)),
        "cycles_measured": len(delays),
        "output_swing_v": float(np.max(np.abs(x - y))),
    }
    if (power_from_s is None) != (power_to_s is None):
        raise ValueError("power_from_s and power_to_s must be provided together")
    if power_from_s is not None and power_to_s is not None:
        if power_to_s <= power_from_s:
            raise ValueError("power_to_s must be greater than power_from_s")
        power = vectors.get(power_vector)
        if power is None:
            raise ValueError("transient output does not contain a p_inst vector")
        mask = (time > power_from_s) & (time < power_to_s)
        t_segment = np.concatenate(([power_from_s], time[mask], [power_to_s]))
        p_segment = np.interp(t_segment, time, power)
        result["power_w"] = float(np.trapezoid(p_segment, t_segment) / (power_to_s - power_from_s))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--vdd", type=float, required=True)
    parser.add_argument("--period-s", type=float, required=True)
    parser.add_argument("--power-from-s", type=float)
    parser.add_argument("--power-to-s", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = extract_delay(
        args.waveform,
        args.vdd,
        args.period_s,
        power_from_s=args.power_from_s,
        power_to_s=args.power_to_s,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
