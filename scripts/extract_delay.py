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
        crossings.append(float(time[index] + fraction * (time[index + 1] - time[index])))
    return crossings


def _falling_crossings(time, signal, threshold: float):
    indices = np.flatnonzero((signal[:-1] >= threshold) & (signal[1:] < threshold))
    crossings = []
    for index in indices:
        fraction = (threshold - signal[index]) / (signal[index + 1] - signal[index])
        crossings.append(float(time[index] + fraction * (time[index + 1] - time[index])))
    return crossings


def _rising_crossing(t, values, threshold: float, start: float, stop: float):
    indices = np.flatnonzero(
        (values[:-1] < threshold) & (values[1:] >= threshold)
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
    input_positive: str | None = "v(inp)",
    input_negative: str | None = "v(inn)",
    power_vector: str = "p_inst",
    power_from_s: float | None = None,
    power_to_s: float | None = None,
    reset_threshold_ratio: float = 0.9,
    inverting: bool = True,
    check_polarity: bool = True,
) -> dict:
    """Extract worst-case evaluation delay, reset time, and optional average power.

    Enforces full-cycle StrongARM operation across every clock cycle:
    - outputs must be precharged above ``reset_threshold_ratio * vdd`` before each rising edge;
    - differential output ``|X - Y|`` must cross ``vdd / 2`` within half a clock period on every
      cycle (no stuck/partial-cycle latches);
    - when input vectors are present and ``check_polarity`` is True, final output polarity at the
      end of each evaluation phase must match the input differential sign;
    - during every complete reset half-cycle, ``min(X, Y)`` must recover above
      ``reset_threshold_ratio * vdd`` within half a clock period.
    """
    if vdd <= 0 or period_s <= 0:
        raise ValueError("vdd and period_s must be positive")
    if not 0.5 < reset_threshold_ratio < 1.0:
        raise ValueError("reset_threshold_ratio must be in (0.5, 1.0)")
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
    decision_threshold = vdd / 2.0
    reset_threshold = reset_threshold_ratio * vdd

    edges = _rising_crossings(time, clock, decision_threshold)
    if not edges:
        raise ValueError(f"clock has no rising crossing at {decision_threshold:.3f} V")
    falling_edges = _falling_crossings(time, clock, decision_threshold)

    inp_vec = None
    inn_vec = None
    if check_polarity and input_positive and input_negative:
        inp_key = input_positive.lower()
        inn_key = input_negative.lower()
        if inp_key in vectors and inn_key in vectors:
            inp_vec = vectors[inp_key]
            inn_vec = vectors[inn_key]

    vdiff_signed = x - y
    vdiff_abs = np.abs(vdiff_signed)
    min_output = np.minimum(x, y)

    delays: list[float] = []
    cycle_swings: list[float] = []
    for cycle_idx, edge in enumerate(edges):
        window_stop = min(float(edge + period_s / 2.0), float(time[-1]))
        x_edge = float(np.interp(edge, time, x))
        y_edge = float(np.interp(edge, time, y))
        if min(x_edge, y_edge) < reset_threshold or abs(x_edge - y_edge) >= decision_threshold:
            raise ValueError(
                f"comparator outputs not reset before cycle {cycle_idx} at t={edge:.3e} s "
                f"(X={x_edge:.3f} V, Y={y_edge:.3f} V; required min >= {reset_threshold:.3f} V "
                f"and |X-Y| < {decision_threshold:.3f} V)"
            )

        crossing = _rising_crossing(
            time,
            vdiff_abs,
            decision_threshold,
            edge,
            window_stop,
        )
        if crossing is None:
            in_window = (time >= edge) & (time <= window_stop)
            peak_diff = float(np.max(vdiff_abs[in_window])) if np.any(in_window) else 0.0
            raise ValueError(
                f"no differential decision crossing |X-Y| >= {decision_threshold:.3f} V in cycle "
                f"{cycle_idx} within half a clock period after t={edge:.3e} s "
                f"(max |X-Y| was {peak_diff:.3f} V)"
            )
        delays.append(crossing - edge)

        in_window = (time >= edge) & (time <= window_stop)
        if np.any(in_window):
            cycle_swings.append(float(np.max(vdiff_abs[in_window])))

        if inp_vec is not None and inn_vec is not None:
            vin_diff = float(np.interp(edge, time, inp_vec) - np.interp(edge, time, inn_vec))
            if abs(vin_diff) > 1e-6:
                vout_end = float(np.interp(window_stop, time, vdiff_signed))
                expected_sign = -1.0 if ((vin_diff > 0.0) == inverting) else 1.0
                if expected_sign * vout_end < decision_threshold:
                    raise ValueError(
                        f"wrong decision polarity in cycle {cycle_idx} at t={edge:.3e} s "
                        f"(Vin_diff={vin_diff:+.3e} V, end-of-eval X-Y={vout_end:+.3f} V)"
                    )

    reset_times: list[float] = []
    for cycle_idx, fall_edge in enumerate(falling_edges):
        reset_stop = float(fall_edge + period_s / 2.0)
        if reset_stop > float(time[-1]):
            continue
        min_at_fall = float(np.interp(fall_edge, time, min_output))
        in_reset = (time >= fall_edge) & (time <= reset_stop)
        if min_at_fall >= reset_threshold and np.any(in_reset) and float(np.min(min_output[in_reset])) >= reset_threshold:
            reset_times.append(0.0)
            continue
        rst_crossing = _rising_crossing(
            time,
            min_output,
            reset_threshold,
            fall_edge,
            reset_stop,
        )
        if rst_crossing is None:
            peak_rst = float(np.max(min_output[in_reset])) if np.any(in_reset) else min_at_fall
            raise ValueError(
                f"comparator failed to reset min(X,Y) >= {reset_threshold:.3f} V after cycle "
                f"{cycle_idx} falling edge at t={fall_edge:.3e} s "
                f"(highest observed min(X,Y) was {peak_rst:.3f} V)"
            )
        reset_times.append(rst_crossing - fall_edge)

    delay_max = float(max(delays))
    result = {
        "delay_s": delay_max,
        "delay_mean_s": float(np.mean(delays)),
        "cycles_measured": len(delays),
        "output_swing_v": float(min(cycle_swings)) if cycle_swings else float(np.max(vdiff_abs)),
    }
    if reset_times:
        reset_max = float(max(reset_times))
        result["reset_time_s"] = reset_max
        result["reset_time_mean_s"] = float(np.mean(reset_times))
        half_period_min_s = max(delay_max, reset_max)
    else:
        half_period_min_s = delay_max
    if half_period_min_s > 0.0:
        result["speed_hz"] = 1.0 / (2.0 * half_period_min_s)

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
    """Parse waveform measurement options and print extracted delay metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--vdd", type=float, required=True)
    parser.add_argument("--period-s", type=float, required=True)
    parser.add_argument("--power-from-s", type=float)
    parser.add_argument("--power-to-s", type=float)
    parser.add_argument("--reset-threshold-ratio", type=float, default=0.9)
    parser.add_argument("--non-inverting", action="store_true")
    parser.add_argument("--no-polarity-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = extract_delay(
        args.waveform,
        args.vdd,
        args.period_s,
        power_from_s=args.power_from_s,
        power_to_s=args.power_to_s,
        reset_threshold_ratio=args.reset_threshold_ratio,
        inverting=not args.non_inverting,
        check_polarity=not args.no_polarity_check,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
