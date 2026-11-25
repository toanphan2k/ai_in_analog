"""Extract differential gain and -3 dB bandwidth from ngspice AC wrdata."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json
from scripts.waveform_io import parse_ac


def _vector(vectors: dict, name: str) -> np.ndarray:
    for key, value in vectors.items():
        if key.lower() == name.lower():
            return np.asarray(value, dtype=complex)
    raise ValueError(f"AC output is missing vector {name!r}")


def extract_gain(
    waveform: str | Path,
    *,
    output_positive: str = "v(X)",
    output_negative: str = "v(Y)",
    input_positive: str = "v(inp)",
    input_negative: str = "v(inn)",
) -> dict:
    """Compute differential DC/peak gain and interpolated -3 dB bandwidth."""
    frequency, vectors = parse_ac(waveform)
    output_diff = _vector(vectors, output_positive) - _vector(vectors, output_negative)
    input_diff = _vector(vectors, input_positive) - _vector(vectors, input_negative)
    if np.any(np.abs(input_diff) == 0):
        raise ValueError("input differential AC stimulus contains zero magnitude")
    gain = np.abs(output_diff / input_diff)
    if gain.size == 0 or not np.isfinite(gain).all():
        raise ValueError("AC gain contains no finite samples")

    dc_gain = float(gain[0])
    target = dc_gain / math.sqrt(2)
    below = np.flatnonzero(gain <= target)
    bandwidth = float("nan")
    if dc_gain > 0 and len(below):
        index = int(below[0])
        if index == 0:
            bandwidth = float(frequency[0])
        else:
            f0, f1 = float(frequency[index - 1]), float(frequency[index])
            g0, g1 = float(gain[index - 1]), float(gain[index])
            if f0 > 0 and f1 > 0 and g0 != g1:
                fraction = (target - g0) / (g1 - g0)
                bandwidth = float(
                    math.exp(math.log(f0) + fraction * (math.log(f1) - math.log(f0)))
                )
    return {
        "dc_gain_v_per_v": dc_gain,
        "peak_gain_v_per_v": float(gain.max()),
        "peak_gain_frequency_hz": float(frequency[int(np.argmax(gain))]),
        "bandwidth_3db_hz": bandwidth,
        "ac_freq_points": int(len(frequency)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse AC waveform options and print extracted gain metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = extract_gain(args.waveform)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
