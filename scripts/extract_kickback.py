"""Extract peak and RMS differential kickback noise from ngspice transient wrdata."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json
from scripts.waveform_io import load_transient


def extract_kickback(
    waveform: str | Path,
    *,
    from_s: float = 1.0e-9,
    to_s: float = 3.0e-9,
    differential_vector: str = "vkb_diff",
) -> dict[str, float]:
    """Extract peak and RMS differential kickback voltage over a steady-state window."""
    if to_s <= from_s:
        raise ValueError("kickback to_s must be greater than from_s")
    vectors = load_transient(waveform)
    time = vectors.pop("time", vectors.pop("t", None))
    if time is None:
        raise ValueError("kickback waveform must contain a time column")

    diff_key = differential_vector.lower()
    if diff_key in vectors:
        vkb_diff = vectors[diff_key]
    elif all(name in vectors for name in ("v(inp)", "v(inn)", "v(src_p)", "v(src_n)")):
        vkb_diff = (vectors["v(inp)"] - vectors["v(inn)"]) - (
            vectors["v(src_p)"] - vectors["v(src_n)"]
        )
    else:
        raise ValueError(f"kickback output is missing vector {differential_vector!r}")

    mask = (time >= from_s) & (time <= to_s)
    if not np.any(mask):
        raise ValueError(f"kickback waveform has no samples inside [{from_s}, {to_s}] s")

    t_seg = np.concatenate(([from_s], time[(time > from_s) & (time < to_s)], [to_s]))
    v_seg = np.interp(t_seg, time, vkb_diff)
    peak_v = float(np.max(np.abs(v_seg)))
    rms_v = float(math.sqrt(max(0.0, float(np.trapezoid(v_seg**2, t_seg)) / (to_s - from_s))))
    return {
        "kickback_noise_v": peak_v,
        "kickback_rms_v": rms_v,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse kickback waveform options and print extracted kickback metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--from-s", type=float, default=1.0e-9)
    parser.add_argument("--to-s", type=float, default=3.0e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = extract_kickback(args.waveform, from_s=args.from_s, to_s=args.to_s)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
