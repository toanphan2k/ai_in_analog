"""Extract integrated input-referred and output-referred RMS noise from ngspice output."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts._common import write_json
from scripts.waveform_io import load_noise_spectrum, load_scalar_metrics


def extract_noise(
    waveform: str | Path,
    *,
    dc_gain_v_per_v: float | None = None,
    metrics_path: str | Path | None = None,
    output_spectrum: str = "onoise_spectrum",
) -> dict[str, float]:
    """Integrate output noise PSD over frequency and refer it to the comparator input."""
    frequency, spectra = load_noise_spectrum(waveform)
    spec_key = output_spectrum.lower()
    if spec_key not in spectra:
        raise ValueError(f"noise output is missing spectrum {output_spectrum!r}")
    onoise = np.asarray(spectra[spec_key], dtype=float)
    if frequency.size < 2 or not (np.isfinite(frequency).all() and np.isfinite(onoise).all()):
        raise ValueError("noise spectrum contains insufficient or non-finite samples")
    onoise_rms = float(math.sqrt(max(0.0, float(np.trapezoid(onoise**2, frequency)))))

    gain = dc_gain_v_per_v
    scalars = load_scalar_metrics(metrics_path) if metrics_path and Path(metrics_path).is_file() else {}
    if gain is None or not math.isfinite(float(gain)) or float(gain) <= 0.0:
        gain = scalars.get("dc_gain_1khz")
    if gain is not None and math.isfinite(float(gain)) and float(gain) > 0.0:
        inoise_rms = onoise_rms / float(gain)
    elif "input_referred_noise_v" in scalars and scalars["input_referred_noise_v"] > 0.0:
        inoise_rms = float(scalars["input_referred_noise_v"])
    else:
        raise ValueError("positive DC gain is required to compute input-referred noise")

    return {
        "input_referred_noise_v": float(inoise_rms),
        "output_referred_noise_v": float(onoise_rms),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse noise waveform options and print extracted noise metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--dc-gain", type=float)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = extract_noise(
        args.waveform,
        dc_gain_v_per_v=args.dc_gain,
        metrics_path=args.metrics,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
