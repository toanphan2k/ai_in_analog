#!/usr/bin/env python3
"""Generate all standard plots from one or more results.csv files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.plot_feasibility_rate import DEFAULT_STEM as FEAS_STEM, plot_feasibility_rate
from plot.plot_kickback_vs_noise import DEFAULT_STEM as KB_STEM, plot_kickback_vs_noise
from plot.plot_objective_score import DEFAULT_STEM as OBJ_STEM, plot_objective_score
from plot.plot_offset_vs_noise import DEFAULT_STEM as OFF_STEM, plot_offset_vs_noise
from plot.plot_power_vs_delay import DEFAULT_STEM as PWR_STEM, plot_power_vs_delay
from plot.plot_transient_waveforms import plot_transient_waveforms


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        "--csv",
        dest="output_dir",
        nargs="+",
        required=True,
        help="path(s) to run output directory(ies) (e.g. run/strongarm_5ghz/<run_id>) or results.csv file(s)",
    )
    parser.add_argument(
        "-o",
        "--plot-dir",
        dest="plot_dir",
        type=Path,
        default=None,
        help=(
            "directory for output PNGs "
            "(default: <ws>/plot_<algorithm>_seed_<seed>_iteration_<iteration>_<run_id>/)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of best candidates per run for transient waveform plots (default: 5)",
    )
    parser.add_argument(
        "--skip-transient",
        action="store_true",
        help="skip generating top-K transient waveform plots",
    )
    parser.add_argument("--dpi", type=int, default=180, help="output image DPI (default: 180)")
    parser.add_argument("--show", action="store_true", help="display plots interactively")
    args = parser.parse_args(argv)

    def _out(stem: str) -> Path | None:
        return (args.plot_dir / f"{stem}.png") if args.plot_dir is not None else None

    plot_objective_score(args.output_dir, output=_out(OBJ_STEM), dpi=args.dpi, show=args.show)
    plot_feasibility_rate(args.output_dir, output=_out(FEAS_STEM), dpi=args.dpi, show=args.show)
    plot_power_vs_delay(args.output_dir, output=_out(PWR_STEM), dpi=args.dpi, show=args.show)
    plot_offset_vs_noise(args.output_dir, output=_out(OFF_STEM), dpi=args.dpi, show=args.show)
    plot_kickback_vs_noise(args.output_dir, output=_out(KB_STEM), dpi=args.dpi, show=args.show)
    if not args.skip_transient and args.top_k > 0:
        saved = plot_transient_waveforms(
            args.output_dir,
            output_dir=args.plot_dir,
            top_k=args.top_k,
            dpi=args.dpi,
            show=args.show,
            allow_missing=True,
        )
        if not saved:
            print("Skipped transient waveform plots: no transient.txt waveforms found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
