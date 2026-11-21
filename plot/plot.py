#!/usr/bin/env python3
"""Generate all 5 standard plots from one or more results.csv files."""
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True, help="path(s) to results.csv file(s)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "directory for output PNGs "
            "(default: <ws>/plot_<algorithm>_seed_<seed>_iteration_<iteration>/)"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180, help="output image DPI (default: 180)")
    parser.add_argument("--show", action="store_true", help="display plots interactively")
    args = parser.parse_args(argv)

    def _out(stem: str) -> Path | None:
        return (args.output_dir / f"{stem}.png") if args.output_dir is not None else None

    plot_objective_score(args.csv, output=_out(OBJ_STEM), dpi=args.dpi, show=args.show)
    plot_feasibility_rate(args.csv, output=_out(FEAS_STEM), dpi=args.dpi, show=args.show)
    plot_power_vs_delay(args.csv, output=_out(PWR_STEM), dpi=args.dpi, show=args.show)
    plot_offset_vs_noise(args.csv, output=_out(OFF_STEM), dpi=args.dpi, show=args.show)
    plot_kickback_vs_noise(args.csv, output=_out(KB_STEM), dpi=args.dpi, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
