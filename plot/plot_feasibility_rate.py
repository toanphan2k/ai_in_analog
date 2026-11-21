#!/usr/bin/env python3
"""Plot 2: Feasibility Rate vs. Evaluation Number."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot._common import (
    build_arg_parser,
    configure_matplotlib,
    default_output_path,
    load_results_csv,
    save_or_show,
)

DEFAULT_STEM = "plot2_feasibility_rate_vs_eval"


def _rolling_rate(flags: np.ndarray, window: int) -> np.ndarray:
    n = flags.size
    window = max(1, min(window, n))
    cumsum = np.concatenate(([0.0], np.cumsum(flags.astype(float))))
    rates = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i + 1 - window)
        rates[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return rates * 100.0


def plot_feasibility_rate(
    csv_paths: Sequence[str | Path],
    *,
    output: Path | None = None,
    title: str | None = None,
    dpi: int = 180,
    show: bool = False,
    window: int = 10,
) -> Path:
    """Render Feasibility Rate vs. Evaluation Number and return saved image path."""
    runs = load_results_csv(csv_paths)
    plt = configure_matplotlib(show)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        x = run.eval_numbers
        cumulative_pct = np.cumsum(run.feasible.astype(float)) / x * 100.0
        eff_window = max(1, min(window, x.size))
        rolling_pct = _rolling_rate(run.feasible, eff_window)

        ax.plot(
            x,
            cumulative_pct,
            color=color,
            linewidth=2.2,
            label=f"{run.label} cumulative ({cumulative_pct[-1]:.1f}%)",
        )
        if x.size >= 4:
            ax.plot(
                x,
                rolling_pct,
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
                label=f"{run.label} rolling (w={eff_window})",
            )

    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("Feasibility Rate (%)")
    ax.set_ylim(-2.0, 105.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title or "Feasibility Rate vs. Evaluation Number")
    ax.legend(loc="best")

    out_path = output or default_output_path(runs, DEFAULT_STEM)
    return save_or_show(fig, out_path, dpi=dpi, show=show)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser(__doc__ or "Plot Feasibility Rate vs. Evaluation Number.")
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="rolling window size in evaluations (default: 10)",
    )
    args = parser.parse_args(argv)
    plot_feasibility_rate(
        args.csv,
        output=args.output,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
        window=args.window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
