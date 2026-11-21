#!/usr/bin/env python3
"""Plot 1: Objective Score vs. Evaluation Number."""
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

DEFAULT_STEM = "plot1_objective_score_vs_eval"
SENTINEL_SCORE = 999.0


def plot_objective_score(
    csv_paths: Sequence[str | Path],
    *,
    output: Path | None = None,
    title: str | None = None,
    dpi: int = 180,
    show: bool = False,
    include_failed: bool = False,
) -> Path:
    """Render Objective Score vs. Evaluation Number and return saved image path."""
    runs = load_results_csv(csv_paths)
    plt = configure_matplotlib(show)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted_any = False
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        finite = np.isfinite(run.objective_score)
        if include_failed:
            mask = finite
        else:
            mask = finite & run.feasible & (run.objective_score < SENTINEL_SCORE)
            if not np.any(mask):
                mask = finite & (run.objective_score < SENTINEL_SCORE)

        if not np.any(mask):
            continue
        plotted_any = True
        x = run.eval_numbers[mask]
        y = run.objective_score[mask]
        running_best = np.minimum.accumulate(y)
        best_idx = int(np.argmin(y))

        ax.scatter(
            x,
            y,
            color=color,
            alpha=0.35,
            s=26,
            edgecolors="none",
            label=f"{run.label} (evaluations)",
        )
        ax.plot(
            x,
            running_best,
            color=color,
            linewidth=2.2,
            label=f"{run.label} best-so-far (min={y[best_idx]:.4g} @ eval {x[best_idx]})",
        )
        ax.scatter(
            [x[best_idx]],
            [y[best_idx]],
            color=color,
            marker="*",
            s=130,
            zorder=5,
            edgecolors="black",
            linewidths=0.6,
        )

    if not plotted_any:
        raise ValueError("No valid objective scores found to plot.")

    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("RL Objective Score (lower is better)")
    ax.set_title(title or "Objective Score vs. Evaluation Number")
    ax.legend(loc="best")

    out_path = output or default_output_path(runs, DEFAULT_STEM)
    return save_or_show(fig, out_path, dpi=dpi, show=show)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser(__doc__ or "Plot Objective Score vs. Evaluation Number.")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="include failed/infeasible sentinel scores (e.g., 1000.0)",
    )
    args = parser.parse_args(argv)
    plot_objective_score(
        args.csv,
        output=args.output,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
        include_failed=args.include_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
