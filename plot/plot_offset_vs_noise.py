#!/usr/bin/env python3
"""Plot 4: Offset vs. Noise."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot._common import (
    VOLTAGE_UNITS,
    build_arg_parser,
    configure_matplotlib,
    default_output_path,
    load_results_csv,
    pareto_front_indices,
    save_or_show,
    select_unit,
)

DEFAULT_STEM = "plot4_offset_vs_noise"


def plot_offset_vs_noise(
    csv_paths: Sequence[str | Path],
    *,
    output: Path | None = None,
    title: str | None = None,
    dpi: int = 180,
    show: bool = False,
    log_x: bool = False,
    log_y: bool = False,
) -> Path:
    """Render |Input-Referred Offset| vs. Input-Referred Noise with Pareto front."""
    runs = load_results_csv(csv_paths)
    plt = configure_matplotlib(show)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_noise = np.concatenate(
        [r.noise_v[[s.lower() != "failed" for s in r.statuses]] for r in runs]
    )
    all_offset = np.concatenate(
        [np.abs(r.offset_v[[s.lower() != "failed" for s in r.statuses]]) for r in runs]
    )
    noise_scale, noise_unit = select_unit(all_noise, VOLTAGE_UNITS)
    offset_scale, offset_unit = select_unit(all_offset, VOLTAGE_UNITS)

    single_run = len(runs) == 1
    plotted_any = False
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        not_failed = np.array([s.lower() != "failed" for s in run.statuses], dtype=bool)
        offset_abs = np.abs(run.offset_v)
        valid = not_failed & np.isfinite(run.noise_v) & np.isfinite(offset_abs)
        if not np.any(valid):
            continue
        plotted_any = True
        x_all = run.noise_v * noise_scale
        y_all = offset_abs * offset_scale

        infeasible_mask = valid & ~run.feasible
        if np.any(infeasible_mask):
            ax.scatter(
                x_all[infeasible_mask],
                y_all[infeasible_mask],
                marker="x",
                color="gray" if single_run else color,
                alpha=0.55,
                s=34,
                label=f"{run.label} infeasible" if not single_run else "Infeasible",
            )

        feas_mask = valid & run.feasible
        if np.any(feas_mask):
            x_f = x_all[feas_mask]
            y_f = y_all[feas_mask]
            if single_run:
                sc = ax.scatter(
                    x_f,
                    y_f,
                    c=run.eval_numbers[feas_mask],
                    cmap="viridis",
                    s=46,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.35,
                    label=f"{run.label} feasible (n={x_f.size})",
                )
                fig.colorbar(sc, ax=ax, label="Evaluation Number")
            else:
                ax.scatter(
                    x_f,
                    y_f,
                    color=color,
                    s=42,
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=0.3,
                    label=f"{run.label} feasible (n={x_f.size})",
                )

            front = pareto_front_indices(x_f, y_f)
            if front.size >= 1:
                front_color = "crimson" if single_run else color
                ax.plot(
                    x_f[front],
                    y_f[front],
                    color=front_color,
                    linewidth=2.0,
                    marker="o",
                    markersize=5.5,
                    label=f"{run.label} Pareto front ({front.size})" if not single_run else f"Pareto front ({front.size})",
                )

    if not plotted_any:
        raise ValueError("No valid (input_referred_noise_v, input_referred_offset_v) points found.")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(f"Input-Referred Noise ({noise_unit})")
    ax.set_ylabel(f"|Input-Referred Offset| ({offset_unit})")
    ax.set_title(title or "Offset vs. Noise")
    ax.legend(loc="best")

    out_path = output or default_output_path(runs, DEFAULT_STEM)
    return save_or_show(fig, out_path, dpi=dpi, show=show)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser(__doc__ or "Plot Offset vs. Noise.")
    parser.add_argument("--log-x", action="store_true", help="use log scale for Noise axis")
    parser.add_argument("--log-y", action="store_true", help="use log scale for Offset axis")
    args = parser.parse_args(argv)
    plot_offset_vs_noise(
        args.output_dir,
        output=args.save_path,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
        log_x=args.log_x,
        log_y=args.log_y,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
