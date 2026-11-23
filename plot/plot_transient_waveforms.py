#!/usr/bin/env python3
"""Plot 6: Transient Waveforms for Top-K Best Candidates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot._common import (
    configure_matplotlib,
    default_output_dir,
    load_results_csv,
)
from scripts.waveform_io import load_transient

DEFAULT_STEM = "plot6_transient"


def _rising_crossings(time: np.ndarray, sig: np.ndarray, threshold: float) -> list[float]:
    indices = np.flatnonzero((sig[:-1] < threshold) & (sig[1:] >= threshold))
    return [
        float(time[i] + (threshold - sig[i]) / (sig[i + 1] - sig[i]) * (time[i + 1] - time[i]))
        for i in indices
    ]


def _falling_crossings(time: np.ndarray, sig: np.ndarray, threshold: float) -> list[float]:
    indices = np.flatnonzero((sig[:-1] >= threshold) & (sig[1:] < threshold))
    return [
        float(time[i] + (threshold - sig[i]) / (sig[i + 1] - sig[i]) * (time[i + 1] - time[i]))
        for i in indices
    ]


def _first_rising_in_window(
    time: np.ndarray, sig: np.ndarray, threshold: float, start: float, stop: float
) -> float | None:
    indices = np.flatnonzero((sig[:-1] < threshold) & (sig[1:] >= threshold))
    for i in indices:
        t_cross = float(time[i] + (threshold - sig[i]) / (sig[i + 1] - sig[i]) * (time[i + 1] - time[i]))
        if start <= t_cross <= stop:
            return t_cross
    return None


def _load_transient_config(run_dir: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "waveform": "transient.txt",
        "vdd_v": 1.8,
        "period_s": 5.0e-10,
        "reset_threshold_ratio": 0.9,
        "output_positive": "v(x)",
        "output_negative": "v(y)",
        "clock_vector": "v(ck)",
        "input_positive": "v(inp)",
        "input_negative": "v(inn)",
        "power_vector": "p_inst",
    }
    sizing_path = run_dir / "sizing.yaml"
    if not sizing_path.is_file():
        return defaults
    try:
        cfg = yaml.safe_load(sizing_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return defaults
    transient_cfg = (cfg.get("analyses") or {}).get("transient") or {}
    if isinstance(transient_cfg, dict):
        for key in defaults:
            if key in transient_cfg and transient_cfg[key] is not None:
                val = transient_cfg[key]
                defaults[key] = val.lower() if isinstance(val, str) and key != "waveform" else val
    return defaults


def _resolve_waveform_path(run_dir: Path, sample_id: str, waveform_name: str) -> tuple[Path | None, str]:
    suffix = sample_id.rsplit("_", 1)[-1]
    folder_candidates = [f"sample_{suffix}", sample_id]
    for folder in folder_candidates:
        wf_path = run_dir / "data" / folder / waveform_name
        if wf_path.is_file():
            return wf_path, folder
    return None, folder_candidates[0]


def _load_top_candidates(csv_path: Path, top_k: int) -> list[dict[str, Any]]:
    run_dir = csv_path.parent
    candidates: list[dict[str, Any]] = []
    seen_samples: set[str] = set()

    cand_json = run_dir / "candidates.json"
    if cand_json.is_file():
        try:
            payload = json.loads(cand_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        for item in payload.get("candidates", []) if isinstance(payload, dict) else []:
            sample_id = str(item.get("sample", "")).strip()
            if not sample_id or sample_id in seen_samples:
                continue
            seen_samples.add(sample_id)
            candidates.append(item)
            if len(candidates) >= top_k:
                return candidates

    # Fall back to ranking rows from results.csv if candidates.json is missing or has fewer than top_k
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def _f(row: dict[str, str], key: str) -> float:
        raw = (row.get(key) or "").strip()
        if not raw:
            return math.nan
        try:
            val = float(raw)
        except ValueError:
            return math.nan
        return val if math.isfinite(val) else math.nan

    evaluated_rows = [
        r for r in rows if (r.get("status") or "").strip().lower() == "evaluated"
    ]
    evaluated_rows.sort(
        key=lambda r: (
            0 if (r.get("rl_feasible") or "").strip().lower() in {"true", "1", "yes"} else 1,
            _f(r, "rl_objective_score") if math.isfinite(_f(r, "rl_objective_score")) else math.inf,
        )
    )
    for row in evaluated_rows:
        sample_id = (row.get("sample") or "").strip()
        if not sample_id or sample_id in seen_samples:
            continue
        seen_samples.add(sample_id)
        score = _f(row, "rl_objective_score")
        metrics = {
            "delay_s": _f(row, "delay_s"),
            "delay_mean_s": _f(row, "delay_mean_s"),
            "reset_time_s": _f(row, "reset_time_s"),
            "output_swing_v": _f(row, "output_swing_v"),
            "power_w": _f(row, "power_w"),
            "input_referred_offset_v": _f(row, "input_referred_offset_v"),
            "input_referred_noise_v": _f(row, "input_referred_noise_v"),
            "kickback_noise_v": _f(row, "kickback_noise_v"),
        }
        candidates.append(
            {
                "sample": sample_id,
                "score": score,
                "feasible": (row.get("rl_feasible") or "").strip().lower() in {"true", "1", "yes"},
                "metrics": metrics,
            }
        )
        if len(candidates) >= top_k:
            break
    return candidates


def _fmt_metric(metrics: dict[str, Any], key: str, scale: float, fmt: str, unit: str) -> str | None:
    val = metrics.get(key)
    if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
        return None
    return f"{val * scale:{fmt}} {unit}"


def _plot_single_candidate(
    plt,
    *,
    rank: int,
    run_label: str,
    cand: dict[str, Any],
    folder: str,
    wf_path: Path,
    cfg: dict[str, Any],
    overview_cycles: int,
    title: str | None,
    out_path: Path,
    dpi: int,
    show: bool,
) -> Path:
    from matplotlib.gridspec import GridSpec

    vdd = float(cfg["vdd_v"])
    period_s = float(cfg["period_s"])
    decision_th = vdd / 2.0
    reset_th = float(cfg["reset_threshold_ratio"]) * vdd

    vecs = load_transient(
        wf_path,
        default_names=(cfg["output_positive"], cfg["output_negative"], cfg["clock_vector"], "i(vdd)"),
    )
    t = vecs["time"]
    t_ns = t * 1e9
    vx = vecs[cfg["output_positive"]]
    vy = vecs[cfg["output_negative"]]
    vck = vecs[cfg["clock_vector"]]
    vinp = vecs.get(cfg["input_positive"])
    vinn = vecs.get(cfg["input_negative"])
    p_inst = vecs.get(cfg["power_vector"])

    vdiff = vx - vy
    vdiff_abs = np.abs(vdiff)
    vmin_xy = np.minimum(vx, vy)

    rise_edges = _rising_crossings(t, vck, decision_th)
    fall_edges = _falling_crossings(t, vck, decision_th)

    delays: list[float] = []
    crossings: list[float | None] = []
    for edge in rise_edges:
        stop = min(edge + period_s / 2.0, float(t[-1]))
        tc = _first_rising_in_window(t, vdiff_abs, decision_th, edge, stop)
        delays.append((tc - edge) if tc is not None else math.nan)
        crossings.append(tc)

    if delays and any(math.isfinite(d) for d in delays):
        worst_idx = int(np.nanargmax(delays))
        worst_edge = rise_edges[worst_idx]
        worst_cross = crossings[worst_idx]
        worst_delay_ps = delays[worst_idx] * 1e12
    else:
        worst_idx = 0
        worst_edge = rise_edges[0] if rise_edges else float(t[0])
        worst_cross = None
        worst_delay_ps = math.nan

    fall_after = [fe for fe in fall_edges if fe > worst_edge]
    worst_fall = fall_after[0] if fall_after else worst_edge + period_s / 2.0
    rst_cross = _first_rising_in_window(
        t, vmin_xy, reset_th, worst_fall, min(worst_fall + period_s / 2.0, float(t[-1]))
    )
    rst_ps = ((rst_cross - worst_fall) * 1e12) if rst_cross is not None else math.nan

    metrics = cand.get("metrics") or {}
    n_left_rows = 4 if p_inst is not None else 3
    fig = plt.figure(figsize=(13.6, 7.6))
    gs = GridSpec(n_left_rows, 2, figure=fig, width_ratios=[1.42, 1.0], hspace=0.34, wspace=0.36)

    overview_end_ns = min(float(t_ns[-1]), overview_cycles * period_s * 1e9)
    t_win_mask = t_ns <= overview_end_ns
    tl = t_ns[t_win_mask]

    # Left Row 0: Clock & Differential Input
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(tl, vck[t_win_mask], color="#1f77b4", lw=1.6, label=cfg["clock_vector"])
    ax0.set_ylabel("Clock (V)", color="#1f77b4")
    ax0.set_ylim(-0.15, vdd + 0.25)
    ax0.set_title(f"First {overview_cycles} Cycles Overview (0 – {overview_end_ns:.2g} ns)")
    lines0, labels0 = ax0.get_legend_handles_labels()
    if vinp is not None and vinn is not None:
        vin_diff_mv = (vinp[t_win_mask] - vinn[t_win_mask]) * 1e3
        ax0_r = ax0.twinx()
        ax0_r.spines["top"].set_visible(False)
        ax0_r.grid(False)
        ax0_r.plot(tl, vin_diff_mv, color="#d62728", lw=1.4, ls="--", label="v(inp) - v(inn)")
        ax0_r.set_ylabel("Vin_diff (mV)", color="#d62728")
        vspan = max(float(np.max(np.abs(vin_diff_mv))), 10.0) * 1.4
        ax0_r.set_ylim(-vspan, vspan)
        lines0r, labels0r = ax0_r.get_legend_handles_labels()
        lines0 += lines0r
        labels0 += labels0r
    ax0.legend(lines0, labels0, loc="upper right", ncol=2)
    ax0.set_xlim(float(tl[0]), overview_end_ns)

    # Left Row 1: Single-ended Outputs
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(tl, vx[t_win_mask], color="#2ca02c", lw=1.5, label="v(X)")
    ax1.plot(tl, vy[t_win_mask], color="#ff7f0e", lw=1.5, label="v(Y)")
    ax1.axhline(decision_th, color="gray", ls=":", lw=1.0, label=f"VDD/2 ({decision_th:.2f} V)")
    ax1.axhline(
        reset_th,
        color="#9467bd",
        ls="--",
        lw=1.0,
        alpha=0.8,
        label=f"{cfg['reset_threshold_ratio']:.1g}·VDD ({reset_th:.2f} V)",
    )
    ax1.set_ylabel("Outputs (V)")
    y_lo = min(-0.35, float(np.min(vmin_xy[t_win_mask])) - 0.15)
    y_hi = max(vdd + 0.55, float(np.max(np.maximum(vx[t_win_mask], vy[t_win_mask]))) + 0.15)
    ax1.set_ylim(y_lo, y_hi)
    ax1.legend(loc="lower right", ncol=2)

    # Left Row 2: Differential Output
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(tl, vdiff[t_win_mask], color="#8c564b", lw=1.5, label="v(X) - v(Y)")
    ax2.axhline(decision_th, color="gray", ls=":", lw=1.0, label=f"±VDD/2 (±{decision_th:.2f} V)")
    ax2.axhline(-decision_th, color="gray", ls=":", lw=1.0)
    ax2.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax2.set_ylabel("vdiff (V)")
    ax2.set_ylim(-(vdd + 0.35), vdd + 0.35)
    ax2.legend(loc="upper right", ncol=2)

    # Left Row 3 (optional): Instantaneous Power
    left_axes = [ax0, ax1, ax2]
    if p_inst is not None:
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax3.plot(tl, p_inst[t_win_mask] * 1e3, color="#17becf", lw=1.2, label="p_inst (mW)")
        avg_p = metrics.get("power_w")
        if avg_p is not None and math.isfinite(avg_p):
            ax3.axhline(
                avg_p * 1e3,
                color="#d62728",
                ls="--",
                lw=1.2,
                label=f"Avg power ({avg_p * 1e6:.1f} µW)",
            )
        ax3.set_ylabel("Power (mW)")
        ax3.set_xlabel("Time (ns)")
        ax3.legend(loc="upper right", ncol=2)
        left_axes.append(ax3)
    else:
        ax2.set_xlabel("Time (ns)")

    for a in left_axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    # Right column: Zoom on worst-case evaluation + reset cycle
    z_start = max(float(t_ns[0]), (worst_edge - 0.1 * period_s) * 1e9)
    z_stop = min(float(t_ns[-1]), (worst_edge + 0.9 * period_s) * 1e9)
    zm = (t_ns >= z_start) & (t_ns <= z_stop)
    tz = t_ns[zm]

    split = 2 if n_left_rows == 4 else 1
    ax_zt = fig.add_subplot(gs[0:split, 1])
    ax_zt.plot(tz, vck[zm], color="#1f77b4", lw=1.4, ls="--", alpha=0.85, label="v(ck)")
    ax_zt.plot(tz, vx[zm], color="#2ca02c", lw=1.8, label="v(X)")
    ax_zt.plot(tz, vy[zm], color="#ff7f0e", lw=1.8, label="v(Y)")
    ax_zt.axhline(decision_th, color="gray", ls=":", lw=1.0)
    ax_zt.axhline(reset_th, color="#9467bd", ls="--", lw=1.0, alpha=0.8)
    if worst_cross is not None and math.isfinite(worst_delay_ps):
        ax_zt.axvspan(
            worst_edge * 1e9,
            worst_cross * 1e9,
            color="#ff9896",
            alpha=0.28,
            label=f"Eval delay ({worst_delay_ps:.1f} ps)",
        )
    if rst_cross is not None and math.isfinite(rst_ps):
        ax_zt.axvspan(
            worst_fall * 1e9,
            rst_cross * 1e9,
            color="#c5b0d5",
            alpha=0.32,
            label=f"Reset time ({rst_ps:.1f} ps)",
        )
    ax_zt.set_xlim(z_start, z_stop)
    ax_zt.set_ylim(y_lo, y_hi)
    ax_zt.set_ylabel("Voltage (V)")
    zoom_title = f"Zoom: Worst-Case Cycle #{worst_idx}"
    if vinp is not None and vinn is not None:
        vin_at_edge_mv = (float(np.interp(worst_edge, t, vinp)) - float(np.interp(worst_edge, t, vinn))) * 1e3
        zoom_title += f" (Vin_diff={vin_at_edge_mv:+.0f} mV)"
    ax_zt.set_title(zoom_title)
    ax_zt.legend(loc="lower right", fontsize=8.0)
    plt.setp(ax_zt.get_xticklabels(), visible=False)

    ax_zb = fig.add_subplot(gs[split:n_left_rows, 1], sharex=ax_zt)
    ax_zb.plot(tz, vdiff_abs[zm], color="#8c564b", lw=1.8, label="|v(X) - v(Y)|")
    ax_zb.plot(tz, vmin_xy[zm], color="#9467bd", lw=1.6, label="min(v(X), v(Y))")
    ax_zb.axhline(decision_th, color="gray", ls=":", lw=1.1, label=f"Decision th ({decision_th:.2f} V)")
    ax_zb.axhline(reset_th, color="#9467bd", ls="--", lw=1.1, alpha=0.8, label=f"Reset th ({reset_th:.2f} V)")
    ax_zb.axvline(worst_edge * 1e9, color="#1f77b4", ls=":", lw=1.0, alpha=0.7)
    if worst_cross is not None and math.isfinite(worst_delay_ps):
        ax_zb.scatter([worst_cross * 1e9], [decision_th], color="#d62728", s=45, zorder=5)
        ax_zb.annotate(
            f"delay = {worst_delay_ps:.1f} ps",
            xy=(worst_cross * 1e9, decision_th),
            xytext=(z_start + 0.03 * (z_stop - z_start), 0.58),
            fontsize=8.3,
            color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.9),
        )
    if rst_cross is not None and math.isfinite(rst_ps):
        ax_zb.axvline(worst_fall * 1e9, color="#1f77b4", ls=":", lw=1.0, alpha=0.7)
        ax_zb.scatter([rst_cross * 1e9], [reset_th], color="#6a3d9a", s=45, zorder=5)
        ax_zb.annotate(
            f"reset = {rst_ps:.1f} ps",
            xy=(rst_cross * 1e9, reset_th),
            xytext=(rst_cross * 1e9 + 0.06 * (z_stop - z_start), reset_th - 0.28),
            fontsize=8.3,
            color="#6a3d9a",
            arrowprops=dict(arrowstyle="->", color="#6a3d9a", lw=0.9),
        )
    ax_zb.set_xlim(z_start, z_stop)
    ax_zb.set_ylim(-0.25, y_hi - 0.1)
    ax_zb.set_ylabel("Voltage (V)")
    ax_zb.set_xlabel("Time (ns)")
    ax_zb.legend(loc="center right", fontsize=8.0)

    score = cand.get("score")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) and math.isfinite(score) else "n/a"
    header = title or f"Rank #{rank}: {folder}  ({cand['sample']}) — {run_label}, Score = {score_str}"
    summary_parts = []
    delay_s = _fmt_metric(metrics, "delay_s", 1e12, ".2f", "ps")
    delay_mean_s = _fmt_metric(metrics, "delay_mean_s", 1e12, ".2f", "ps")
    if delay_s and delay_mean_s:
        summary_parts.append(f"delay_max={delay_s} (mean {delay_mean_s})")
    elif delay_s:
        summary_parts.append(f"delay_max={delay_s}")
    for label, key, scale, fmt, unit in (
        ("reset", "reset_time_s", 1e12, ".2f", "ps"),
        ("swing", "output_swing_v", 1.0, ".3f", "V"),
        ("power", "power_w", 1e6, ".1f", "µW"),
        ("offset", "input_referred_offset_v", 1e3, ".2f", "mV"),
        ("noise", "input_referred_noise_v", 1e6, ".1f", "µV"),
        ("kickback", "kickback_noise_v", 1e3, ".2f", "mV"),
    ):
        formatted = _fmt_metric(metrics, key, scale, fmt, unit)
        if formatted:
            summary_parts.append(f"{label}={formatted}")

    suptitle = f"{header}\n{' | '.join(summary_parts)}" if summary_parts else header
    fig.suptitle(suptitle, fontsize=10.5, y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_transient_waveforms(
    csv_paths: Sequence[str | Path],
    *,
    output_dir: Path | None = None,
    top_k: int = 5,
    overview_cycles: int = 4,
    title: str | None = None,
    dpi: int = 180,
    show: bool = False,
    allow_missing: bool = False,
) -> list[Path]:
    """Render transient waveforms for the top-K best candidates of each run."""
    runs = load_results_csv(csv_paths)
    plt = configure_matplotlib(show)
    base_out_dir = output_dir or default_output_dir(runs)
    multi_run = len(runs) > 1

    saved_paths: list[Path] = []
    for run in runs:
        run_dir = run.path.parent
        cfg = _load_transient_config(run_dir)
        candidates = _load_top_candidates(run.path, top_k=top_k)
        for rank, cand in enumerate(candidates, start=1):
            wf_path, folder = _resolve_waveform_path(run_dir, cand["sample"], cfg["waveform"])
            if wf_path is None:
                continue
            prefix = f"{DEFAULT_STEM}_{run_dir.name}_rank{rank}_{folder}" if multi_run else f"{DEFAULT_STEM}_rank{rank}_{folder}"
            out_path = base_out_dir / f"{prefix}.png"
            saved_paths.append(
                _plot_single_candidate(
                    plt,
                    rank=rank,
                    run_label=run.label,
                    cand=cand,
                    folder=folder,
                    wf_path=wf_path,
                    cfg=cfg,
                    overview_cycles=overview_cycles,
                    title=title,
                    out_path=out_path,
                    dpi=dpi,
                    show=show,
                )
            )

    if not saved_paths and not allow_missing:
        raise FileNotFoundError("No transient waveform files found for top candidates in the given run(s).")
    return saved_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "Plot transient waveforms for top-K best candidates."
    )
    parser.add_argument(
        "--data-dir",
        "--csv",
        dest="data_dir",
        nargs="+",
        required=True,
        help="path(s) to run directory(ies) (e.g. run/strongarm/<run_id>) or results.csv file(s)",
    )
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of best candidates per run to plot (default: 5)",
    )
    parser.add_argument(
        "--overview-cycles",
        type=int,
        default=4,
        help="number of clock cycles shown in the overview panel (default: 4)",
    )
    parser.add_argument("--title", default=None, help="custom plot title")
    parser.add_argument("--dpi", type=int, default=180, help="output image DPI (default: 180)")
    parser.add_argument("--show", action="store_true", help="display plots interactively")
    args = parser.parse_args(argv)
    plot_transient_waveforms(
        args.data_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        overview_cycles=args.overview_cycles,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
