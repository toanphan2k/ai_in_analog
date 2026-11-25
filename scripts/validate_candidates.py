#!/usr/bin/env python3
"""Post-search validation for top-K candidates: small-input delay, PVT corners, and MC offset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from runner.ngspice_runner import NgspiceRun
from scripts.deck_generation import render_sized_deck
from scripts.extract_delay import extract_delay
from scripts.netlist_tools import load_sizing_config
from scripts.pdk_models import (
    CORNERS,
    format_prbs_pwl,
    prbs_bits,
    resolve_sky130_root,
    rewrite_deck_with_reduced_models,
    write_spiceinit,
)
from scripts.waveform_io import load_transient
from src.selector import variable_names

SMALL_INPUT_LEVELS_V = (0.001, 0.005, 0.010, 0.100)
TEMPERATURES_C = (0.0, 27.0, 85.0)
VDD_SCALES = (0.9, 1.0, 1.1)

# Default SKY130 mismatch slopes (V*um) if PDK lookup unavailable
DEFAULT_AVT_N = 3.356e-3
DEFAULT_AVT_P = 5.856e-3


def _read_pdk_avt(pdk_root: Path | None) -> tuple[float, float]:
    if pdk_root is None:
        return DEFAULT_AVT_N, DEFAULT_AVT_P
    ref_dir = pdk_root / "libs.ref" / "sky130_fd_pr" / "spice"
    out = {"nfet_01v8": DEFAULT_AVT_N, "pfet_01v8": DEFAULT_AVT_P}
    for fam in tuple(out):
        mm_file = ref_dir / f"sky130_fd_pr__{fam}__mismatch.corner.spice"
        if not mm_file.is_file():
            continue
        prefix = f".param sky130_fd_pr__{fam}__vth0_slope="
        for raw in mm_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if line.startswith(prefix):
                try:
                    out[fam] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
                break
    return out["nfet_01v8"], out["pfet_01v8"]


def _load_top_candidates(
    run_dir: Path,
    sizing_config: Mapping[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    design_names = variable_names(sizing_config)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _parse_csv_row(row: Mapping[str, str]) -> dict[str, Any] | None:
        sample = (row.get("sample") or "").strip()
        if not sample or sample in seen:
            return None
        size: dict[str, float] = {}
        for col in design_names:
            raw = (row.get(col) or "").strip()
            if not raw:
                return None
            size[col] = float(raw)
        pelgrom_raw = (row.get("input_referred_offset_v") or "").strip()
        pelgrom_v = float(pelgrom_raw) if pelgrom_raw else math.nan
        seen.add(sample)
        return {"sample": sample, "size": size, "pelgrom_offset_v": pelgrom_v}

    cand_csv = run_dir / "candidates.csv"
    if cand_csv.is_file():
        with cand_csv.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                parsed = _parse_csv_row(row)
                if parsed is not None:
                    candidates.append(parsed)
                    if len(candidates) >= top_k:
                        return candidates

    cand_json = run_dir / "candidates.json"
    if cand_json.is_file() and len(candidates) < top_k:
        payload = json.loads(cand_json.read_text(encoding="utf-8"))
        for item in payload.get("candidates", []):
            sample = str(item.get("sample", "")).strip()
            if not sample or sample in seen:
                continue
            size = {k: float(v) for k, v in (item.get("size") or {}).items() if k in design_names}
            if len(size) != len(design_names):
                continue
            m = item.get("metrics") or {}
            pelgrom_v = float(m["input_referred_offset_v"]) if m.get("input_referred_offset_v") is not None else math.nan
            seen.add(sample)
            candidates.append({"sample": sample, "size": size, "pelgrom_offset_v": pelgrom_v})
            if len(candidates) >= top_k:
                return candidates

    res_csv = run_dir / "results.csv"
    if res_csv.is_file() and len(candidates) < top_k:
        with res_csv.open("r", newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if (r.get("status") or "").strip() == "evaluated"]
        rows.sort(
            key=lambda r: (
                0 if (r.get("rl_feasible") or "").strip().lower() in {"true", "1", "yes"} else 1,
                float(r["rl_objective_score"]) if (r.get("rl_objective_score") or "").strip() else math.inf,
            )
        )
        for row in rows:
            parsed = _parse_csv_row(row)
            if parsed is not None:
                candidates.append(parsed)
                if len(candidates) >= top_k:
                    break
    return candidates


def _load_selected_candidates(
    run_dir: Path,
    sizing_config: Mapping[str, Any],
    sample_queries: Sequence[str],
) -> list[dict[str, Any]]:
    """Load specific samples by full sample name or numeric/suffix token (e.g. '010697')."""
    design_names = variable_names(sizing_config)
    records_by_name: dict[str, dict[str, Any]] = {}

    def _ingest_csv(csv_path: Path) -> None:
        if not csv_path.is_file():
            return
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sample = (row.get("sample") or "").strip()
                if not sample or sample in records_by_name:
                    continue
                size: dict[str, float] = {}
                valid = True
                for col in design_names:
                    raw = (row.get(col) or "").strip()
                    if not raw:
                        valid = False
                        break
                    size[col] = float(raw)
                if not valid:
                    continue
                pelgrom_raw = (row.get("input_referred_offset_v") or "").strip()
                pelgrom_v = float(pelgrom_raw) if pelgrom_raw else math.nan
                records_by_name[sample] = {
                    "sample": sample,
                    "size": size,
                    "pelgrom_offset_v": pelgrom_v,
                }

    _ingest_csv(run_dir / "results.csv")
    _ingest_csv(run_dir / "candidates.csv")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_query in sample_queries:
        q = raw_query.strip()
        if not q:
            continue
        match = records_by_name.get(q)
        if match is None:
            norm_suffix = q.removeprefix("sample_")
            matches = [
                rec
                for name, rec in records_by_name.items()
                if name == q or name.endswith(f"_{norm_suffix}") or name.endswith(norm_suffix)
            ]
            if len(matches) == 1:
                match = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous sample selector {q!r}: matched {[m['sample'] for m in matches[:5]]}"
                )
        if match is None:
            raise ValueError(f"Sample {q!r} not found in {run_dir / 'results.csv'}")
        if match["sample"] not in seen:
            seen.add(match["sample"])
            selected.append(match)
    return selected


def _resolve_template_path(run_dir: Path, netlist_override: Path | None) -> Path:
    if netlist_override is not None:
        cand = netlist_override / "netlist.spice" if netlist_override.is_dir() else netlist_override
        if cand.is_file():
            return cand.resolve()
        raise FileNotFoundError(f"Netlist template not found: {cand}")
    design_name = run_dir.parent.name
    cand = PROJECT_DIR / "netlists" / design_name / "netlist.spice"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Could not infer netlist template from {run_dir} (checked {cand}); pass --netlist."
    )


def _build_transient_only_deck(
    sized_deck: str,
    *,
    corner: str,
    temp_c: float,
    vdd_v: float,
    period_s: float,
    vid_diff_v: float,
    n_cycles: int = 16,
    constant_vid_v: float | None = None,
    gate_shifts_v: Mapping[str, float] | None = None,
) -> str:
    """Construct a fast transient-only SPICE deck from a sized StrongARM deck."""
    deck = rewrite_deck_with_reduced_models(sized_deck, corner_override=corner)
    vcm_v = 0.5 * vdd_v

    # Strip original .control ... .endc and replace with transient-only block
    control_idx = deck.lower().find("\n.control")
    if control_idx == -1:
        raise ValueError("Template deck is missing a .control section")
    header_and_body = deck[:control_idx]

    # Override temperature, VDD, Vcm, clock amplitude, and stimulus
    header_and_body = re.sub(r"(?m)^\.temp\s+\S+", f".temp {temp_c:g}", header_and_body)
    header_and_body = re.sub(r"(?m)^\.param\s+VDD=\S+", f".param VDD={vdd_v:.6g}", header_and_body)
    header_and_body = re.sub(r"(?m)^VDD\s+vdd\s+0\s+\S+", f"VDD vdd 0 {vdd_v:.6g}", header_and_body)
    header_and_body = re.sub(r"(?m)^Vcm\s+vcm_node\s+0\s+DC\s+\S+", f"Vcm vcm_node 0 DC {vcm_v:.6g}", header_and_body)
    header_and_body = re.sub(
        r"(?m)^(VCK_pulse\s+\S+\s+0\s+PULSE\(\s*0\s+)\S+",
        rf"\g<1>{vdd_v:.6g}",
        header_and_body,
    )

    if constant_vid_v is not None:
        stim_card = f"Vprbs prbs_raw 0 DC {constant_vid_v:.8e}"
    else:
        bits = prbs_bits(n_cycles, order=5, seed=0)
        stim_card = format_prbs_pwl(bits, period_s=period_s, vid_diff_v=vid_diff_v, slew_s=1.0e-12)

    if re.search(r"(?m)^Vprbs\s+", header_and_body):
        header_and_body = re.sub(r"(?m)^Vprbs\s+.*$", stim_card, header_and_body, count=1)
    else:
        # Legacy deck fallback (Vinp/Vinn PWL directly)
        v_hi = vcm_v + 0.5 * (constant_vid_v if constant_vid_v is not None else vid_diff_v)
        v_lo = vcm_v - 0.5 * (constant_vid_v if constant_vid_v is not None else vid_diff_v)
        header_and_body = re.sub(r"(?m)^Vinp\s+src_p_tr\s+0\s+.*$", f"Vinp src_p_tr 0 DC {v_hi:.8e}", header_and_body)
        header_and_body = re.sub(r"(?m)^Vinn\s+src_n_tr\s+0\s+.*$", f"Vinn src_n_tr 0 DC {v_lo:.8e}", header_and_body)

    # Optional per-device gate Vth shift injections for Monte-Carlo offset bisection
    if gate_shifts_v:
        out_lines: list[str] = []
        inj_sources: list[str] = []
        for line in header_and_body.splitlines():
            stripped = line.strip()
            tok = stripped.split()[0] if stripped and not stripped.startswith(("*", ".")) else ""
            if tok in gate_shifts_v and abs(gate_shifts_v[tok]) > 0.0:
                parts = stripped.split()
                # SPICE subckt instance: Xname d g s b model ...
                orig_gate = parts[2]
                inj_node = f"g_mc_{tok.lower()}"
                parts[2] = inj_node
                inj_sources.append(f"Vmc_{tok} {inj_node} {orig_gate} DC {-gate_shifts_v[tok]:.8e}")
                out_lines.append(" ".join(parts))
            else:
                out_lines.append(line)
        header_and_body = "\n".join(out_lines + inj_sources)

    tstop = n_cycles * period_s
    tstep = period_s / 300.0
    control_block = "\n".join([
        "",
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        f"tran {tstep:.6e} {tstop:.6e}",
        "let vdiff = v(X)-v(Y)",
        "let p_inst = -v(vdd)*i(VDD)",
        "wrdata transient.txt v(X) v(Y) v(ck) v(inp) v(inn) vdiff i(VDD) p_inst",
        ".endc",
        ".end",
        "",
    ])
    return header_and_body + control_block


def _run_transient_eval(
    deck_text: str,
    *,
    vdd_v: float,
    period_s: float,
    reset_ratio: float,
    inverting: bool,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="val_sa_"))
    try:
        write_spiceinit(workdir)
        deck_path = workdir / "netlist.spice"
        deck_path.write_text(deck_text, encoding="utf-8")
        runner = NgspiceRun(timeout_s=timeout_s)
        res = runner.run(deck_path, workdir, execute=True)
        wf = workdir / "transient.txt"
        if res.get("returncode") != 0 or not wf.is_file() or wf.stat().st_size == 0:
            return {"status": "failed", "error": f"ngspice rc={res.get('returncode')}"}
        metrics = extract_delay(
            wf,
            vdd_v,
            period_s,
            reset_threshold_ratio=reset_ratio,
            inverting=inverting,
            check_polarity=True,
        )
        return {"status": "evaluated", "error": "", **metrics}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _decide_sign_single_cycle(
    sized_deck: str,
    *,
    vdd_v: float,
    period_s: float,
    vid_v: float,
    gate_shifts_v: Mapping[str, float],
    inverting: bool,
    timeout_s: float = 45.0,
) -> int:
    """Return +1 if comparator decides logic '1' (as if Vin_diff > Vos), else -1."""
    deck_text = _build_transient_only_deck(
        sized_deck,
        corner="tt",
        temp_c=27.0,
        vdd_v=vdd_v,
        period_s=period_s,
        vid_diff_v=abs(vid_v),
        n_cycles=2,
        constant_vid_v=vid_v,
        gate_shifts_v=gate_shifts_v,
    )
    workdir = Path(tempfile.mkdtemp(prefix="val_mc_"))
    try:
        write_spiceinit(workdir)
        deck_path = workdir / "netlist.spice"
        deck_path.write_text(deck_text, encoding="utf-8")
        res = NgspiceRun(timeout_s=timeout_s).run(deck_path, workdir, execute=True)
        wf = workdir / "transient.txt"
        if res.get("returncode") != 0 or not wf.is_file() or wf.stat().st_size == 0:
            return 0
        vecs = load_transient(wf)
        t = vecs["time"]
        vx, vy, vck = vecs["v(x)"], vecs["v(y)"], vecs["v(ck)"]
        idx = np.flatnonzero((vck[:-1] < 0.5 * vdd_v) & (vck[1:] >= 0.5 * vdd_v))
        if idx.size == 0:
            return 0
        # Use second cycle if present (settled past initial OP), else first
        cyc = min(1, idx.size - 1)
        t_eval_end = min(float(t[-1]), float(t[idx[cyc]] + 0.48 * period_s))
        vdiff_end = float(np.interp(t_eval_end, t, vx - vy))
        # Inverting StrongARM: Vin_diff > Vos -> X falls, Y stays high -> X - Y < 0
        if inverting:
            return 1 if vdiff_end < 0.0 else -1
        return 1 if vdiff_end > 0.0 else -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _bisect_mc_offset_one(
    sized_deck: str,
    size: Mapping[str, float],
    *,
    sample_idx: int,
    seed: int,
    avt_n: float,
    avt_p: float,
    l_n: float,
    l_p: float,
    l_rst: float,
    vdd_v: float,
    period_s: float,
    inverting: bool,
    span_v: float = 0.04,
    bisect_iters: int = 10,
) -> float:
    rng = np.random.default_rng(seed + sample_idx * 7919)
    sig_in = avt_n / math.sqrt(max(size["w_in"] * size["l_in"], 1e-6))
    sig_n = avt_n / math.sqrt(max(size["w_n"] * l_n, 1e-6))
    sig_p = avt_p / math.sqrt(max(size["w_p"] * l_p, 1e-6))
    sig_rst = avt_p / math.sqrt(max(size["w_rst"] * l_rst, 1e-6))

    shifts = {
        "XM1": float(rng.normal(0.0, sig_in)),
        "XM2": float(rng.normal(0.0, sig_in)),
        "XM3": float(rng.normal(0.0, sig_n)),
        "XM4": float(rng.normal(0.0, sig_n)),
        "XM5": float(rng.normal(0.0, sig_p)),
        "XM6": float(rng.normal(0.0, sig_p)),
        "XMr1": float(rng.normal(0.0, sig_rst)),
        "XMr2": float(rng.normal(0.0, sig_rst)),
    }

    lo, hi = -span_v, span_v
    s_lo = _decide_sign_single_cycle(
        sized_deck, vdd_v=vdd_v, period_s=period_s, vid_v=lo, gate_shifts_v=shifts, inverting=inverting
    )
    s_hi = _decide_sign_single_cycle(
        sized_deck, vdd_v=vdd_v, period_s=period_s, vid_v=hi, gate_shifts_v=shifts, inverting=inverting
    )
    if s_lo == 0 or s_hi == 0 or s_lo == s_hi:
        return math.nan

    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        s_mid = _decide_sign_single_cycle(
            sized_deck, vdd_v=vdd_v, period_s=period_s, vid_v=mid, gate_shifts_v=shifts, inverting=inverting
        )
        if s_mid == 0:
            return math.nan
        if s_mid == s_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def validate_run_candidates(
    run_dir: Path,
    *,
    netlist_override: Path | None = None,
    top_k: int = 5,
    samples: Sequence[str] | None = None,
    val_subdir: str = "validation",
    workers: int = 8,
    mc_samples: int = 16,
    bisect_iters: int = 10,
    fast: bool = False,
    seed: int = 0,
) -> Path:
    run_dir = run_dir.resolve()
    sizing_path = run_dir / "sizing.yaml"
    if not sizing_path.is_file():
        raise FileNotFoundError(f"sizing.yaml not found in {run_dir}")
    sizing_cfg = load_sizing_config(sizing_path)
    template_path = _resolve_template_path(run_dir, netlist_override)
    template_text = template_path.read_text(encoding="utf-8")

    if samples:
        candidates = _load_selected_candidates(run_dir, sizing_cfg, samples)
    else:
        candidates = _load_top_candidates(run_dir, sizing_cfg, top_k=top_k)
    if not candidates:
        raise ValueError(f"No evaluated candidates found to validate in {run_dir}")

    tran_cfg = (sizing_cfg.get("analyses") or {}).get("transient") or {}
    vdd_nom = float(tran_cfg.get("vdd_v", 1.8))
    period_s = float(tran_cfg.get("period_s", 2.0e-10))
    reset_ratio = float(tran_cfg.get("reset_threshold_ratio", 0.9))
    inverting = bool(tran_cfg.get("inverting", True))
    params_cfg = sizing_cfg.get("parameters") or {}
    l_n = float((params_cfg.get("l_n") or {}).get("default", 0.15))
    l_p = float((params_cfg.get("l_p") or {}).get("default", 0.15))
    l_rst = float((params_cfg.get("l_rst") or {}).get("default", 0.15))

    pdk_root = resolve_sky130_root()
    avt_n, avt_p = _read_pdk_avt(pdk_root)

    if fast:
        pvt_grid = [
            ("tt", 27.0, 1.0),
            ("ff", 0.0, 1.1),
            ("ss", 85.0, 0.9),
            ("sf", 85.0, 0.9),
            ("fs", 0.0, 0.9),
        ]
        n_cycles = 8
    else:
        pvt_grid = [(c, temp, vs) for c in CORNERS for temp in TEMPERATURES_C for vs in VDD_SCALES]
        n_cycles = 16

    small_input_rows: list[dict[str, Any]] = []
    pvt_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for rank, cand in enumerate(candidates, start=1):
            sample_name = cand["sample"]
            size = cand["size"]
            sized_deck = render_sized_deck(template_text, size, sizing_cfg)
            print(f"[{rank}/{len(candidates)}] Validating {sample_name} ...")

            # 1. Small-input overdrive sweep
            si_futures = []
            for vid_v in SMALL_INPUT_LEVELS_V:
                deck = _build_transient_only_deck(
                    sized_deck,
                    corner="tt",
                    temp_c=27.0,
                    vdd_v=vdd_nom,
                    period_s=period_s,
                    vid_diff_v=vid_v,
                    n_cycles=n_cycles,
                )
                si_futures.append(
                    (
                        vid_v,
                        pool.submit(
                            _run_transient_eval,
                            deck,
                            vdd_v=vdd_nom,
                            period_s=period_s,
                            reset_ratio=reset_ratio,
                            inverting=inverting,
                        ),
                    )
                )
            si_by_mv: dict[int, dict[str, Any]] = {}
            for vid_v, fut in si_futures:
                res = fut.result()
                mv = int(round(vid_v * 1e3))
                si_by_mv[mv] = res
                small_input_rows.append(
                    {
                        "rank": rank,
                        "sample": sample_name,
                        "vin_diff_mv": mv,
                        "status": res["status"],
                        "delay_s": res.get("delay_s", ""),
                        "delay_mean_s": res.get("delay_mean_s", ""),
                        "reset_time_s": res.get("reset_time_s", ""),
                        "output_swing_v": res.get("output_swing_v", ""),
                        "voh_v": res.get("voh_v", ""),
                        "vol_v": res.get("vol_v", ""),
                        "overshoot_v": res.get("overshoot_v", ""),
                        "error": res.get("error", ""),
                    }
                )

            # 2. PVT corner sweep
            pvt_futures = []
            for corner, temp_c, v_scale in pvt_grid:
                vdd_c = round(vdd_nom * v_scale, 4)
                deck = _build_transient_only_deck(
                    sized_deck,
                    corner=corner,
                    temp_c=temp_c,
                    vdd_v=vdd_c,
                    period_s=period_s,
                    vid_diff_v=0.100,
                    n_cycles=n_cycles,
                )
                pvt_futures.append(
                    (
                        corner,
                        temp_c,
                        vdd_c,
                        pool.submit(
                            _run_transient_eval,
                            deck,
                            vdd_v=vdd_c,
                            period_s=period_s,
                            reset_ratio=reset_ratio,
                            inverting=inverting,
                        ),
                    )
                )
            pvt_pass = 0
            pvt_delays_ps: list[float] = []
            pvt_resets_ps: list[float] = []
            pvt_swings_v: list[float] = []
            for corner, temp_c, vdd_c, fut in pvt_futures:
                res = fut.result()
                if res["status"] == "evaluated":
                    pvt_pass += 1
                    pvt_delays_ps.append(float(res["delay_s"]) * 1e12)
                    if res.get("reset_time_s") is not None:
                        pvt_resets_ps.append(float(res["reset_time_s"]) * 1e12)
                    pvt_swings_v.append(float(res["output_swing_v"]))
                pvt_rows.append(
                    {
                        "rank": rank,
                        "sample": sample_name,
                        "corner": corner,
                        "temp_c": temp_c,
                        "vdd_v": vdd_c,
                        "status": res["status"],
                        "delay_s": res.get("delay_s", ""),
                        "reset_time_s": res.get("reset_time_s", ""),
                        "output_swing_v": res.get("output_swing_v", ""),
                        "voh_v": res.get("voh_v", ""),
                        "vol_v": res.get("vol_v", ""),
                        "overshoot_v": res.get("overshoot_v", ""),
                        "error": res.get("error", ""),
                    }
                )

            # 3. Monte-Carlo transient offset bisection
            mc_futures = [
                (
                    mc_i,
                    pool.submit(
                        _bisect_mc_offset_one,
                        sized_deck,
                        size,
                        sample_idx=mc_i,
                        seed=seed + rank * 100000,
                        avt_n=avt_n,
                        avt_p=avt_p,
                        l_n=l_n,
                        l_p=l_p,
                        l_rst=l_rst,
                        vdd_v=vdd_nom,
                        period_s=period_s,
                        inverting=inverting,
                        bisect_iters=bisect_iters,
                    ),
                )
                for mc_i in range(1, mc_samples + 1)
            ]
            vos_values: list[float] = []
            for mc_i, fut in mc_futures:
                vos_v = fut.result()
                if math.isfinite(vos_v):
                    vos_values.append(vos_v)
                mc_rows.append(
                    {
                        "rank": rank,
                        "sample": sample_name,
                        "mc_index": mc_i,
                        "vos_v": vos_v if math.isfinite(vos_v) else "",
                    }
                )

            mc_sigma_mv = float(np.std(vos_values, ddof=1) * 1e3) if len(vos_values) >= 2 else math.nan
            mc_mean_mv = float(np.mean(vos_values) * 1e3) if vos_values else math.nan
            pelgrom_mv = (
                cand["pelgrom_offset_v"] * 1e3 if math.isfinite(cand["pelgrom_offset_v"]) else math.nan
            )

            def _d_ps(mv: int) -> str | float:
                r = si_by_mv.get(mv, {})
                return round(float(r["delay_s"]) * 1e12, 3) if r.get("delay_s") is not None else ""

            summary_rows.append(
                {
                    "rank": rank,
                    "sample": sample_name,
                    "delay_1mv_ps": _d_ps(1),
                    "delay_5mv_ps": _d_ps(5),
                    "delay_10mv_ps": _d_ps(10),
                    "delay_100mv_ps": _d_ps(100),
                    "pvt_pass": pvt_pass,
                    "pvt_total": len(pvt_grid),
                    "pvt_worst_delay_ps": round(max(pvt_delays_ps), 3) if pvt_delays_ps else "",
                    "pvt_worst_reset_ps": round(max(pvt_resets_ps), 3) if pvt_resets_ps else "",
                    "pvt_min_swing_v": round(min(pvt_swings_v), 4) if pvt_swings_v else "",
                    "pelgrom_offset_mv": round(pelgrom_mv, 3) if math.isfinite(pelgrom_mv) else "",
                    "mc_offset_sigma_mv": round(mc_sigma_mv, 3) if math.isfinite(mc_sigma_mv) else "",
                    "mc_offset_mean_mv": round(mc_mean_mv, 3) if math.isfinite(mc_mean_mv) else "",
                    "mc_valid_samples": len(vos_values),
                }
            )

    val_dir = run_dir / val_subdir
    _write_csv(
        val_dir / "small_input_delay.csv",
        (
            "rank",
            "sample",
            "vin_diff_mv",
            "status",
            "delay_s",
            "delay_mean_s",
            "reset_time_s",
            "output_swing_v",
            "voh_v",
            "vol_v",
            "overshoot_v",
            "error",
        ),
        small_input_rows,
    )
    _write_csv(
        val_dir / "pvt_corners.csv",
        (
            "rank",
            "sample",
            "corner",
            "temp_c",
            "vdd_v",
            "status",
            "delay_s",
            "reset_time_s",
            "output_swing_v",
            "voh_v",
            "vol_v",
            "overshoot_v",
            "error",
        ),
        pvt_rows,
    )
    _write_csv(
        val_dir / "monte_carlo_offset.csv",
        ("rank", "sample", "mc_index", "vos_v"),
        mc_rows,
    )
    _write_csv(
        val_dir / "validation_summary.csv",
        (
            "rank",
            "sample",
            "delay_1mv_ps",
            "delay_5mv_ps",
            "delay_10mv_ps",
            "delay_100mv_ps",
            "pvt_pass",
            "pvt_total",
            "pvt_worst_delay_ps",
            "pvt_worst_reset_ps",
            "pvt_min_swing_v",
            "pelgrom_offset_mv",
            "mc_offset_sigma_mv",
            "mc_offset_mean_mv",
            "mc_valid_samples",
        ),
        summary_rows,
    )
    print(f"Saved validation results to: {val_dir}")
    return val_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="run output directory containing candidates.csv / results.csv",
    )
    parser.add_argument("--netlist", type=Path, default=None, help="optional override netlist directory")
    parser.add_argument("--top-k", type=int, default=5, help="number of top candidates to validate (default: 5)")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="specific sample names or numeric suffixes to validate instead of --top-k (e.g. 010697 010707)",
    )
    parser.add_argument(
        "--val-subdir",
        type=str,
        default="validation",
        help="output subdirectory under --output-dir (default: validation)",
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel ngspice workers (default: 8)")
    parser.add_argument("--mc-samples", type=int, default=16, help="Monte-Carlo mismatch samples per candidate (default: 16)")
    parser.add_argument("--bisect-iters", type=int, default=10, help="bisection steps per MC offset sample (default: 10)")
    parser.add_argument("--fast", action="store_true", help="run 5 extreme PVT corners instead of full 45-corner grid")
    parser.add_argument("--seed", type=int, default=0, help="random seed for Monte-Carlo mismatch")
    args = parser.parse_args(argv)

    validate_run_candidates(
        args.output_dir,
        netlist_override=args.netlist,
        top_k=args.top_k,
        samples=args.samples,
        val_subdir=args.val_subdir,
        workers=args.workers,
        mc_samples=args.mc_samples,
        bisect_iters=args.bisect_iters,
        fast=args.fast,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
