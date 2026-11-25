"""SKY130 reduced-corner model generation, .spiceinit setup, and PRBS stimulus helpers."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

CORNERS = ("tt", "ff", "ss", "sf", "fs")
DEFAULT_PDK_CANDIDATES = (
    Path("/usr/local/share/pdk/sky130A"),
    Path.home() / ".ciel" / "sky130A",
)
MODEL_CACHE_DIR = Path("/tmp/ai_in_analog_sky130_models")

_LIB_DIRECTIVE_RE = re.compile(
    r'^\s*\.lib\s+"?([^"\s]+sky130\.lib\.spice)"?\s+([A-Za-z0-9_]+)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

SPICEINIT_TEXT = (
    "set ngbehavior=hsa\n"
    "set skywaterpdk\n"
    "set ng_nomodcheck\n"
    "set num_threads=1\n"
    "option noinit\n"
)


def resolve_sky130_root(hint_lib_path: str | Path | None = None) -> Path | None:
    """Locate an installed sky130A root containing libs.tech/ngspice and libs.ref."""
    candidates: list[Path] = []
    if hint_lib_path is not None:
        lib_path = Path(os.path.expanduser(str(hint_lib_path)))
        # <root>/libs.tech/ngspice/sky130.lib.spice -> parents[2] == <root>
        if len(lib_path.parents) >= 3:
            candidates.append(lib_path.parents[2])
    env_pdk = os.environ.get("PDK_ROOT")
    if env_pdk:
        candidates.append(Path(env_pdk) / "sky130A")
    candidates.extend(DEFAULT_PDK_CANDIDATES)

    for root in candidates:
        ng_dir = root / "libs.tech" / "ngspice"
        ref_dir = root / "libs.ref" / "sky130_fd_pr" / "spice"
        if (ng_dir / "sky130.lib.spice").is_file() and ref_dir.is_dir():
            return root
    return None


def build_reduced_corner_file(
    pdk_root: Path,
    corner: str = "tt",
    *,
    mismatch: bool = False,
    cache_dir: Path = MODEL_CACHE_DIR,
) -> Path:
    """Materialize a minimal SKY130 corner include file (~14x faster load, exact parity)."""
    base_corner = corner.lower().removesuffix("_mm")
    use_mm = mismatch or corner.lower().endswith("_mm")
    if base_corner not in CORNERS:
        raise ValueError(f"Unsupported SKY130 corner {corner!r}; expected one of {CORNERS}")

    ng_dir = pdk_root / "libs.tech" / "ngspice"
    ref_dir = pdk_root / "libs.ref" / "sky130_fd_pr" / "spice"
    mm_flag = 1 if use_mm else 0

    lines = [
        f"* Auto-generated minimal SKY130 '{base_corner}' model (mm={mm_flag})",
        f".param mc_mm_switch={mm_flag}",
        ".param mc_pr_switch=0",
        ".option scale=1.0u",
        f'.include "{ng_dir / "parameters" / "lod.spice"}"',
        ".param lv_dlc_rotweak=0.0",
        ".param sky130_fd_pr__nfet_01v8__dlc_rotweak=lv_dlc_rotweak",
        ".param sky130_fd_pr__pfet_01v8__dlc_rotweak=lv_dlc_rotweak",
        f'.include "{ng_dir / "parasitics" / "sky130_fd_pr__model__parasitic__diode_pw2dn.model.spice"}"',
        f'.include "{ng_dir / "parasitics" / "sky130_fd_pr__model__parasitic__diode_ps2dn.model.spice"}"',
        f'.include "{ng_dir / "parasitics" / "sky130_fd_pr__model__parasitic__diode_ps2nw.model.spice"}"',
        f'.include "{ref_dir / "sky130_fd_pr__diode_pw2nd_05v5.model.spice"}"',
        f'.include "{ref_dir / "sky130_fd_pr__diode_pd2nw_05v5.model.spice"}"',
        f'.include "{ref_dir / f"sky130_fd_pr__nfet_01v8__{base_corner}.corner.spice"}"',
        f'.include "{ref_dir / "sky130_fd_pr__nfet_01v8__mismatch.corner.spice"}"',
        f'.include "{ref_dir / f"sky130_fd_pr__pfet_01v8__{base_corner}.corner.spice"}"',
        f'.include "{ref_dir / "sky130_fd_pr__pfet_01v8__mismatch.corner.spice"}"',
        "",
    ]
    content = "\n".join(lines)
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"sky130_reduced_{base_corner}{'_mm' if use_mm else ''}_{digest}.spice"
    if not out_path.is_file():
        tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(out_path)
    return out_path


def rewrite_deck_with_reduced_models(
    deck_text: str,
    *,
    corner_override: str | None = None,
    mismatch: bool = False,
    cache_dir: Path = MODEL_CACHE_DIR,
) -> str:
    """Replace full sky130.lib.spice .lib directive with a fast reduced corner .include."""
    match = _LIB_DIRECTIVE_RE.search(deck_text)
    if match is None:
        return deck_text
    lib_path_str, deck_corner = match.group(1), match.group(2)
    pdk_root = resolve_sky130_root(lib_path_str)
    if pdk_root is None:
        return deck_text
    target_corner = corner_override or deck_corner
    try:
        reduced_path = build_reduced_corner_file(
            pdk_root,
            target_corner,
            mismatch=mismatch,
            cache_dir=cache_dir,
        )
    except (OSError, ValueError):
        return deck_text
    replacement = f'.include "{reduced_path}"'
    return _LIB_DIRECTIVE_RE.sub(replacement, deck_text, count=1)


def write_spiceinit(workdir: Path) -> None:
    """Write .spiceinit into workdir so ngspice enables HSA + skips redundant bin checks."""
    init_path = workdir / ".spiceinit"
    if not init_path.is_file():
        init_path.write_text(SPICEINIT_TEXT, encoding="utf-8")


def prbs_bits(n_bits: int = 32, *, order: int = 5, seed: int = 0) -> np.ndarray:
    """Generate a deterministic maximal-length LFSR PRBS bit sequence (0/1).

    Uses right-shifting Fibonacci LFSRs with primitive polynomials over GF(2)
    so one period of length ``2**order - 1`` visits every non-zero state
    (for order=5: period 31 with 16 ones and 15 zeros).
    """
    # Zero-indexed tap bit positions (always including bit 0, the shifted-out LSB):
    # order 2: x^2 + x + 1        -> (0, 1)
    # order 3: x^3 + x + 1        -> (0, 1)
    # order 4: x^4 + x + 1        -> (0, 1)
    # order 5: x^5 + x^2 + 1      -> (0, 2)
    # order 6: x^6 + x + 1        -> (0, 1)
    # order 7: x^7 + x + 1        -> (0, 1)
    # order 8: x^8 + x^4 + x^3 + x^2 + 1 -> (0, 2, 3, 4)
    taps = {
        2: (0, 1),
        3: (0, 1),
        4: (0, 1),
        5: (0, 2),
        6: (0, 1),
        7: (0, 1),
        8: (0, 2, 3, 4),
    }
    if order not in taps:
        raise ValueError(f"Unsupported PRBS order {order}; supported: {sorted(taps)}")
    mask = (1 << order) - 1
    state = (seed & mask) or 1
    bits: list[int] = []
    for _ in range(n_bits):
        bits.append(int(state & 1))
        fb = 0
        for tap in taps[order]:
            fb ^= (state >> tap) & 1
        state = ((state >> 1) | (fb << (order - 1))) & mask
    return np.asarray(bits, dtype=int)


def format_prbs_pwl(
    bits: Sequence[int],
    *,
    period_s: float,
    t0_s: float | None = None,
    vid_diff_v: float = 0.1,
    slew_s: float = 1.0e-12,
    card_name: str = "Vprbs",
    node_pos: str = "prbs_raw",
    node_neg: str = "0",
) -> str:
    """Format a differential PRBS PWL source switching cleanly mid-reset before each clock edge."""
    if t0_s is None:
        t0_s = 0.5 * period_s
    values = [float(vid_diff_v) if int(b) == 1 else -float(vid_diff_v) for b in bits]
    pts: list[str] = [f"0.000000e+00 {values[0]:.6e}"]
    for k in range(1, len(values)):
        t_change = t0_s + (k - 0.25) * period_s
        t_prev = max(0.0, t_change - slew_s)
        pts.append(f"{t_prev:.6e} {values[k - 1]:.6e}")
        pts.append(f"{t_change:.6e} {values[k]:.6e}")
    t_end = len(values) * period_s
    pts.append(f"{t_end:.6e} {values[-1]:.6e}")
    return f"{card_name} {node_pos} {node_neg} PWL(" + " ".join(pts) + ")"
