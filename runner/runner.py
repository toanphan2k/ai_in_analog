"""CLI runner for named AI-in-analog analysis build directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .ngspice_runner import NgspiceRun


ANALYSES = ("delay", "offset", "gain")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def run_analysis(
    analysis: str,
    deck: str | Path,
    *,
    execute: bool = False,
    ngspice_bin: str = "ngspice",
    timeout_s: float = 120.0,
) -> dict:
    """Run or describe one named analysis deck in its build directory."""
    if analysis not in ANALYSES:
        raise ValueError(f"unsupported analysis {analysis!r}; choose from {ANALYSES}")
    deck = Path(deck).resolve()
    if execute and "* Analysis profile:" in deck.read_text(encoding="utf-8"):
        raise ValueError(
            f"{deck} is a profile marker, not a runnable SPICE deck; generate a full deck first"
        )
    workdir = PROJECT_DIR / "build" / analysis
    result = NgspiceRun(ngspice_bin, timeout_s).run(deck, workdir, execute=execute)
    if execute and result.get("returncode") != 0:
        raise RuntimeError(
            f"ngspice failed for {analysis} with return code {result.get('returncode')}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Parse runner options and print the resulting command or execution record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=ANALYSES)
    parser.add_argument("deck", type=Path)
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly run ngspice; otherwise only report the command",
    )
    args = parser.parse_args(argv)
    result = run_analysis(
        args.analysis,
        args.deck,
        execute=args.execute,
        ngspice_bin=args.ngspice,
        timeout_s=args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
