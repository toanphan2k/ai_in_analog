"""Build and optionally execute an ngspice batch command.

The default is dry-run mode: this module creates the exact command and work-directory
layout but does not launch ngspice.  Set ``execute=True`` only after a caller explicitly
chooses to run a simulation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


class NgspiceRun:
    def __init__(self, ngspice_bin: str = "ngspice", timeout_s: float = 120.0):
        self.ngspice_bin = ngspice_bin
        self.timeout_s = timeout_s

    def command(self, deck: str | Path, workdir: str | Path) -> list[str]:
        deck = Path(deck)
        workdir = Path(workdir)
        return [self.ngspice_bin, "-b", "-o", str(workdir / "ngspice.log"), str(deck)]

    def run(self, deck: str | Path, workdir: str | Path,
            *, execute: bool = False) -> dict:
        deck = Path(deck)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        command = self.command(deck, workdir)
        result = {
            "executed": False,
            "command": command,
            "cwd": str(workdir),
            "deck": str(deck),
        }
        if not execute:
            return result
        proc = subprocess.run(
            command,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
            text=True,
        )
        result.update(
            executed=True,
            returncode=proc.returncode,
            stdout=proc.stdout,
        )
        return result
