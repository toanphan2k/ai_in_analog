"""Metric schema for one StrongARM ngspice evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class StrongArmMetrics:
    delay_s: float
    power_w: float
    input_referred_offset_v: float
    kickback_noise_v: float
    output_swing_v: float
    fmax_hz: float
    energy_j: float = float("nan")
    output: Dict[str, float] | None = None

    def as_dict(self) -> Dict[str, float]:
        row = asdict(self)
        row.pop("output", None)
        return row
