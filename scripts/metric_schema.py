"""Metric schema for one StrongARM ngspice evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class StrongArmMetrics:
    """Measured StrongARM performance metrics in SI units."""

    delay_s: float
    power_w: float
    input_referred_offset_v: float
    input_referred_noise_v: float
    kickback_noise_v: float
    output_swing_v: float
    fmax_hz: float
    dc_gain_v_per_v: float = float("nan")
    energy_j: float = float("nan")
    output: Dict[str, float] | None = None

    def as_dict(self) -> Dict[str, float]:
        """Return serializable scalar metrics, excluding optional raw output data."""
        row = asdict(self)
        row.pop("output", None)
        return row
