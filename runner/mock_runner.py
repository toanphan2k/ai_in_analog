"""Deterministic, sizing-dependent mock measurements for pipeline smoke tests."""
from __future__ import annotations

from typing import Mapping

import numpy as np


class MockRunner:
    """Produce repeatable synthetic circuit metrics without launching SPICE."""

    def __init__(self, *, seed: int = 0, failure_rate: float = 0.03):
        if not 0.0 <= failure_rate < 1.0:
            raise ValueError("failure_rate must be in [0, 1)")
        self.seed = seed
        self.failure_rate = failure_rate

    def evaluate(
        self,
        sizes: Mapping[str, float],
        normalized: np.ndarray,
        *,
        sample_index: int,
        vdd_v: float,
    ) -> dict:
        """Return one mock result record using a smooth sizing-dependent response."""
        rng = np.random.default_rng(self.seed + sample_index * 104729)
        vector = np.asarray(normalized, dtype=float)
        axes = np.full(8, 0.5)
        axes[: min(len(vector), len(axes))] = vector[: len(axes)]
        jitter = rng.normal(0.0, 0.025)
        log_delay = (
            -0.22
            - 0.62 * axes[0]
            + 0.24 * axes[1]
            - 0.28 * axes[2]
            - 0.37 * axes[3]
            + 0.11 * axes[4]
            - 0.31 * axes[5]
            + 0.08 * axes[6]
            + 0.17 * axes[7]
            + jitter
        )
        delay_s = 4.5e-10 * float(np.exp(log_delay))
        reset_time_s = 2.8e-10 * float(np.exp(-0.9 * axes[4] + 0.25 * axes[0] + 0.5 * jitter))
        width_terms = [
            float(sizes[name])
            for name in ("w_in", "w_n", "w_p", "w_rst", "w_tail", "w_8")
            if name in sizes
        ]
        width_load = sum(width_terms) / max(len(width_terms), 1)
        power_w = (
            1.2e-3
            + 1.2e-3 * width_load
            + 1.4e-3
            * (0.25 * axes[0] + 0.2 * axes[2] + 0.3 * axes[3] + 0.25 * axes[5])
        ) * float(np.exp(rng.normal(0.0, 0.04)))
        output_swing_v = min(
            0.98 * vdd_v,
            vdd_v * (0.72 + 0.18 * axes[0] - 0.08 * axes[4]),
        )
        gain = 8.0 + 14.0 * axes[2] + 18.0 * axes[3] + 5.0 * axes[7]
        offset_v = rng.normal(0.0, 2.5e-3) * (1.0 + 0.15 * axes[4])
        input_referred_noise_v = 1.1e-3 * (1.0 + 0.35 * (1.0 - axes[0]))
        kickback_noise_v = 0.8e-3 * (1.0 + 0.25 * axes[5])
        failed = rng.random() < self.failure_rate
        comparison_period_s = 5.0e-10
        energy_j = power_w * comparison_period_s
        speed_hz = 1.0 / (2.0 * max(delay_s, reset_time_s))
        voh_v = min(vdd_v, 0.5 * vdd_v + 0.5 * output_swing_v)
        vol_v = max(0.0, voh_v - output_swing_v)
        overshoot_v = vdd_v * (0.04 + 0.08 * axes[4])
        metrics = {
            "delay_s": delay_s,
            "delay_mean_s": delay_s * (1.0 + abs(jitter)),
            "reset_time_s": reset_time_s,
            "reset_time_mean_s": reset_time_s * (1.0 + abs(jitter) * 0.5),
            "speed_hz": speed_hz,
            "cycles_measured": 32,
            "power_w": power_w,
            "energy_j": energy_j,
            "energy_per_comparison_j": energy_j,
            "input_referred_offset_v": abs(offset_v),
            "offset_v": abs(offset_v),
            "input_referred_noise_v": input_referred_noise_v,
            "output_referred_noise_v": input_referred_noise_v * gain,
            "kickback_noise_v": kickback_noise_v,
            "kickback_rms_v": kickback_noise_v * 0.25,
            "output_swing_v": output_swing_v,
            "voh_v": voh_v,
            "vol_v": vol_v,
            "overshoot_v": overshoot_v,
            "dc_gain_v_per_v": gain,
            "peak_gain_v_per_v": gain * 1.08,
            "peak_gain_frequency_hz": 1.0 / (2.0 * np.pi * delay_s),
            "bandwidth_3db_hz": 1.0 / (2.0 * np.pi * delay_s * 1.4),
            "ac_freq_points": 276,
        }
        failure_info = (
            {
                "reason": "no_decision",
                "cycles_completed": 0,
                "total_cycles": 32,
                "peak_diff_v": min(0.49 * vdd_v, 0.5 * output_swing_v),
                "decision_threshold_v": 0.5 * vdd_v,
                "reset_min_v": None,
                "reset_threshold_v": None,
            }
            if failed
            else None
        )
        return {
            "status": "failed" if failed else "evaluated",
            "metrics": {} if failed else metrics,
            "failure_info": failure_info,
            "error": "mock metastability/failure" if failed else "",
            "files": {"runner": "mock", "seed": self.seed + sample_index * 104729},
        }
