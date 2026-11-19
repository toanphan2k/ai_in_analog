"""Lightweight candidate selector for the next W/L combination.

This scaffold does not train a production surrogate or launch a simulation.  It starts
from deterministic Sobol-like Halton candidates and, when data exists, ranks candidates
by a ridge-regression estimate plus a simple uncertainty proxy.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def variable_names(sizing_config: Mapping) -> tuple[str, ...]:
    """Return the configured optimization-vector order."""
    optimization = sizing_config.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise ValueError("sizing config 'optimization' must be a mapping")
    names = optimization.get("variables")
    if not isinstance(names, list) or not names:
        raise ValueError("sizing config must define a non-empty optimization.variables list")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("optimization.variables entries must be strings")
    if len(set(names)) != len(names):
        raise ValueError("optimization.variables must not contain duplicates")
    unknown = set(names) - set(sizing_config.get("parameters", {}))
    if unknown:
        raise ValueError(
            f"optimization variables are not defined as parameters: {', '.join(sorted(unknown))}"
        )
    return tuple(str(name) for name in names)


def halton(index: int, base: int) -> float:
    """Return the radical-inverse value for one Halton sequence dimension."""
    value = 0.0
    factor = 1.0 / base
    while index:
        value += (index % base) * factor
        index //= base
        factor /= base
    return value


def _prime_bases(count: int) -> list[int]:
    """Return the first ``count`` prime numbers for Halton dimensions."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def candidate_pool(
    count: int,
    sizing_config: Mapping,
    seed: int = 0,
) -> np.ndarray:
    """Generate seeded Halton candidates in the normalized sizing domain."""
    if count <= 0:
        raise ValueError("candidate count must be positive")
    dimension = len(variable_names(sizing_config))
    primes = _prime_bases(dimension)
    pool = np.array(
        [[halton(i + 1, base) for base in primes] for i in range(count)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    pool = np.where(rng.random(pool.shape) < 0.5, pool, 1.0 - pool)
    return np.clip(pool, 0.0, 1.0)


def select_next(candidates: np.ndarray, observed_x: np.ndarray | None,
                observed_y: np.ndarray | None, kappa: float = 1.0) -> np.ndarray:
    """Select the lowest ridge-regression estimate with a simple uncertainty bonus."""
    candidates = np.asarray(candidates, dtype=float)
    if observed_x is None or observed_y is None or len(observed_x) == 0:
        return candidates[0]

    observed_x = np.asarray(observed_x, dtype=float)
    observed_y = np.asarray(observed_y, dtype=float)
    design = np.column_stack((np.ones(len(observed_x)), observed_x))
    ridge = 1e-6 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + ridge) @ design.T @ observed_y

    candidate_design = np.column_stack((np.ones(len(candidates)), candidates))
    mean = candidate_design @ beta
    residual = observed_y - design @ beta
    sigma = np.full(len(candidates), max(float(np.sqrt(np.mean(residual ** 2))), 1e-12))
    lcb = mean - kappa * sigma
    return candidates[int(np.argmin(lcb))]
