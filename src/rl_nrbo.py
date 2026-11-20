"""NRBO orchestration for PPO/SAC direct-evaluation control."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from src.ppo_controller import PPOController
from src.rl_controller import DirectEvaluationEnv
from src.sac_controller import SACController


def _next_population(
    population: np.ndarray,
    scores: np.ndarray,
    action: np.ndarray,
    random: np.random.Generator,
) -> np.ndarray:
    """Apply one NRBO generation using RL-selected continuous controls."""
    controls = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
    restart_probability = 0.02 + 0.48 * controls[0]
    newton_weight = controls[1]
    step_limit = 0.02 + 0.38 * controls[2]

    population = population[np.argsort(scores)]
    best = population[0].copy()
    dimension = population.shape[1]
    elite = population[: max(2, len(population) // 5)]
    next_population = [best]

    while len(next_population) < len(population):
        current = population[len(next_population)].copy()
        if random.random() < restart_probability:
            proposal = random.random(dimension)
        else:
            first, second = random.choice(len(population), size=2, replace=False)
            guide = elite[random.integers(len(elite))]
            newton_move = current + random.random() * (best - current)
            population_move = (
                current
                + random.random() * (guide - current)
                + 0.2 * (population[first] - population[second])
            )
            direction = (
                newton_weight * (newton_move - current)
                + (1.0 - newton_weight) * (population_move - current)
            )
            proposal = current + np.clip(direction, -step_limit, step_limit)
        next_population.append(np.clip(proposal, 0.0, 1.0))
    return np.asarray(next_population)


def optimize_with_rl_nrbo(
    *,
    algorithm: str,
    dimension: int,
    population_size: int,
    top_k: int,
    timesteps: int,
    episode_length: int,
    seed: int,
    evaluate_candidate: Callable[[np.ndarray], Mapping],
    score_candidate: Callable[[Mapping], float],
    is_feasible: Callable[[Mapping], bool],
    checkpoint_dir: str | Path,
    resume_model: str | Path | None = None,
) -> list[dict]:
    """Train/load PPO or SAC and return directly measured NRBO candidates."""
    algorithm = algorithm.upper()
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if dimension <= 0 or population_size < 4 or top_k <= 0:
        raise ValueError("dimension and top_k must be positive; population must be at least 4")
    if timesteps < 64 or episode_length < 2:
        raise ValueError("timesteps must be at least 64 and episode_length at least 2")

    environment = DirectEvaluationEnv(
        dimension=dimension,
        population_size=population_size,
        episode_length=episode_length,
        seed=seed,
        evaluate_candidate=evaluate_candidate,
        score_candidate=score_candidate,
        is_feasible=is_feasible,
        next_population=_next_population,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "PPO":
        controller = PPOController(environment, seed)
    else:
        controller = SACController(environment, seed, timesteps)

    if resume_model is not None:
        controller.load(Path(resume_model).parent)

    controller.learn(timesteps)
    observation, _ = environment.reset(seed=seed + 1)
    for _ in range(episode_length):
        action = controller.predict(observation)
        observation, _, terminated, _, _ = environment.step(action)
        if terminated:
            break
    controller.save(checkpoint_dir)

    ranked = sorted(
        environment.archive.values(),
        key=lambda item: (not item["feasible"], item["score"]),
    )
    results = []
    seen = set()
    for item in ranked:
        record = item["record"]
        size = record.get("size", {})
        signature = tuple(sorted(size.items()))
        if signature in seen:
            continue
        seen.add(signature)
        results.append(
            {
                "sample": record.get("name"),
                "size": size,
                "metrics": record.get("metrics", {}),
                "score": item["score"],
                "feasible": item["feasible"],
                "status": record.get("status"),
                "error": record.get("error", ""),
            }
        )
        if len(results) >= top_k:
            break
    return results
