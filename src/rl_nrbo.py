"""Direct-feedback PPO/SAC controllers for NRBO sizing search."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC


def _next_population(
    population: np.ndarray,
    scores: np.ndarray,
    action: np.ndarray,
    random: np.random.Generator,
) -> np.ndarray:
    """Apply one NRBO generation using continuous RL-selected operator controls."""
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
    policy_path: str | Path | None = None,
) -> list[dict]:
    """Train PPO/SAC on direct circuit evaluations and return best measured designs."""
    algorithm = algorithm.upper()
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if dimension <= 0 or population_size < 4 or top_k <= 0:
        raise ValueError("dimension and top_k must be positive; population must be at least 4")
    if timesteps < 64 or episode_length < 2:
        raise ValueError("timesteps must be at least 64 and episode_length at least 2")

    class DirectEvaluationEnv(gym.Env):
        """Search population environment rewarded by measured SPICE/mock metrics."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(
                low=np.zeros(3, dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32,
            )
            self.archive: dict[tuple[float, ...], dict] = {}
            self.evaluation_cache: dict[tuple[float, ...], tuple[dict, float, bool]] = {}
            self.population = np.empty((population_size, dimension), dtype=float)
            self.scores = np.empty(population_size, dtype=float)
            self.feasible = np.zeros(population_size, dtype=bool)
            self.step_count = 0
            self.best_score = float("inf")
            self.initial_scale = 1.0

        def _evaluate_population(self) -> None:
            scores = []
            feasible = []
            for candidate in self.population:
                key = tuple(np.round(candidate, 10))
                cached = self.evaluation_cache.get(key)
                if cached is None:
                    record = evaluate_candidate(candidate.copy())
                    candidate_score = float(score_candidate(record))
                    candidate_feasible = bool(is_feasible(record))
                    if not np.isfinite(candidate_score):
                        candidate_score = 1e6
                        candidate_feasible = False
                    cached = (dict(record), candidate_score, candidate_feasible)
                    self.evaluation_cache[key] = cached
                record, candidate_score, candidate_feasible = cached
                scores.append(candidate_score)
                feasible.append(candidate_feasible)
                previous = self.archive.get(key)
                if previous is None or candidate_score < previous["score"]:
                    self.archive[key] = {
                        "score": candidate_score,
                        "feasible": candidate_feasible,
                        "record": record,
                    }
            self.scores = np.asarray(scores, dtype=float)
            self.feasible = np.asarray(feasible, dtype=bool)

        def _observation(self) -> np.ndarray:
            diversity = min(float(np.mean(np.std(self.population, axis=0))) * 2.0, 1.0)
            feasible_fraction = float(np.mean(self.feasible))
            return np.asarray(
                [
                    self.step_count / episode_length,
                    np.tanh(self.best_score / self.initial_scale),
                    diversity,
                    feasible_fraction,
                    min(float(np.mean(self.scores)), 1e6) / 1e6,
                ],
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            """Start a search episode with a fresh sizing population."""
            super().reset(seed=seed)
            _ = options
            self.population = self.np_random.random((population_size, dimension))
            self.step_count = 0
            self.best_score = float("inf")
            self._evaluate_population()
            self.best_score = float(np.min(self.scores))
            self.initial_scale = max(abs(self.best_score), 1e-6)
            return self._observation(), {}

        def step(self, action):
            """Apply RL-selected NRBO controls and reward measured improvement."""
            previous_best = self.best_score
            self.population = _next_population(
                self.population,
                self.scores,
                action,
                self.np_random,
            )
            self._evaluate_population()
            self.best_score = min(previous_best, float(np.min(self.scores)))
            improvement = previous_best - self.best_score
            feasible_fraction = float(np.mean(self.feasible))
            reward = float(
                2.0 * np.tanh(improvement / self.initial_scale)
                + 0.05 * feasible_fraction
                - 0.01
            )
            self.step_count += 1
            terminated = self.step_count >= episode_length
            return self._observation(), reward, terminated, False, {}

        def render(self):
            """The headless simulator optimization environment has no renderer."""
            return None

    environment = DirectEvaluationEnv()
    if algorithm == "PPO":
        controller = PPO(
            "MlpPolicy",
            environment,
            n_steps=8,
            batch_size=8,
            gamma=0.95,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    else:
        controller = SAC(
            "MlpPolicy",
            environment,
            learning_starts=16,
            buffer_size=max(1000, timesteps * 2),
            batch_size=16,
            train_freq=1,
            gradient_steps=1,
            gamma=0.95,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    controller.learn(total_timesteps=timesteps, progress_bar=False)
    observation, _ = environment.reset(seed=seed + 1)
    for _ in range(episode_length):
        action, _ = controller.predict(observation, deterministic=True)
        observation, _, terminated, _, _ = environment.step(action)
        if terminated:
            break
    if policy_path is not None:
        controller.save(str(policy_path))

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
