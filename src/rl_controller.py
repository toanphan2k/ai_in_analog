"""Common interface and direct-evaluation environment for RL-NRBO controllers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DirectEvaluationEnv(gym.Env):
    """NRBO population environment rewarded by direct simulator/mock measurements."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dimension: int,
        population_size: int,
        episode_length: int,
        seed: int,
        evaluate_candidate: Callable[[np.ndarray, int], Mapping],
        score_candidate: Callable[[Mapping], float],
        is_feasible: Callable[[Mapping], bool],
        next_population: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.random.Generator],
            np.ndarray,
        ],
        initial_iteration: int = 0,
    ):
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
        self.dimension = dimension
        self.population_size = population_size
        self.episode_length = episode_length
        self.evaluate_candidate = evaluate_candidate
        self.score_candidate = score_candidate
        self.is_feasible = is_feasible
        self.next_population = next_population
        self.rng = np.random.default_rng(seed)
        self.archive: dict[tuple[float, ...], dict] = {}
        self.evaluation_cache: dict[tuple[float, ...], tuple[dict, float, bool]] = {}
        self.population = np.empty((population_size, dimension), dtype=float)
        self.scores = np.empty(population_size, dtype=float)
        self.feasible = np.zeros(population_size, dtype=bool)
        self.step_count = 0
        self.iteration = initial_iteration
        self.best_score = float("inf")
        self.initial_scale = 1.0

    def _evaluate_population(self) -> None:
        scores = []
        feasible = []
        for candidate in self.population:
            key = tuple(np.round(candidate, 10))
            cached = self.evaluation_cache.get(key)
            if cached is None:
                record = self.evaluate_candidate(candidate.copy(), self.iteration)
                candidate_score = float(self.score_candidate(record))
                candidate_feasible = bool(self.is_feasible(record))
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
                self.step_count / self.episode_length,
                np.tanh(self.best_score / self.initial_scale),
                diversity,
                feasible_fraction,
                min(float(np.mean(self.scores)), 1e6) / 1e6,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        """Start a new search episode with a fresh candidate population."""
        super().reset(seed=seed)
        _ = options
        self.population = self.np_random.random((self.population_size, self.dimension))
        self.step_count = 0
        self.iteration += 1
        self.best_score = float("inf")
        self._evaluate_population()
        self.best_score = float(np.min(self.scores))
        self.initial_scale = max(abs(self.best_score), 1e-6)
        return self._observation(), {}

    def step(self, action):
        """Apply RL-selected NRBO controls and reward direct improvement."""
        previous_best = self.best_score
        self.population = self.next_population(
            self.population,
            self.scores,
            action,
            self.np_random,
        )
        self.iteration += 1
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
        terminated = self.step_count >= self.episode_length
        return self._observation(), reward, terminated, False, {}

    def render(self):
        """This headless search environment has no visual renderer."""
        return None


class RLController:
    """Common interface for PPO and SAC checkpoint behavior."""

    def __init__(
        self,
        *,
        environment: DirectEvaluationEnv,
        seed: int,
    ):
        self.environment = environment
        self.seed = seed

    def learn(self, timesteps: int) -> None:
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, checkpoint_dir: Path) -> None:
        raise NotImplementedError

    def load(self, checkpoint_dir: Path) -> None:
        raise NotImplementedError
