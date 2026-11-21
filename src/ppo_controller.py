"""PPO implementation for direct-evaluation NRBO control."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.rl_controller import DirectEvaluationEnv, RLController


class PPOController(RLController):
    """On-policy PPO controller with model checkpoint persistence."""

    def __init__(self, environment: DirectEvaluationEnv, seed: int, timesteps: int = 64):
        super().__init__(environment=environment, seed=seed)
        rollout_steps = min(8, max(2, timesteps))
        self.model = PPO(
            "MlpPolicy",
            environment,
            n_steps=rollout_steps,
            batch_size=rollout_steps,
            gamma=0.95,
            seed=seed,
            verbose=0,
            device="cpu",
        )

    def learn(self, timesteps: int) -> None:
        self.model.learn(total_timesteps=timesteps, progress_bar=False)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def save(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint_dir / "model.zip"))

    def load(self, checkpoint_dir: Path) -> None:
        model_path = checkpoint_dir / "model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint model does not exist: {model_path}")
        self.model = PPO.load(
            str(model_path),
            env=self.environment,
            seed=self.seed,
            device="cpu",
        )
