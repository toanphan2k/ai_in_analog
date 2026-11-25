"""SAC implementation for direct-evaluation NRBO control."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.rl_controller import DirectEvaluationEnv, RLController


class SACController(RLController):
    """Off-policy SAC controller with model and replay-buffer checkpoint persistence."""

    def __init__(self, environment: DirectEvaluationEnv, seed: int, timesteps: int):
        super().__init__(environment=environment, seed=seed)
        batch_size = min(16, max(2, timesteps))
        learning_starts = min(16, max(1, timesteps // 2))
        self.model = SAC(
            "MlpPolicy",
            environment,
            learning_starts=learning_starts,
            buffer_size=max(1000, timesteps * 2),
            batch_size=batch_size,
            train_freq=1,
            gradient_steps=1,
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
        replay_path = checkpoint_dir / "replay_buffer.pkl"
        self.model.save_replay_buffer(str(replay_path))

    def load(self, checkpoint_dir: Path) -> None:
        model_path = checkpoint_dir / "model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"SAC checkpoint model does not exist: {model_path}")
        self.model = SAC.load(
            str(model_path),
            env=self.environment,
            seed=self.seed,
            device="cpu",
        )
        replay_path = checkpoint_dir / "replay_buffer.pkl"
        if replay_path.is_file():
            self.model.load_replay_buffer(str(replay_path))
