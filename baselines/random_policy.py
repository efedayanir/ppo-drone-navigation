from __future__ import annotations

import numpy as np

from baselines.base import BaselinePolicy
from env.drone_env import DroneNavEnv


class RandomPolicy(BaselinePolicy):
    name = "random"

    def act(
        self,
        env: DroneNavEnv,
        obs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del obs
        return np.array(
            [
                rng.uniform(env.action_space.low[0], env.action_space.high[0]),
                rng.uniform(env.action_space.low[1], env.action_space.high[1]),
            ],
            dtype=np.float32,
        )
