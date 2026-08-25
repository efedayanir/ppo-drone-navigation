from __future__ import annotations

import math

import numpy as np

from baselines.base import BaselinePolicy
from env.drone_env import DroneNavEnv


class GreedyGoalPolicy(BaselinePolicy):
    name = "greedy_goal"

    def __init__(self, speed_scale: float = 1.0):
        self.speed_scale = float(speed_scale)

    def act(
        self,
        env: DroneNavEnv,
        obs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del obs, rng
        goal_vec = env.goal - env.position
        heading = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        speed = env.max_speed * self.speed_scale
        return np.array([heading, speed], dtype=np.float32)
