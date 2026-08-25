from __future__ import annotations

import math

import numpy as np

from baselines.base import BaselinePolicy
from env.drone_env import DroneNavEnv


class WallAvoidingGreedyPolicy(BaselinePolicy):
    name = "wall_avoiding_greedy"

    def __init__(self, speed_scale: float = 0.9, wall_margin: float = 0.8):
        self.speed_scale = float(speed_scale)
        self.wall_margin = float(wall_margin)

    def act(
        self,
        env: DroneNavEnv,
        obs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del obs, rng
        goal_vec = env.goal - env.position
        goal_norm = float(np.linalg.norm(goal_vec))

        if goal_norm > 1e-8:
            desired = goal_vec / goal_norm
        else:
            desired = np.array(
                [math.cos(env.heading), math.sin(env.heading)],
                dtype=np.float32,
            )

        wall_push = np.zeros(2, dtype=np.float32)
        x, y = float(env.position[0]), float(env.position[1])

        if x < self.wall_margin:
            wall_push[0] += (self.wall_margin - x) / self.wall_margin
        if env.width - x < self.wall_margin:
            wall_push[0] -= (self.wall_margin - (env.width - x)) / self.wall_margin
        if y < self.wall_margin:
            wall_push[1] += (self.wall_margin - y) / self.wall_margin
        if env.height - y < self.wall_margin:
            wall_push[1] -= (self.wall_margin - (env.height - y)) / self.wall_margin

        combined = desired + 1.5 * wall_push
        heading = math.atan2(float(combined[1]), float(combined[0]))
        speed = env.max_speed * self.speed_scale

        return np.array([heading, speed], dtype=np.float32)
