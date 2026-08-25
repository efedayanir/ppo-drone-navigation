from __future__ import annotations

import math

import numpy as np

from baselines.base import BaselinePolicy
from env.drone_env import DroneNavEnv


class ObstacleAwarePolicy(BaselinePolicy):
    name = "obstacle_aware"

    def __init__(
        self,
        speed_scale: float = 0.9,
        repulsion_gain: float = 1.25,
        sensor_threshold: float = 0.65,
        side_bias: float = 0.35,
    ):
        self.speed_scale = float(speed_scale)
        self.repulsion_gain = float(repulsion_gain)
        self.sensor_threshold = float(sensor_threshold)
        self.side_bias = float(side_bias)

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

        readings = env._get_sensor_readings()
        repulsion = np.zeros(2, dtype=np.float32)

        for i, reading in enumerate(readings):
            normalized_reading = float(reading)
            if not env.normalize_sensor_readings:
                normalized_reading /= max(env.sensor_max_range, 1e-8)

            if normalized_reading >= self.sensor_threshold:
                continue

            ray_angle = env.heading + (2.0 * math.pi * i / env.num_rays)
            ray_dir = np.array(
                [math.cos(ray_angle), math.sin(ray_angle)],
                dtype=np.float32,
            )

            closeness = (
                self.sensor_threshold - normalized_reading
            ) / max(self.sensor_threshold, 1e-8)
            repulsion -= ray_dir * closeness * self.repulsion_gain

        combined = desired + repulsion

        if float(np.linalg.norm(combined)) < 1e-6:
            combined = np.array(
                [
                    math.cos(env.heading + self.side_bias),
                    math.sin(env.heading + self.side_bias),
                ],
                dtype=np.float32,
            )

        heading = math.atan2(float(combined[1]), float(combined[0]))

        min_reading = float(np.min(readings)) if len(readings) else 1.0
        if not env.normalize_sensor_readings:
            min_reading /= max(env.sensor_max_range, 1e-8)

        obstacle_speed_factor = np.clip(0.35 + min_reading, 0.25, 1.0)
        speed = env.max_speed * self.speed_scale * float(obstacle_speed_factor)

        return np.array([heading, speed], dtype=np.float32)
