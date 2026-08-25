import math
from typing import Iterable

import numpy as np

from env.obstacles import Obstacle


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_b - point_a))


def distance_to_goal(position: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(goal - position))


def point_is_safe(
    point: np.ndarray,
    radius: float,
    width: float,
    height: float,
    obstacles: Iterable[Obstacle],
) -> bool:
    if point[0] - radius < 0.0 or point[0] + radius > width:
        return False
    if point[1] - radius < 0.0 or point[1] + radius > height:
        return False

    for obstacle in obstacles:
        dist = math.hypot(point[0] - obstacle.x, point[1] - obstacle.y)
        if dist < radius + obstacle.radius:
            return False
    return True


def check_collision(
    position: np.ndarray,
    drone_radius: float,
    width: float,
    height: float,
    obstacles: Iterable[Obstacle],
) -> bool:
    x, y = float(position[0]), float(position[1])

    if x - drone_radius <= 0.0 or x + drone_radius >= width:
        return True
    if y - drone_radius <= 0.0 or y + drone_radius >= height:
        return True

    for obstacle in obstacles:
        if math.hypot(x - obstacle.x, y - obstacle.y) <= drone_radius + obstacle.radius:
            return True
    return False


def check_goal(position: np.ndarray, goal: np.ndarray, goal_radius: float) -> bool:
    return distance_to_goal(position, goal) <= goal_radius + 1e-6