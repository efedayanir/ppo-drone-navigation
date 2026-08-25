import math
from dataclasses import dataclass
from typing import List


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


def generate_obstacles(
    rng,
    width: float,
    height: float,
    config,
) -> List[Obstacle]:
    if not bool(config.get("enabled", True)):
        return []

    count = int(rng.integers(config["count_min"], config["count_max"] + 1))
    radius_min = float(config["radius_min"])
    radius_max = float(config["radius_max"])
    margin = float(config["safe_obstacle_margin"])

    obstacles: List[Obstacle] = []
    attempts = 0
    max_attempts = 1000

    while len(obstacles) < count and attempts < max_attempts:
        attempts += 1
        radius = float(rng.uniform(radius_min, radius_max))
        x = float(rng.uniform(radius, width - radius))
        y = float(rng.uniform(radius, height - radius))
        candidate = Obstacle(x=x, y=y, radius=radius)

        valid = True
        for existing in obstacles:
            dist = math.hypot(candidate.x - existing.x, candidate.y - existing.y)
            if dist < candidate.radius + existing.radius + margin:
                valid = False
                break

        if valid:
            obstacles.append(candidate)

    return obstacles
