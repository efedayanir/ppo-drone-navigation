import numpy as np


def sample_start_and_goal(
    rng,
    width,
    height,
    drone_radius,
    goal_radius,
    config,
    point_is_safe_fn,
):
    cfg = config or {}

    max_tries = int(cfg.get("max_sampling_tries", 100))
    spawn_margin = float(cfg.get("safe_spawn_margin", drone_radius))
    goal_margin = float(cfg.get("safe_goal_margin", goal_radius))
    min_dist = float(cfg.get("min_start_goal_distance", 2.0))

    last_start = None
    last_goal = None

    for _ in range(max_tries):
        start = rng.uniform(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([width, height], dtype=np.float32),
        )
        goal = rng.uniform(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([width, height], dtype=np.float32),
        )

        last_start = start
        last_goal = goal

        if not point_is_safe_fn(start, drone_radius + spawn_margin):
            continue

        if not point_is_safe_fn(goal, goal_radius + goal_margin):
            continue

        if np.linalg.norm(start - goal) < min_dist:
            continue

        return start.astype(np.float32), goal.astype(np.float32)

    raise RuntimeError(
        f"Failed to sample valid start/goal after {max_tries} attempts"
    )