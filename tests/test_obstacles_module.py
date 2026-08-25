import numpy as np

from env.obstacles import generate_obstacles


def test_generate_obstacles_returns_expected_count():
    rng = np.random.default_rng(123)
    config = {
        "count_min": 5,
        "count_max": 5,
        "radius_min": 0.3,
        "radius_max": 0.8,
        "safe_obstacle_margin": 0.2,
    }

    obstacles = generate_obstacles(
        rng=rng,
        width=10.0,
        height=10.0,
        config=config,
    )

    assert len(obstacles) == 5


def test_generate_obstacles_stay_within_world_bounds():
    rng = np.random.default_rng(123)
    config = {
        "count_min": 5,
        "count_max": 5,
        "radius_min": 0.3,
        "radius_max": 0.8,
        "safe_obstacle_margin": 0.2,
    }

    obstacles = generate_obstacles(
        rng=rng,
        width=10.0,
        height=10.0,
        config=config,
    )

    for obs in obstacles:
        assert obs.x - obs.radius >= 0.0
        assert obs.x + obs.radius <= 10.0
        assert obs.y - obs.radius >= 0.0
        assert obs.y + obs.radius <= 10.0


def test_generate_obstacles_are_non_overlapping_with_margin():
    rng = np.random.default_rng(123)
    config = {
        "count_min": 5,
        "count_max": 5,
        "radius_min": 0.3,
        "radius_max": 0.8,
        "safe_obstacle_margin": 0.2,
    }

    obstacles = generate_obstacles(
        rng=rng,
        width=10.0,
        height=10.0,
        config=config,
    )

    margin = config["safe_obstacle_margin"]
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            oi = obstacles[i]
            oj = obstacles[j]
            dist = np.hypot(oi.x - oj.x, oi.y - oj.y)
            assert dist >= oi.radius + oj.radius + margin


def test_generate_obstacles_respects_disabled_flag():
    rng = np.random.default_rng(123)
    config = {
        "enabled": False,
        "count_min": 5,
        "count_max": 5,
        "radius_min": 0.3,
        "radius_max": 0.8,
        "safe_obstacle_margin": 0.2,
    }

    obstacles = generate_obstacles(
        rng=rng,
        width=10.0,
        height=10.0,
        config=config,
    )

    assert obstacles == []
