import numpy as np
import pytest

from env.sampling import sample_start_and_goal


def test_sample_start_and_goal_returns_valid_points():
    rng = np.random.default_rng(123)

    def point_is_safe(point, radius):
        return True

    config = {
        "safe_spawn_margin": 0.8,
        "safe_goal_margin": 0.8,
        "min_start_goal_distance": 2.0,
    }

    start, goal = sample_start_and_goal(
        rng=rng,
        width=10.0,
        height=10.0,
        drone_radius=0.2,
        goal_radius=0.3,
        config=config,
        point_is_safe_fn=point_is_safe,
    )

    assert isinstance(start, np.ndarray)
    assert isinstance(goal, np.ndarray)
    assert start.shape == (2,)
    assert goal.shape == (2,)
    assert np.linalg.norm(goal - start) >= config["min_start_goal_distance"]


def test_sample_start_and_goal_respects_point_is_safe():
    rng = np.random.default_rng(123)

    def point_is_safe(point, radius):
        return point[0] > 5.0

    config = {
        "safe_spawn_margin": 0.8,
        "safe_goal_margin": 0.8,
        "min_start_goal_distance": 2.0,
    }

    start, goal = sample_start_and_goal(
        rng=rng,
        width=10.0,
        height=10.0,
        drone_radius=0.2,
        goal_radius=0.3,
        config=config,
        point_is_safe_fn=point_is_safe,
    )

    assert start[0] > 5.0
    assert goal[0] > 5.0


def test_sample_start_and_goal_raises_when_no_valid_positions():
    rng = np.random.default_rng(123)

    def point_is_safe(point, radius):
        return False

    config = {
        "safe_spawn_margin": 0.8,
        "safe_goal_margin": 0.8,
        "min_start_goal_distance": 2.0,
    }

    with pytest.raises(RuntimeError):
        sample_start_and_goal(
            rng=rng,
            width=10.0,
            height=10.0,
            drone_radius=0.2,
            goal_radius=0.3,
            config=config,
            point_is_safe_fn=point_is_safe,
        )