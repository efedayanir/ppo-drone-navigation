import numpy as np
import pytest

from env.drone_env import DroneNavEnv
from env.obstacles import Obstacle

from tests.test_utils import BASE_CONFIG, make_env


def test_point_is_safe_rejects_wall_overlap():
    env = make_env()
    env.reset(seed=123)
    env.obstacles = []

    point = np.array([0.1, 5.0], dtype=np.float32)
    assert env._point_is_safe(point, radius=0.2) is False


def test_point_is_safe_rejects_obstacle_overlap():
    env = make_env()
    env.reset(seed=123)
    env.obstacles = [Obstacle(x=5.0, y=5.0, radius=0.5)]

    point = np.array([5.3, 5.0], dtype=np.float32)
    assert env._point_is_safe(point, radius=0.2) is False


def test_point_is_safe_accepts_clear_point():
    env = make_env()
    env.reset(seed=123)
    env.obstacles = [Obstacle(x=8.0, y=8.0, radius=0.5)]

    point = np.array([2.0, 2.0], dtype=np.float32)
    assert env._point_is_safe(point, radius=0.2) is True


def test_generated_obstacles_are_within_world_and_non_overlapping():
    env = make_env()
    env.reset(seed=123)

    for obs in env.obstacles:
        assert obs.radius >= BASE_CONFIG["environment"]["obstacles"]["radius_min"]
        assert obs.radius <= BASE_CONFIG["environment"]["obstacles"]["radius_max"]
        assert obs.x - obs.radius >= 0.0
        assert obs.x + obs.radius <= BASE_CONFIG["environment"]["world"]["width"]
        assert obs.y - obs.radius >= 0.0
        assert obs.y + obs.radius <= BASE_CONFIG["environment"]["world"]["height"]

    margin = BASE_CONFIG["environment"]["obstacles"]["safe_obstacle_margin"]
    for i in range(len(env.obstacles)):
        for j in range(i + 1, len(env.obstacles)):
            oi = env.obstacles[i]
            oj = env.obstacles[j]
            dist = np.hypot(oi.x - oj.x, oi.y - oj.y)
            assert dist >= oi.radius + oj.radius + margin


def test_sampled_start_and_goal_are_safe_from_obstacles():
    env = make_env()
    env.reset(seed=123)

    obs_cfg = BASE_CONFIG["environment"]["obstacles"]
    spawn_margin = obs_cfg["safe_spawn_margin"]
    goal_margin = obs_cfg["safe_goal_margin"]

    assert env._point_is_safe(env.position, env.drone_radius + spawn_margin)
    assert env._point_is_safe(env.goal, env.goal_radius + goal_margin)


def test_sample_start_and_goal_raises_when_no_valid_positions():
    config = dict(BASE_CONFIG)
    config["environment"] = dict(BASE_CONFIG["environment"])
    config["environment"]["world"] = {"width": 1.0, "height": 1.0}
    config["environment"]["goal"] = dict(BASE_CONFIG["environment"]["goal"])
    config["environment"]["goal"]["min_start_goal_distance"] = 10.0
    config["environment"]["obstacles"] = dict(BASE_CONFIG["environment"]["obstacles"])
    config["environment"]["obstacles"]["count_min"] = 0
    config["environment"]["obstacles"]["count_max"] = 0

    env = DroneNavEnv(config)
    env.np_random = np.random.default_rng(123)

    with pytest.raises(RuntimeError):
        env._sample_start_and_goal()


def test_wrap_angle_keeps_result_in_pi_range():
    wrapped = DroneNavEnv._wrap_angle(4 * np.pi)
    assert -np.pi <= wrapped < np.pi


def test_wrap_angle_wraps_positive_and_negative_angles():
    assert np.isclose(DroneNavEnv._wrap_angle(np.pi + 0.1), -np.pi + 0.1, atol=1e-6)
    assert np.isclose(DroneNavEnv._wrap_angle(-np.pi - 0.1), np.pi - 0.1, atol=1e-6)