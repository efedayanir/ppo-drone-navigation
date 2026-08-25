import math

import numpy as np

from env.drone_env import DroneNavEnv
from tests.test_utils import BASE_CONFIG, make_env


def test_step_returns_gymnasium_five_tuple():
    env = make_env()
    env.reset(seed=123)

    result = env.step(np.array([0.0, 0.5], dtype=np.float32))

    assert len(result) == 5

    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_increments_step_count():
    env = make_env()
    env.reset(seed=123)

    assert env.step_count == 0
    env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert env.step_count == 1


def test_step_updates_prev_action_and_last_action():
    env = make_env()
    env.reset(seed=123)

    action = np.array([0.25, 0.6], dtype=np.float32)
    env.step(action)

    assert np.allclose(env.prev_action, action)
    assert np.allclose(env.last_action, action)


def test_step_within_episode_keeps_step_count_consistent():
    env = make_env()
    env.reset(seed=123)

    env.step(np.array([0.0, 0.5], dtype=np.float32))
    env.step(np.array([0.1, 0.6], dtype=np.float32))

    assert env.step_count == 2


def test_zero_speed_action_keeps_motion_small_from_rest():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    old_position = env.position.copy()
    env.step(np.array([0.0, 0.0], dtype=np.float32))

    displacement = np.linalg.norm(env.position - old_position)
    assert displacement < 1e-3


def test_positive_speed_action_changes_position():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    old_position = env.position.copy()
    env.step(np.array([0.0, 1.0], dtype=np.float32))

    displacement = np.linalg.norm(env.position - old_position)
    assert displacement > 0.0


def test_heading_moves_toward_absolute_heading_target():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.velocity = np.zeros(2, dtype=np.float32)
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    target_heading = math.pi / 2.0
    env.step(np.array([target_heading, 0.0], dtype=np.float32))

    assert env.heading > 0.0


def test_action_is_clipped_when_out_of_bounds():
    env = make_env()
    env.reset(seed=123)

    big_action = np.array([10.0, 10.0], dtype=np.float32)
    env.step(big_action)

    assert env.prev_action[0] <= math.pi + 1e-6
    assert env.prev_action[0] >= -math.pi - 1e-6
    assert env.prev_action[1] <= BASE_CONFIG["environment"]["drone"]["max_speed"] + 1e-6
    assert env.prev_action[1] >= 0.0 - 1e-6


def test_observation_after_step_matches_observation_space_shape():
    env = make_env()
    env.reset(seed=123)

    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert obs.shape == env.observation_space.shape


def test_observation_after_step_is_float32():
    env = make_env()
    env.reset(seed=123)

    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert obs.dtype == np.float32


def test_observation_after_step_is_bounded_by_space():
    env = make_env()
    env.reset(seed=123)

    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert env.observation_space.contains(obs)


def test_reward_is_finite():
    env = make_env()
    env.reset(seed=123)

    _, reward, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert np.isfinite(reward)


def test_timeout_sets_truncated_true():
    config = dict(BASE_CONFIG)
    config["environment"] = dict(BASE_CONFIG["environment"])
    config["environment"]["episode"] = dict(BASE_CONFIG["environment"]["episode"])
    config["environment"]["episode"]["max_steps"] = 1

    env = DroneNavEnv(config)
    env.reset(seed=123)

    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is True


def test_step_updates_prev_distance_to_goal():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([2.0, 2.0], dtype=np.float32)
    env.goal = np.array([8.0, 2.0], dtype=np.float32)
    env.heading = 0.0
    env.velocity = np.zeros(2, dtype=np.float32)
    env.angular_velocity = 0.0
    env.prev_distance_to_goal = np.linalg.norm(env.goal - env.position)
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    env.step(np.array([0.0, 1.0], dtype=np.float32))

    actual_distance = np.linalg.norm(env.goal - env.position)
    assert np.isclose(env.prev_distance_to_goal, actual_distance)


def test_info_contains_expected_keys_after_step():
    env = make_env()
    env.reset(seed=123)

    _, _, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))

    expected_keys = {
        "collision",
        "reached_goal",
        "truncated",
        "stuck",
        "reward_terms",
    }

    assert expected_keys.issubset(info.keys())


def test_reward_terms_exist_and_total_matches_reward():
    env = make_env()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))

    reward_terms = info["reward_terms"]
    assert isinstance(reward_terms, dict)

    expected_reward_keys = {
        "collision",
        "goal",
        "progress",
        "step",
        "smoothness",
        "stall",
        "alignment",
        "timeout",
        "low_speed",
        "stuck",
        "total",
    }
    assert expected_reward_keys.issubset(reward_terms.keys())
    assert np.isclose(reward_terms["total"], reward)


def test_speed_does_not_exceed_max_speed():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([9.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    for _ in range(20):
        env.step(np.array([0.0, 1.5], dtype=np.float32))

    speed = np.linalg.norm(env.velocity)
    assert speed <= BASE_CONFIG["environment"]["drone"]["max_speed"] + 1e-6


def test_speed_normalized_stays_in_valid_range():
    env = make_env()
    env.reset(seed=123)

    _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

    # Eğer info'da speed_normalized yoksa env tasarımına göre bu testi kaldır.
    if "speed_normalized" in info:
        assert 0.0 <= info["speed_normalized"] <= 1.0 + 1e-6