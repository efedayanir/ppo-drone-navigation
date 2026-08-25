from __future__ import annotations

import numpy as np
import yaml

from baselines.greedy_goal import GreedyGoalPolicy
from baselines.obstacle_aware import ObstacleAwarePolicy
from baselines.random_policy import RandomPolicy
from baselines.wall_avoiding import WallAvoidingGreedyPolicy
from env.drone_env import DroneNavEnv


CONFIG_PATH = "configs/ablations/lidar_8.yaml"


def load_test_env() -> DroneNavEnv:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    env = DroneNavEnv(config)
    env.reset(seed=43)
    return env


def assert_valid_action(env: DroneNavEnv, action: np.ndarray) -> None:
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert env.action_space.contains(action)


def test_random_policy_returns_valid_action() -> None:
    env = load_test_env()
    rng = np.random.default_rng(43)

    try:
        obs, _ = env.reset(seed=43)
        action = RandomPolicy().act(env, obs, rng)
        assert_valid_action(env, action)
    finally:
        env.close()


def test_greedy_goal_policy_returns_valid_action() -> None:
    env = load_test_env()
    rng = np.random.default_rng(43)

    try:
        obs, _ = env.reset(seed=43)
        action = GreedyGoalPolicy().act(env, obs, rng)
        assert_valid_action(env, action)
    finally:
        env.close()


def test_obstacle_aware_policy_returns_valid_action() -> None:
    env = load_test_env()
    rng = np.random.default_rng(43)

    try:
        obs, _ = env.reset(seed=43)
        action = ObstacleAwarePolicy().act(env, obs, rng)
        assert_valid_action(env, action)
    finally:
        env.close()


def test_wall_avoiding_policy_returns_valid_action() -> None:
    env = load_test_env()
    rng = np.random.default_rng(43)

    try:
        obs, _ = env.reset(seed=43)
        action = WallAvoidingGreedyPolicy().act(env, obs, rng)
        assert_valid_action(env, action)
    finally:
        env.close()


def test_each_policy_can_complete_one_environment_step() -> None:
    policies = [
        RandomPolicy(),
        GreedyGoalPolicy(),
        ObstacleAwarePolicy(),
        WallAvoidingGreedyPolicy(),
    ]

    for policy in policies:
        env = load_test_env()
        rng = np.random.default_rng(43)

        try:
            obs, _ = env.reset(seed=43)
            action = policy.act(env, obs, rng)

            next_obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(next_obs)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        finally:
            env.close()