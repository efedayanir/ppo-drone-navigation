from __future__ import annotations

import math

import numpy as np

from env.drone_env import DroneNavEnv
from env.reward import compute_reward_with_info
from tests.test_utils import BASE_CONFIG, make_env


def clone_config() -> dict:
    config = dict(BASE_CONFIG)
    config["environment"] = dict(BASE_CONFIG["environment"])
    config["environment"]["episode"] = dict(BASE_CONFIG["environment"]["episode"])
    config["environment"]["obstacles"] = dict(BASE_CONFIG["environment"]["obstacles"])
    config["environment"]["goal"] = dict(BASE_CONFIG["environment"]["goal"])
    config["reward"] = dict(BASE_CONFIG["reward"])
    config["reward"]["shaping"] = dict(BASE_CONFIG["reward"]["shaping"])
    return config


def make_public_scenario_env(
    position: list[float],
    goal: list[float],
    heading: float = 0.0,
) -> DroneNavEnv:
    config = clone_config()
    config["environment"]["obstacles"]["enabled"] = False
    config["environment"]["episode"]["max_steps"] = 200
    config["reward"]["stuck_patience"] = 200

    env = DroneNavEnv(config)
    env.reset(
        seed=123,
        options={
            "position": position,
            "goal": goal,
            "heading": heading,
            "obstacles": [],
        },
    )
    return env


def rollout_total_reward(env: DroneNavEnv, actions: list[list[float]]) -> tuple[float, dict]:
    total_reward = 0.0
    info: dict = {}
    for action in actions:
        _, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, info


def test_timeout_penalty_is_applied_when_episode_is_truncated():
    config = clone_config()
    config["reward"]["timeout_penalty"] = 40.0

    reward, terms = compute_reward_with_info(
        action=np.array([0.0, 0.5], dtype=np.float32),
        velocity=np.array([0.1, 0.0], dtype=np.float32),
        prev_action=np.array([0.0, 0.5], dtype=np.float32),
        prev_distance_to_goal=5.0,
        current_distance=5.0,
        collision=False,
        reached_goal=False,
        truncated=True,
        config=config["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert terms["timeout"] == -40.0
    assert reward == terms["total"]
    assert reward < 0.0


def test_low_speed_penalty_applies_far_from_goal():
    config = clone_config()
    config["reward"]["low_speed_penalty"] = 0.35
    config["reward"]["shaping"]["stall_distance_gate"] = 0.5
    config["reward"]["shaping"]["min_speed_reward_gate"] = 0.10

    reward, terms = compute_reward_with_info(
        action=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        prev_action=np.array([0.0, 0.0], dtype=np.float32),
        prev_distance_to_goal=5.0,
        current_distance=5.0,
        collision=False,
        reached_goal=False,
        truncated=False,
        config=config["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert terms["low_speed"] == -0.35
    assert reward < 0.0


def test_stall_penalty_applies_only_when_far_enough_from_goal():
    config = clone_config()
    config["reward"]["stall_penalty"] = 1.5
    config["reward"]["shaping"]["stall_distance_gate"] = 0.5
    config["reward"]["shaping"]["stall_progress_threshold"] = 1e-3
    config["reward"]["shaping"]["stall_speed_threshold"] = 0.05

    _, far_terms = compute_reward_with_info(
        action=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        prev_action=np.array([0.0, 0.0], dtype=np.float32),
        prev_distance_to_goal=5.0,
        current_distance=5.0,
        collision=False,
        reached_goal=False,
        truncated=False,
        config=config["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    _, near_terms = compute_reward_with_info(
        action=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        prev_action=np.array([0.0, 0.0], dtype=np.float32),
        prev_distance_to_goal=0.2,
        current_distance=0.2,
        collision=False,
        reached_goal=False,
        truncated=False,
        config=config["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert far_terms["stall"] == -1.5
    assert near_terms["stall"] == 0.0


def test_no_goal_bonus_when_collision_and_goal_happen_together():
    config = clone_config()

    reward, terms = compute_reward_with_info(
        action=np.array([0.0, 0.5], dtype=np.float32),
        velocity=np.array([0.1, 0.0], dtype=np.float32),
        prev_action=np.array([0.0, 0.5], dtype=np.float32),
        prev_distance_to_goal=0.2,
        current_distance=0.1,
        collision=True,
        reached_goal=True,
        truncated=False,
        config=config["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert terms["collision"] == -config["reward"]["collision_penalty"]
    assert terms["goal"] == 0.0
    assert reward < 0.0


def test_stuck_termination_fires_after_no_progress_patience():
    config = clone_config()
    config["environment"]["obstacles"]["enabled"] = False
    config["environment"]["episode"]["max_steps"] = 300
    config["reward"]["stuck_patience"] = 5
    config["reward"]["stuck_progress_epsilon"] = 0.003
    config["reward"]["stuck_speed_threshold"] = 0.05
    config["reward"]["stuck_penalty"] = 8.0

    env = DroneNavEnv(config)
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(5.0, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    terminated = False
    info = {}

    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert info["stuck"] is True
    assert info["reward_terms"]["stuck"] == -8.0
    assert reward < 0.0

    env.close()


def test_idle_policy_does_not_receive_positive_average_reward_far_from_goal():
    config = clone_config()
    config["environment"]["obstacles"]["enabled"] = False
    config["environment"]["episode"]["max_steps"] = 20
    config["reward"]["stuck_patience"] = 100
    config["reward"]["low_speed_penalty"] = 0.35
    config["reward"]["stall_penalty"] = 1.5

    env = DroneNavEnv(config)
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(5.0, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    rewards = []
    for _ in range(10):
        _, reward, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        rewards.append(reward)
        if terminated or truncated:
            break

    assert np.mean(rewards) < 0.0
    env.close()


def test_orbit_like_tangential_motion_does_not_accumulate_positive_reward():
    env = make_public_scenario_env(position=[5.0, 6.0], goal=[5.0, 5.0], heading=0.0)

    total_reward, info = rollout_total_reward(env, [[0.0, 0.6]] * 20)

    assert total_reward < 0.0
    assert info["reached_goal"] is False
    assert info["collision"] is False
    env.close()


def test_wall_hugging_motion_does_not_accumulate_positive_reward():
    env = make_public_scenario_env(
        position=[0.25, 5.0],
        goal=[8.0, 5.0],
        heading=math.pi / 2.0,
    )

    total_reward, info = rollout_total_reward(env, [[math.pi / 2.0, 0.8]] * 20)

    assert total_reward < 0.0
    assert info["reached_goal"] is False
    assert info["collision"] is False
    env.close()


def test_side_to_side_oscillation_does_not_accumulate_positive_reward():
    env = make_public_scenario_env(position=[5.0, 5.0], goal=[5.0, 8.0], heading=0.0)
    oscillating_actions = [[math.pi / 2.0, 0.8], [-math.pi / 2.0, 0.8]] * 10

    total_reward, info = rollout_total_reward(env, oscillating_actions)

    assert total_reward < 0.0
    assert info["reached_goal"] is False
    assert info["collision"] is False
    env.close()


def test_action_space_documents_absolute_heading_and_speed_semantics():
    env = make_env()
    assert np.isclose(env.action_space.low[0], -np.pi)
    assert np.isclose(env.action_space.high[0], np.pi)
    assert np.isclose(env.action_space.low[1], 0.0)
    assert np.isclose(env.action_space.high[1], env.max_speed)
    env.close()
