import numpy as np

from env.drone_env import DroneNavEnv
from env.reward import compute_reward_with_info
from tests.test_utils import BASE_CONFIG, make_env


def test_reward_is_finite():
    env = make_env()
    env.reset(seed=123)

    reward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert np.isfinite(reward)


def test_progress_toward_goal_increases_reward():
    env = make_env()
    env.reset(seed=123)

    env.goal = np.array([8.0, 2.0], dtype=np.float32)
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.array([0.5, 0.0], dtype=np.float32)

    env.prev_distance_to_goal = 6.0
    env.position = np.array([3.0, 2.0], dtype=np.float32)
    reward_progress = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    env.prev_distance_to_goal = 6.0
    env.position = np.array([1.0, 2.0], dtype=np.float32)
    reward_regress = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward_progress > reward_regress


def test_step_penalty_makes_idle_reward_negative_when_no_progress():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 3.0
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)

    reward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward < 0.0


def test_collision_penalty_strongly_decreases_reward():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 3.0
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)

    reward_no_collision = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )
    reward_collision = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=True,
        reached_goal=False,
    )

    assert reward_collision < reward_no_collision
    assert reward_collision < 0.0


def test_goal_reward_strongly_increases_reward():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 0.3
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)

    reward_no_goal = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )
    reward_goal = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=True,
    )

    assert reward_goal > reward_no_goal
    assert reward_goal >= BASE_CONFIG["reward"]["goal_reward"] - 5.0


def test_smoothness_penalty_punishes_large_action_change():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 3.0
    env.velocity = np.array([0.2, 0.0], dtype=np.float32)

    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    reward_small_change = env._compute_reward(
        np.array([0.01, 0.01], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    reward_large_change = env._compute_reward(
        np.array([1.0, 1.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward_large_change < reward_small_change


def test_zero_smoothness_penalty_when_action_unchanged():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 3.0
    env.velocity = np.array([0.2, 0.0], dtype=np.float32)
    env.prev_action = np.array([0.2, 0.7], dtype=np.float32)

    reward_same = env._compute_reward(
        np.array([0.2, 0.7], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )
    reward_diff = env._compute_reward(
        np.array([0.3, 0.9], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward_same > reward_diff


def test_smoothness_wraps_heading_difference_across_pi_boundary():
    _, info = compute_reward_with_info(
        action=np.array([-np.pi, 0.5], dtype=np.float32),
        velocity=np.array([0.2, 0.0], dtype=np.float32),
        prev_action=np.array([np.pi, 0.5], dtype=np.float32),
        prev_distance_to_goal=3.0,
        current_distance=3.0,
        collision=False,
        reached_goal=False,
        config=BASE_CONFIG["reward"],
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert np.isclose(info["smoothness"], 0.0, atol=1e-6)


def test_progress_clipping_limits_large_progress_reward():
    env = make_env()
    env.reset(seed=123)

    env.prev_distance_to_goal = 10.0
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.1, 5.0], dtype=np.float32)
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.array([0.2, 0.0], dtype=np.float32)

    reward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert np.isfinite(reward)


def test_reward_penalizes_backward_progress():
    env = make_env()
    env.reset(seed=123)

    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    env.velocity = np.array([0.2, 0.0], dtype=np.float32)

    env.prev_distance_to_goal = 3.0
    env.position = np.array([4.0, 5.0], dtype=np.float32)
    reward_backward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    env.prev_distance_to_goal = 3.0
    env.position = np.array([6.0, 5.0], dtype=np.float32)
    reward_forward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward_backward < reward_forward


def test_stall_penalty_applies_when_no_progress_and_no_speed():
    config = dict(BASE_CONFIG)
    config["reward"] = dict(BASE_CONFIG["reward"])
    config["reward"]["stall_penalty"] = 2.0
    config["reward"]["shaping"] = dict(BASE_CONFIG["reward"]["shaping"])
    config["reward"]["shaping"]["stall_progress_threshold"] = 1e-3
    config["reward"]["shaping"]["stall_speed_threshold"] = 0.05
    config["reward"]["shaping"]["stall_distance_gate"] = 0.5

    env = DroneNavEnv(config)
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 3.0
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)

    env.velocity = np.zeros(2, dtype=np.float32)
    reward_stalled = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    env.velocity = np.array([0.2, 0.0], dtype=np.float32)
    reward_moving = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=False,
    )

    assert reward_stalled < reward_moving


def test_collision_and_goal_together_collision_dominates():
    env = make_env()
    env.reset(seed=123)

    reward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=True,
        reached_goal=True,
    )

    assert reward < 0.0


def test_alignment_reward_requires_goal_and_velocity_alignment():
    config = dict(BASE_CONFIG)
    config["reward"] = dict(BASE_CONFIG["reward"])
    config["reward"]["shaping"] = dict(BASE_CONFIG["reward"]["shaping"])
    config["reward"]["shaping"]["alignment_weight"] = 1.0

    action = np.array([0.0, 0.0], dtype=np.float32)
    prev_action = np.array([0.0, 0.0], dtype=np.float32)
    goal_vector = np.array([1.0, 0.0], dtype=np.float32)

    reward_bad, info_bad = compute_reward_with_info(
        action=action,
        velocity=np.array([0.0, 0.5], dtype=np.float32),
        prev_action=prev_action,
        prev_distance_to_goal=2.0,
        current_distance=2.0,
        collision=False,
        reached_goal=False,
        config=config["reward"],
        goal_vector=goal_vector,
    )

    reward_good, info_good = compute_reward_with_info(
        action=action,
        velocity=np.array([0.5, 0.0], dtype=np.float32),
        prev_action=prev_action,
        prev_distance_to_goal=2.0,
        current_distance=2.0,
        collision=False,
        reached_goal=False,
        config=config["reward"],
        goal_vector=goal_vector,
    )

    assert info_good["alignment"] > info_bad["alignment"]
    assert reward_good > reward_bad


def test_compute_reward_with_info_total_matches_sum_of_terms():
    config = BASE_CONFIG["reward"]

    reward, info = compute_reward_with_info(
        action=np.array([0.1, 0.5], dtype=np.float32),
        velocity=np.array([0.2, 0.0], dtype=np.float32),
        prev_action=np.array([0.0, 0.5], dtype=np.float32),
        prev_distance_to_goal=5.0,
        current_distance=4.8,
        collision=False,
        reached_goal=False,
        config=config,
        goal_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    summed = (
        info["collision"]
        + info["goal"]
        + info["progress"]
        + info["step"]
        + info["smoothness"]
        + info["stall"]
        + info["alignment"]
        + info["timeout"]
        + info["low_speed"]
        + info["stuck"]
    )

    assert np.isclose(reward, info["total"])
    assert np.isclose(info["total"], summed)
