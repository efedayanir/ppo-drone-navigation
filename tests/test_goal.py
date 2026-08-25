import numpy as np

from tests.test_utils import BASE_CONFIG, make_env


def test_goal_detected_inside_goal_radius():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)

    assert env._check_goal() is True


def test_goal_not_detected_outside_goal_radius():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.31, 5.0], dtype=np.float32)

    assert env._check_goal() is False


def test_goal_detected_exactly_on_boundary():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.3, 5.0], dtype=np.float32)

    assert env._check_goal() is True


def test_step_terminates_when_goal_reached():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(5.0, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert isinstance(obs, np.ndarray)
    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is True
    assert reward > 0.0


def test_goal_reward_is_applied():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = 0.3
    env.prev_action = np.zeros(2, dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)

    reward = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=False,
        reached_goal=True,
    )

    assert reward >= BASE_CONFIG["reward"]["goal_reward"] - 1.0


def test_info_reports_small_distance_near_goal():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.trajectory = [(5.0, 5.0)]

    info = env._get_info(collision=False, reached_goal=True, truncated=False)

    assert info["reached_goal"] is True
    assert info["distance_to_goal"] <= BASE_CONFIG["environment"]["goal"]["radius"]


def test_info_reports_normalized_distance_near_goal():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.trajectory = [(5.0, 5.0)]

    info = env._get_info(collision=False, reached_goal=True, truncated=False)

    assert info["distance_to_goal_normalized"] >= 0.0
    assert info["distance_to_goal_normalized"] <= (
        BASE_CONFIG["environment"]["goal"]["radius"] / env.world_diag + 1e-6
    )


def test_reward_terms_include_goal_bonus_after_goal_step():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(5.0, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()

    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["reward_terms"]["goal"] == BASE_CONFIG["reward"]["goal_reward"]
    assert np.isclose(info["reward_terms"]["total"], reward)


def test_goal_does_not_trigger_collision_flag_in_open_space():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([5.2, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(5.0, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["reached_goal"] is True
    assert info["collision"] is False