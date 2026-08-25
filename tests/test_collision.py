import numpy as np

from env.obstacles import Obstacle
from tests.test_utils import BASE_CONFIG, make_env


def test_no_collision_in_open_space():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)

    assert env._check_collision() is False


def test_collision_with_left_wall():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)

    assert env._check_collision() is True


def test_collision_with_right_wall():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([9.85, 5.0], dtype=np.float32)

    assert env._check_collision() is True


def test_collision_with_top_wall():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 9.9], dtype=np.float32)

    assert env._check_collision() is True


def test_collision_with_bottom_wall():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 0.1], dtype=np.float32)

    assert env._check_collision() is True


def test_collision_with_circular_obstacle():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = [Obstacle(x=5.0, y=5.0, radius=0.5)]
    env.position = np.array([5.6, 5.0], dtype=np.float32)

    assert env._check_collision() is True


def test_no_collision_outside_obstacle_margin():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = [Obstacle(x=5.0, y=5.0, radius=0.5)]
    env.position = np.array([6.0, 5.0], dtype=np.float32)

    assert env._check_collision() is False


def test_step_terminates_on_collision():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(0.1, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert isinstance(obs, np.ndarray)
    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert reward < 0.0


def test_collision_penalty_is_present_in_reward_terms():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    reward, terms = env._compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        collision=True,
        reached_goal=False,
    ), None

    assert reward < 0.0


def test_collision_info_contains_reward_terms():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(0.1, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert "reward_terms" in info
    assert info["reward_terms"]["collision"] == -BASE_CONFIG["reward"]["collision_penalty"]
    assert np.isclose(info["reward_terms"]["total"], reward)


def test_collision_does_not_set_goal_flag_in_open_non_goal_case():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(0.1, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["collision"] is True
    assert info["reached_goal"] is False


def test_collision_info_reports_nonnegative_distance_to_goal():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([0.1, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(0.1, 5.0)]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["distance_to_goal"] >= 0.0
    assert info["distance_to_goal_normalized"] >= 0.0