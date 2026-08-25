import numpy as np

from env.obstacles import Obstacle
from tests.test_utils import BASE_CONFIG, make_env


def test_reset_returns_valid_observation_and_info():
    env = make_env()

    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

    expected_keys = {
        "collision",
        "reached_goal",
        "truncated",
        "stuck",
        "distance_to_goal",
        "distance_to_goal_normalized",
        "speed",
        "speed_normalized",
        "heading",
        "angular_velocity",
        "step_count",
        "path_length",
        "path_efficiency",
        "goal_position",
        "agent_position",
        "last_action",
        "reward_terms",
        "no_progress_steps",
        "best_distance_to_goal",
    }
    assert expected_keys.issubset(info.keys())


def test_reset_initializes_step_count_and_velocity():
    env = make_env()

    env.reset(seed=123)

    assert env.step_count == 0
    assert np.allclose(env.velocity, np.zeros(2, dtype=np.float32))
    assert env.angular_velocity == 0.0


def test_reset_initializes_prev_action_and_last_action_to_zero():
    env = make_env()

    env.reset(seed=123)

    zeros = np.zeros(2, dtype=np.float32)
    assert np.allclose(env.prev_action, zeros)
    assert np.allclose(env.last_action, zeros)


def test_reset_creates_nonempty_trajectory_with_start_position():
    env = make_env()

    env.reset(seed=123)

    assert len(env.trajectory) == 1
    start = np.array(env.trajectory[0], dtype=np.float32)
    assert np.allclose(start, env.position)


def test_reset_start_and_goal_are_within_world_bounds():
    env = make_env()

    env.reset(seed=123)

    assert 0.0 <= env.position[0] <= BASE_CONFIG["environment"]["world"]["width"]
    assert 0.0 <= env.position[1] <= BASE_CONFIG["environment"]["world"]["height"]
    assert 0.0 <= env.goal[0] <= BASE_CONFIG["environment"]["world"]["width"]
    assert 0.0 <= env.goal[1] <= BASE_CONFIG["environment"]["world"]["height"]


def test_reset_respects_minimum_start_goal_distance():
    env = make_env()

    env.reset(seed=123)

    distance = np.linalg.norm(env.goal - env.position)
    assert distance >= BASE_CONFIG["environment"]["goal"]["min_start_goal_distance"]


def test_reset_generates_expected_number_of_obstacles():
    env = make_env()

    env.reset(seed=123)

    expected = BASE_CONFIG["environment"]["obstacles"]["count_min"]
    assert len(env.obstacles) == expected


def test_reset_start_position_is_not_in_collision():
    env = make_env()

    env.reset(seed=123)

    assert env._check_collision() is False


def test_reset_goal_is_not_immediately_reached():
    env = make_env()

    env.reset(seed=123)

    assert env._check_goal() is False


def test_reset_sets_prev_distance_to_goal_correctly():
    env = make_env()

    env.reset(seed=123)

    actual_distance = np.linalg.norm(env.goal - env.position)
    assert np.isclose(env.prev_distance_to_goal, actual_distance)
    assert np.isclose(env.best_distance_to_goal, actual_distance)


def test_reset_with_same_seed_is_reproducible():
    env1 = make_env()
    env2 = make_env()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.allclose(env1.position, env2.position)
    assert np.allclose(env1.goal, env2.goal)
    assert np.allclose(obs1, obs2)
    assert len(env1.obstacles) == len(env2.obstacles)

    for o1, o2 in zip(env1.obstacles, env2.obstacles):
        assert np.isclose(o1.x, o2.x)
        assert np.isclose(o1.y, o2.y)
        assert np.isclose(o1.radius, o2.radius)

    assert np.isclose(info1["distance_to_goal"], info2["distance_to_goal"])
    assert np.isclose(info1["distance_to_goal_normalized"], info2["distance_to_goal_normalized"])


def test_reset_options_create_deterministic_scenario_via_public_api():
    env = make_env()

    obs, info = env.reset(
        seed=123,
        options={
            "position": [2.0, 3.0],
            "goal": [7.0, 3.0],
            "heading": np.pi / 2.0,
            "velocity": [0.1, 0.2],
            "angular_velocity": 0.3,
            "obstacles": [
                Obstacle(x=4.0, y=4.0, radius=0.4),
                {"x": 6.0, "y": 6.0, "radius": 0.5},
                (8.0, 2.0, 0.3),
            ],
        },
    )

    assert env.observation_space.contains(obs)
    assert np.allclose(env.position, np.array([2.0, 3.0], dtype=np.float32))
    assert np.allclose(env.goal, np.array([7.0, 3.0], dtype=np.float32))
    assert np.isclose(env.heading, np.pi / 2.0)
    assert np.allclose(env.velocity, np.array([0.1, 0.2], dtype=np.float32))
    assert np.isclose(env.angular_velocity, 0.3)
    assert len(env.obstacles) == 3
    assert env.trajectory == [(2.0, 3.0)]
    assert np.isclose(env.prev_distance_to_goal, 5.0)
    assert np.isclose(env.best_distance_to_goal, 5.0)
    assert info["step_count"] == 0
    assert np.allclose(info["agent_position"], env.position)
    assert np.allclose(info["goal_position"], env.goal)

    env.close()


def test_reset_after_step_clears_episode_progress():
    env = make_env()

    env.reset(seed=123)
    env.step(np.array([0.0, 0.5], dtype=np.float32))

    assert env.step_count == 1
    assert len(env.trajectory) >= 2

    env.reset(seed=456)

    assert env.step_count == 0
    assert len(env.trajectory) == 1
    assert np.allclose(env.velocity, np.zeros(2, dtype=np.float32))
    assert env.angular_velocity == 0.0
    assert np.allclose(env.prev_action, np.zeros(2, dtype=np.float32))
    assert np.allclose(env.last_action, np.zeros(2, dtype=np.float32))
    assert env.no_progress_steps == 0


def test_reset_sets_initial_heading_from_config():
    env = make_env()

    env.reset(seed=123)

    assert np.isclose(
        env.heading,
        BASE_CONFIG["environment"]["drone"]["initial_heading"],
    )


def test_reset_returns_observation_inside_space():
    env = make_env()

    obs, _ = env.reset(seed=123)

    assert env.observation_space.contains(obs)


def test_reset_info_has_zero_speed_and_zero_path_length():
    env = make_env()

    _, info = env.reset(seed=123)

    assert np.isclose(info["speed"], 0.0)
    assert np.isclose(info["speed_normalized"], 0.0)
    assert np.isclose(info["path_length"], 0.0)


def test_reset_info_step_count_is_zero():
    env = make_env()

    _, info = env.reset(seed=123)

    assert info["step_count"] == 0


def test_reset_info_last_action_is_zero():
    env = make_env()

    _, info = env.reset(seed=123)

    assert np.allclose(info["last_action"], np.zeros(2, dtype=np.float32))


def test_reset_reward_terms_are_zeroed():
    env = make_env()

    _, info = env.reset(seed=123)

    reward_terms = info["reward_terms"]
    expected_keys = {
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

    assert expected_keys.issubset(reward_terms.keys())
    assert all(np.isclose(reward_terms[k], 0.0) for k in expected_keys)
