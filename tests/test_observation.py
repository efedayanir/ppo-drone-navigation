import math

import numpy as np

from tests.test_utils import BASE_CONFIG, make_env


def test_observation_shape_matches_observation_space():
    env = make_env()
    obs, _ = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape


def test_observation_dtype_is_float32():
    env = make_env()
    obs, _ = env.reset(seed=123)

    assert obs.dtype == np.float32


def test_observation_contains_expected_number_of_elements():
    env = make_env()
    obs, _ = env.reset(seed=123)

    expected_dim = 2 + 2 + 2 + BASE_CONFIG["environment"]["sensors"]["num_rays"]
    assert obs.shape == (expected_dim,)


def test_goal_delta_in_observation_is_normalized_correctly():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([2.0, 3.0], dtype=np.float32)
    env.goal = np.array([7.5, 8.0], dtype=np.float32)

    obs = env._get_observation()

    expected_goal_delta = np.array([5.5, 5.0], dtype=np.float32)
    expected_goal_delta_norm = expected_goal_delta / env.world_diag

    assert np.isclose(obs[0], expected_goal_delta_norm[0])
    assert np.isclose(obs[1], expected_goal_delta_norm[1])


def test_velocity_in_observation_is_normalized_correctly():
    env = make_env()
    env.reset(seed=123)

    env.velocity = np.array([0.75, -0.75], dtype=np.float32)
    obs = env._get_observation()

    expected_velocity = env.velocity / env.max_speed
    assert np.isclose(obs[2], expected_velocity[0])
    assert np.isclose(obs[3], expected_velocity[1])


def test_velocity_in_observation_is_clipped_to_unit_range():
    env = make_env()
    env.reset(seed=123)

    env.velocity = np.array([10.0, -10.0], dtype=np.float32)
    obs = env._get_observation()

    assert -1.0 <= obs[2] <= 1.0
    assert -1.0 <= obs[3] <= 1.0


def test_heading_representation_is_cos_sin():
    env = make_env()
    env.reset(seed=123)

    env.heading = math.pi / 3.0
    obs = env._get_observation()

    assert np.isclose(obs[4], math.cos(env.heading))
    assert np.isclose(obs[5], math.sin(env.heading))


def test_heading_representation_has_unit_norm():
    env = make_env()
    env.reset(seed=123)

    env.heading = 1.234
    obs = env._get_observation()

    heading_vec = obs[4:6]
    assert np.isclose(np.linalg.norm(heading_vec), 1.0, atol=1e-6)


def test_sensor_slice_exists_and_has_correct_shape():
    env = make_env()
    obs, _ = env.reset(seed=123)

    num_rays = BASE_CONFIG["environment"]["sensors"]["num_rays"]
    sensor_values = obs[-num_rays:]

    assert sensor_values.shape == (num_rays,)


def test_sensor_values_are_bounded_when_normalized():
    env = make_env()
    obs, _ = env.reset(seed=123)

    num_rays = BASE_CONFIG["environment"]["sensors"]["num_rays"]
    sensor_values = obs[-num_rays:]

    assert np.all(sensor_values >= 0.0)
    assert np.all(sensor_values <= 1.0)


def test_observation_changes_when_position_changes():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([2.0, 2.0], dtype=np.float32)
    obs1 = env._get_observation()

    env.position = np.array([3.0, 2.0], dtype=np.float32)
    obs2 = env._get_observation()

    assert not np.allclose(obs1, obs2)


def test_observation_changes_when_heading_changes():
    env = make_env()
    env.reset(seed=123)

    env.heading = 0.0
    obs1 = env._get_observation()

    env.heading = math.pi / 2.0
    obs2 = env._get_observation()

    assert not np.allclose(obs1, obs2)


def test_observation_after_step_still_matches_space():
    env = make_env()
    env.reset(seed=123)

    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32


def test_observation_has_finite_values():
    env = make_env()
    obs, _ = env.reset(seed=123)

    assert np.all(np.isfinite(obs))


def test_observation_space_contains_reset_observation():
    env = make_env()
    obs, _ = env.reset(seed=123)

    assert env.observation_space.contains(obs)


def test_observation_space_contains_step_observation():
    env = make_env()
    env.reset(seed=123)

    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))

    assert env.observation_space.contains(obs)


def test_goal_delta_normalization_stays_within_expected_range():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([0.0, 0.0], dtype=np.float32)
    env.goal = np.array([env.width, env.height], dtype=np.float32)

    obs = env._get_observation()

    assert -1.0 <= obs[0] <= 1.0
    assert -1.0 <= obs[1] <= 1.0


def test_velocity_normalization_stays_within_expected_range():
    env = make_env()
    env.reset(seed=123)

    env.velocity = np.array([env.max_speed, -env.max_speed], dtype=np.float32)
    obs = env._get_observation()

    assert -1.0 <= obs[2] <= 1.0
    assert -1.0 <= obs[3] <= 1.0