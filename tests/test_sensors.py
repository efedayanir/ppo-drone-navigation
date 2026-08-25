import numpy as np

from env.drone_env import DroneNavEnv
from env.obstacles import Obstacle
from tests.test_utils import BASE_CONFIG, make_env


def test_sensor_vector_has_correct_shape():
    env = make_env()
    env.reset(seed=123)

    readings = env._get_sensor_readings()

    assert isinstance(readings, np.ndarray)
    assert readings.shape == (BASE_CONFIG["environment"]["sensors"]["num_rays"],)


def test_sensor_readings_are_normalized_when_enabled():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)

    readings = env._get_sensor_readings()

    assert np.all(readings >= 0.0)
    assert np.all(readings <= 1.0)


def test_front_sensor_detects_nearby_obstacle():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.heading = 0.0
    env.obstacles = [Obstacle(x=6.0, y=5.0, radius=0.4)]

    readings = env._get_sensor_readings()

    assert readings[0] < 1.0


def test_front_sensor_clear_path_gives_longer_range_than_blocked_path():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.heading = 0.0

    env.obstacles = []
    clear_reading = env._get_sensor_readings()[0]

    env.obstacles = [Obstacle(x=6.0, y=5.0, radius=0.4)]
    blocked_reading = env._get_sensor_readings()[0]

    assert blocked_reading < clear_reading


def test_sensor_changes_with_heading_rotation():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.obstacles = [Obstacle(x=6.0, y=5.0, radius=0.4)]

    env.heading = 0.0
    readings_heading_0 = env._get_sensor_readings()

    env.heading = np.pi / 2.0
    readings_heading_90 = env._get_sensor_readings()

    assert not np.allclose(readings_heading_0, readings_heading_90)


def test_ray_cast_returns_boundary_limited_distance_in_open_space():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)

    distance = env._cast_ray(0.0)

    assert distance > 0.0
    assert distance <= BASE_CONFIG["environment"]["sensors"]["max_range"]


def test_sensor_detects_near_wall():
    env = make_env()
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([9.0, 5.0], dtype=np.float32)
    env.heading = 0.0

    readings = env._get_sensor_readings()

    assert readings[0] < 1.0


def test_observation_contains_sensor_values():
    env = make_env()
    env.reset(seed=123)

    obs = env._get_observation()
    num_rays = BASE_CONFIG["environment"]["sensors"]["num_rays"]
    sensor_slice = obs[-num_rays:]

    assert sensor_slice.shape == (num_rays,)
    assert np.all(sensor_slice >= 0.0)
    assert np.all(sensor_slice <= 1.0)


def test_sensor_noise_changes_readings_when_enabled():
    noisy_config = dict(BASE_CONFIG)
    noisy_config["environment"] = dict(BASE_CONFIG["environment"])
    noisy_config["environment"]["sensors"] = dict(BASE_CONFIG["environment"]["sensors"])
    noisy_config["environment"]["sensors"]["add_noise"] = True
    noisy_config["environment"]["sensors"]["noise_std"] = 0.2

    env = DroneNavEnv(noisy_config)
    env.reset(seed=123)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)

    readings_1 = env._get_sensor_readings()
    readings_2 = env._get_sensor_readings()

    assert not np.allclose(readings_1, readings_2)


def test_front_sensor_matches_expected_normalized_circle_hit_distance():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.heading = 0.0
    env.obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    readings = env._get_sensor_readings()

    expected_distance = 1.5
    expected_normalized = expected_distance / env.sensor_max_range

    assert np.isclose(readings[0], expected_normalized, atol=1e-5)


def test_obstacle_is_detected_before_wall_when_both_are_on_same_ray():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.heading = 0.0
    env.obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    reading = env._get_sensor_readings()[0]
    wall_only_distance = (env.width - env.position[0]) / env.sensor_max_range

    assert reading < wall_only_distance


def test_sensor_readings_remain_finite_under_rotation_sweep():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.obstacles = [
        Obstacle(x=7.0, y=5.0, radius=0.5),
        Obstacle(x=5.0, y=7.0, radius=0.5),
    ]

    for heading in np.linspace(-np.pi, np.pi, 50):
        env.heading = float(heading)
        readings = env._get_sensor_readings()
        assert np.all(np.isfinite(readings))
        assert np.all(readings >= 0.0)
        assert np.all(readings <= 1.0)


def test_sensor_readings_change_smoothly_under_small_heading_change():
    env = make_env()
    env.reset(seed=123)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    env.heading = 0.0
    r1 = env._get_sensor_readings()

    env.heading = 0.02
    r2 = env._get_sensor_readings()

    delta = np.linalg.norm(r2 - r1)
    assert delta < 0.2