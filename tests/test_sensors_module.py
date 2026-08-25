import numpy as np

from env.obstacles import Obstacle
from env.sensors import cast_ray, get_sensor_readings


def test_cast_ray_returns_boundary_distance_in_open_space():
    position = np.array([5.0, 5.0], dtype=np.float32)

    distance = cast_ray(
        position=position,
        angle=0.0,
        sensor_max_range=10.0,
        width=10.0,
        height=10.0,
        obstacles=[],
    )

    assert np.isclose(distance, 5.0, atol=1e-6)


def test_cast_ray_is_capped_by_sensor_max_range():
    position = np.array([5.0, 5.0], dtype=np.float32)

    distance = cast_ray(
        position=position,
        angle=0.0,
        sensor_max_range=3.0,
        width=20.0,
        height=20.0,
        obstacles=[],
    )

    assert np.isclose(distance, 3.0, atol=1e-6)


def test_cast_ray_detects_obstacle_before_boundary():
    position = np.array([5.0, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    distance = cast_ray(
        position=position,
        angle=0.0,
        sensor_max_range=10.0,
        width=20.0,
        height=20.0,
        obstacles=obstacles,
    )

    assert np.isclose(distance, 1.5, atol=1e-6)


def test_cast_ray_returns_max_range_when_obstacle_is_beyond_range():
    position = np.array([5.0, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=20.0, y=5.0, radius=1.0)]

    distance = cast_ray(
        position=position,
        angle=0.0,
        sensor_max_range=3.0,
        width=100.0,
        height=100.0,
        obstacles=obstacles,
    )

    assert np.isclose(distance, 3.0, atol=1e-6)


def test_cast_ray_detects_vertical_boundary():
    position = np.array([5.0, 5.0], dtype=np.float32)

    distance = cast_ray(
        position=position,
        angle=np.pi / 2.0,
        sensor_max_range=10.0,
        width=10.0,
        height=10.0,
        obstacles=[],
    )

    assert np.isclose(distance, 5.0, atol=1e-6)


def test_cast_ray_detects_diagonal_boundary():
    position = np.array([5.0, 5.0], dtype=np.float32)

    distance = cast_ray(
        position=position,
        angle=np.pi / 4.0,
        sensor_max_range=20.0,
        width=10.0,
        height=10.0,
        obstacles=[],
    )

    assert np.isclose(distance, np.sqrt(50.0), atol=1e-6)


def test_get_sensor_readings_returns_expected_shape():
    position = np.array([5.0, 5.0], dtype=np.float32)

    readings = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=8,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=10.0,
        height=10.0,
        obstacles=[],
    )

    assert readings.shape == (8,)
    assert np.all(readings >= 0.0)
    assert np.all(readings <= 1.0)


def test_get_sensor_readings_front_ray_matches_expected_normalized_distance():
    position = np.array([5.0, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    readings = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=8,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=20.0,
        height=20.0,
        obstacles=obstacles,
    )

    expected = 1.5 / 3.0
    assert np.isclose(readings[0], expected, atol=1e-6)


def test_get_sensor_readings_without_normalization_stay_in_distance_units():
    position = np.array([5.0, 5.0], dtype=np.float32)

    readings = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=8,
        sensor_max_range=3.0,
        normalize_sensor_readings=False,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=20.0,
        height=20.0,
        obstacles=[],
    )

    assert readings.shape == (8,)
    assert np.all(readings >= 0.0)
    assert np.all(readings <= 3.0)


def test_get_sensor_readings_noise_changes_output():
    position = np.array([5.0, 5.0], dtype=np.float32)

    readings_1 = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=8,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=True,
        sensor_noise_std=0.1,
        rng=np.random.default_rng(123),
        width=20.0,
        height=20.0,
        obstacles=[],
    )

    readings_2 = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=8,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=True,
        sensor_noise_std=0.1,
        rng=np.random.default_rng(456),
        width=20.0,
        height=20.0,
        obstacles=[],
    )

    assert not np.allclose(readings_1, readings_2)


def test_get_sensor_readings_change_smoothly_for_small_heading_change():
    position = np.array([5.0, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=7.0, y=5.0, radius=0.5)]

    readings_1 = get_sensor_readings(
        position=position,
        heading=0.0,
        num_rays=16,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=20.0,
        height=20.0,
        obstacles=obstacles,
    )

    readings_2 = get_sensor_readings(
        position=position,
        heading=0.02,
        num_rays=16,
        sensor_max_range=3.0,
        normalize_sensor_readings=True,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=20.0,
        height=20.0,
        obstacles=obstacles,
    )

    delta = np.linalg.norm(readings_2 - readings_1)
    assert delta < 0.3