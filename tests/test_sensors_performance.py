import time

import numpy as np
import pytest

from env.obstacles import Obstacle
from env.sensors import get_sensor_readings


@pytest.mark.performance
def test_sensor_readings_with_many_rays_and_obstacles_stays_under_budget():
    obstacles = [
        Obstacle(x=0.5 + i, y=0.5 + j, radius=0.08)
        for i in range(10)
        for j in range(10)
    ]

    started_at = time.perf_counter()
    readings = get_sensor_readings(
        position=np.array([5.0, 5.0], dtype=np.float32),
        heading=0.0,
        num_rays=1000,
        sensor_max_range=5.0,
        normalize_sensor_readings=True,
        add_sensor_noise=False,
        sensor_noise_std=0.0,
        rng=np.random.default_rng(123),
        width=10.0,
        height=10.0,
        obstacles=obstacles,
    )
    elapsed = time.perf_counter() - started_at

    assert readings.shape == (1000,)
    assert readings.dtype == np.float32
    assert np.all(np.isfinite(readings))
    assert elapsed < 5.0
