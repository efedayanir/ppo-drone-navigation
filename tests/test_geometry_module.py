import numpy as np

from env.obstacles import Obstacle
from env.geometry import (
    check_collision,
    check_goal,
    point_is_safe,
    wrap_angle,
)


def test_wrap_angle_module_keeps_result_in_pi_range():
    wrapped = wrap_angle(4 * np.pi)
    assert -np.pi <= wrapped < np.pi


def test_wrap_angle_module_wraps_positive_and_negative():
    assert np.isclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1, atol=1e-6)
    assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1, atol=1e-6)


def test_point_is_safe_module_rejects_wall_overlap():
    point = np.array([0.1, 5.0], dtype=np.float32)
    assert point_is_safe(point, 0.2, 10.0, 10.0, []) is False


def test_point_is_safe_module_rejects_obstacle_overlap():
    point = np.array([5.3, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=5.0, y=5.0, radius=0.5)]
    assert point_is_safe(point, 0.2, 10.0, 10.0, obstacles) is False


def test_point_is_safe_module_accepts_clear_point():
    point = np.array([2.0, 2.0], dtype=np.float32)
    obstacles = [Obstacle(x=8.0, y=8.0, radius=0.5)]
    assert point_is_safe(point, 0.2, 10.0, 10.0, obstacles) is True


def test_check_collision_module_detects_wall():
    position = np.array([0.1, 5.0], dtype=np.float32)
    assert check_collision(position, 0.2, 10.0, 10.0, []) is True


def test_check_collision_module_detects_obstacle():
    position = np.array([5.6, 5.0], dtype=np.float32)
    obstacles = [Obstacle(x=5.0, y=5.0, radius=0.5)]
    assert check_collision(position, 0.2, 10.0, 10.0, obstacles) is True


def test_check_goal_module_accepts_boundary():
    position = np.array([5.0, 5.0], dtype=np.float32)
    goal = np.array([5.3, 5.0], dtype=np.float32)
    assert check_goal(position, goal, 0.3) is True