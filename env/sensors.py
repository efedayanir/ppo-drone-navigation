from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _ray_circle_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Optional[float]:
    """
    Return the smallest non-negative ray parameter t such that:
        origin + t * direction
    lies on the circle.

    Returns None if there is no forward intersection.
    Assumes direction is unit length.
    """
    oc = origin - center

    # Solve ||oc + t d||^2 = r^2
    # t^2 + 2(oc.d)t + (oc.oc - r^2) = 0
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)

    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return None

    return min(candidates)


def _ray_aabb_boundary_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    width: float,
    height: float,
) -> float:
    """
    Distance from ray origin to first intersection with the world boundary.
    World is axis-aligned box [0, width] x [0, height].

    Assumes origin is inside or on the boundary.
    Assumes direction is unit length.
    """
    eps = 1e-12
    candidates = []

    dx = float(direction[0])
    dy = float(direction[1])
    ox = float(origin[0])
    oy = float(origin[1])

    if abs(dx) > eps:
        t_left = (0.0 - ox) / dx
        t_right = (width - ox) / dx
        if t_left >= 0.0:
            y = oy + t_left * dy
            if 0.0 <= y <= height:
                candidates.append(t_left)
        if t_right >= 0.0:
            y = oy + t_right * dy
            if 0.0 <= y <= height:
                candidates.append(t_right)

    if abs(dy) > eps:
        t_bottom = (0.0 - oy) / dy
        t_top = (height - oy) / dy
        if t_bottom >= 0.0:
            x = ox + t_bottom * dx
            if 0.0 <= x <= width:
                candidates.append(t_bottom)
        if t_top >= 0.0:
            x = ox + t_top * dx
            if 0.0 <= x <= width:
                candidates.append(t_top)

    if not candidates:
        return 0.0

    return min(candidates)


def cast_ray(
    position: np.ndarray,
    angle: float,
    sensor_max_range: float,
    width: float,
    height: float,
    obstacles: Iterable,
) -> float:
    """
    Cast a ray from position at the given angle and return the distance to the
    first obstacle or boundary, capped at sensor_max_range.
    """
    origin = np.asarray(position, dtype=np.float32)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    min_dist = min(
        float(sensor_max_range),
        _ray_aabb_boundary_intersection(origin, direction, width, height),
    )

    for obstacle in obstacles:
        center = np.array([obstacle.x, obstacle.y], dtype=np.float32)
        hit = _ray_circle_intersection(origin, direction, center, float(obstacle.radius))
        if hit is not None and 0.0 <= hit < min_dist:
            min_dist = hit

    return float(np.clip(min_dist, 0.0, sensor_max_range))


def get_sensor_readings(
    position: np.ndarray,
    heading: float,
    num_rays: int,
    sensor_max_range: float,
    normalize_sensor_readings: bool,
    add_sensor_noise: bool,
    sensor_noise_std: float,
    rng,
    width: float,
    height: float,
    obstacles,
) -> np.ndarray:
    """
    Compute evenly spaced 360-degree lidar-style readings around the agent heading.
    """
    position = np.asarray(position, dtype=np.float32)
    readings = np.zeros(num_rays, dtype=np.float32)

    for i in range(num_rays):
        angle = heading + (2.0 * math.pi * i / num_rays)
        readings[i] = cast_ray(
            position=position,
            angle=angle,
            sensor_max_range=sensor_max_range,
            width=width,
            height=height,
            obstacles=obstacles,
        )

    if add_sensor_noise:
        noise = rng.normal(0.0, sensor_noise_std, size=num_rays).astype(np.float32)
        readings = readings + noise

    readings = np.clip(readings, 0.0, sensor_max_range)

    if normalize_sensor_readings:
        readings = readings / max(sensor_max_range, 1e-8)

    return readings.astype(np.float32)