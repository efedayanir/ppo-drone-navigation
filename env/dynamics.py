import math
from typing import Callable, Tuple

import numpy as np


def apply_dynamics(
    position: np.ndarray,
    velocity: np.ndarray,
    heading: float,
    angular_velocity: float,
    action: np.ndarray,
    dt: float,
    max_speed: float,
    max_acceleration: float,
    max_angular_speed: float,
    heading_kp: float,
    heading_kd: float,
    speed_kp: float,
    speed_kd: float,
    wrap_angle_fn: Callable[[float], float],
    lateral_damping: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simple research-grade low-level dynamics for 2D navigation.

    Action semantics:
        action[0] -> absolute target heading in radians
        action[1] -> absolute target speed in m/s

    Design goals:
    - stationary action meaning
    - bounded dynamics
    - smooth tracking
    - simple but physically coherent motion

    Notes:
    - Heading is tracked with PD control on angular velocity.
    - Speed is tracked along the drone's forward axis with PD control.
    - Lateral velocity is damped to avoid unrealistic sideways drift.
    """

    position = np.asarray(position, dtype=np.float32).copy()
    velocity = np.asarray(velocity, dtype=np.float32).copy()
    action = np.asarray(action, dtype=np.float32)

    target_heading = wrap_angle_fn(float(action[0]))
    target_speed = float(np.clip(action[1], 0.0, max_speed))

    # Forward and lateral unit vectors from current heading
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    lateral = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float32)

    # -------------------------
    # Angular dynamics (PD)
    # -------------------------
    heading_error = wrap_angle_fn(target_heading - heading)
    angular_acceleration = heading_kp * heading_error - heading_kd * angular_velocity

    angular_velocity = float(
        np.clip(
            angular_velocity + angular_acceleration * dt,
            -max_angular_speed,
            max_angular_speed,
        )
    )

    heading = wrap_angle_fn(heading + angular_velocity * dt)

    # Recompute basis after heading update
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    lateral = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float32)

    # -------------------------
    # Linear dynamics
    # -------------------------
    forward_speed = float(np.dot(velocity, forward))
    lateral_speed = float(np.dot(velocity, lateral))

    speed_error = target_speed - forward_speed

    # PD on forward speed
    forward_acc_cmd = speed_kp * speed_error - speed_kd * forward_speed

    # Damping on lateral slip
    lateral_acc_cmd = -lateral_damping * lateral_speed

    # Combine in world frame
    acceleration = forward_acc_cmd * forward + lateral_acc_cmd * lateral

    # Bound total acceleration magnitude
    acc_norm = float(np.linalg.norm(acceleration))
    if acc_norm > max_acceleration and acc_norm > 1e-8:
        acceleration = acceleration / acc_norm * max_acceleration

    velocity = velocity + acceleration.astype(np.float32) * dt

    # Bound speed magnitude
    speed = float(np.linalg.norm(velocity))
    if speed > max_speed and speed > 1e-8:
        velocity = velocity / speed * max_speed

    position = position + velocity * dt

    return (
        position.astype(np.float32),
        velocity.astype(np.float32),
        float(heading),
        float(angular_velocity),
    )