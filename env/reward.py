from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _wrapped_angle_delta(angle_a: float, angle_b: float) -> float:
    delta = angle_a - angle_b
    return float((delta + math.pi) % (2.0 * math.pi) - math.pi)


def _compute_reward_terms(
    action: np.ndarray,
    velocity: np.ndarray,
    prev_action: np.ndarray,
    prev_distance_to_goal: float,
    current_distance: float,
    collision: bool,
    reached_goal: bool,
    truncated: bool,
    config: Dict[str, Any],
    goal_vector: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    shaping = config.get("shaping", {})

    terms = {
        "collision": 0.0,
        "goal": 0.0,
        "progress": 0.0,
        "step": 0.0,
        "smoothness": 0.0,
        "stall": 0.0,
        "alignment": 0.0,
        "timeout": 0.0,
        "low_speed": 0.0,
        "stuck": 0.0,
        "total": 0.0,
    }

    # ------------------------------------------------------------
    # Collision dominates everything.
    # If collision happens, this transition is a failure.
    # Goal reward is suppressed in collision cases.
    # ------------------------------------------------------------
    if collision:
        terms["collision"] = -float(config.get("collision_penalty", 0.0))
        terms["goal"] = 0.0
    elif reached_goal:
        terms["goal"] = float(config.get("goal_reward", 0.0))

    if truncated:
        terms["timeout"] = -float(config.get("timeout_penalty", 0.0))

    # Progress reward
    raw_progress = float(prev_distance_to_goal - current_distance)

    if shaping.get("clip_progress", False):
        raw_progress = _clip(
            raw_progress,
            float(shaping.get("progress_clip_min", -np.inf)),
            float(shaping.get("progress_clip_max", np.inf)),
        )

    progress_normalizer = float(
        shaping.get("progress_normalizer", max(prev_distance_to_goal, 1.0))
    )
    progress_normalizer = max(progress_normalizer, 1e-6)

    relative_progress = raw_progress / progress_normalizer
    terms["progress"] = float(config.get("progress_weight", 0.0)) * relative_progress

    # Per-step penalty
    terms["step"] = -float(config.get("step_penalty", 0.0))

    # Smoothness penalty
    max_speed_for_norm = float(config.get("action_speed_max", 1.5))
    max_speed_for_norm = max(max_speed_for_norm, 1e-6)

    heading_delta = abs(_wrapped_angle_delta(float(action[0]), float(prev_action[0])))
    heading_delta /= math.pi

    speed_delta = abs(float(action[1]) - float(prev_action[1])) / max_speed_for_norm

    action_delta = heading_delta + speed_delta
    terms["smoothness"] = -float(config.get("smoothness_weight", 0.0)) * action_delta

    # Motion / stall logic
    speed = _safe_norm(velocity)

    stall_progress_threshold = float(shaping.get("stall_progress_threshold", 1e-3))
    stall_speed_threshold = float(shaping.get("stall_speed_threshold", 5e-2))
    stall_distance_gate = float(shaping.get("stall_distance_gate", 0.5))
    stall_penalty = float(config.get("stall_penalty", 0.0))

    progress_abs = abs(raw_progress)

    if current_distance > stall_distance_gate:
        if progress_abs < stall_progress_threshold and speed < stall_speed_threshold:
            terms["stall"] = -stall_penalty


    # Low-speed penalty
    low_speed_penalty = float(config.get("low_speed_penalty", 0.0))
    min_speed_reward_gate = float(shaping.get("min_speed_reward_gate", 0.10))

    if current_distance > stall_distance_gate and speed < min_speed_reward_gate:
        terms["low_speed"] = -low_speed_penalty

    # Alignment reward
    alignment_weight = float(shaping.get("alignment_weight", 0.0))

    if goal_vector is not None and alignment_weight != 0.0:
        goal_norm = _safe_norm(goal_vector)

        if goal_norm > 1e-8:
            goal_dir = goal_vector / goal_norm

            if speed > 1e-8:
                vel_dir = velocity / speed
                alignment = float(np.dot(goal_dir, vel_dir))
            else:
                alignment = 0.0

            speed_scale = min(speed / max(min_speed_reward_gate, 1e-6), 1.0)
            terms["alignment"] = alignment_weight * alignment * speed_scale

    # Final total
    terms["total"] = (
        terms["collision"]
        + terms["goal"]
        + terms["progress"]
        + terms["step"]
        + terms["smoothness"]
        + terms["stall"]
        + terms["alignment"]
        + terms["timeout"]
        + terms["low_speed"]
        + terms["stuck"]
    )

    return terms


def compute_reward(
    action: np.ndarray,
    velocity: np.ndarray,
    prev_action: np.ndarray,
    prev_distance_to_goal: float,
    current_distance: float,
    collision: bool,
    reached_goal: bool,
    config: Dict[str, Any],
    goal_vector: Optional[np.ndarray] = None,
    truncated: bool = False,
) -> float:
    terms = _compute_reward_terms(
        action=np.asarray(action, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        prev_action=np.asarray(prev_action, dtype=np.float32),
        prev_distance_to_goal=float(prev_distance_to_goal),
        current_distance=float(current_distance),
        collision=bool(collision),
        reached_goal=bool(reached_goal),
        truncated=bool(truncated),
        config=config,
        goal_vector=None if goal_vector is None else np.asarray(goal_vector, dtype=np.float32),
    )
    return float(terms["total"])


def compute_reward_with_info(
    action: np.ndarray,
    velocity: np.ndarray,
    prev_action: np.ndarray,
    prev_distance_to_goal: float,
    current_distance: float,
    collision: bool,
    reached_goal: bool,
    config: Dict[str, Any],
    goal_vector: Optional[np.ndarray] = None,
    truncated: bool = False,
) -> Tuple[float, Dict[str, float]]:
    terms = _compute_reward_terms(
        action=np.asarray(action, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        prev_action=np.asarray(prev_action, dtype=np.float32),
        prev_distance_to_goal=float(prev_distance_to_goal),
        current_distance=float(current_distance),
        collision=bool(collision),
        reached_goal=bool(reached_goal),
        truncated=bool(truncated),
        config=config,
        goal_vector=None if goal_vector is None else np.asarray(goal_vector, dtype=np.float32),
    )
    return float(terms["total"]), terms
