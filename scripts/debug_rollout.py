from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from env.drone_env import DroneNavEnv
from tests.test_utils import BASE_CONFIG


def make_env() -> DroneNavEnv:
    return DroneNavEnv(BASE_CONFIG)


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def random_rollout_diagnostics(steps: int = 3000, seed: int = 123) -> None:
    _print_header("RANDOM ROLLOUT DIAGNOSTICS")

    env = make_env()
    obs, info = env.reset(seed=seed)

    reward_values: List[float] = []
    episode_lengths: List[int] = []
    episode_final_distances: List[float] = []

    reward_terms: Dict[str, List[float]] = defaultdict(list)

    episode_count = 0
    collision_count = 0
    goal_count = 0
    truncation_count = 0
    stuck_count = 0

    current_episode_len = 0

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(reward):
            print(f"[STEP {step}] BROKEN: reward is NaN/inf")
            return

        if not np.all(np.isfinite(obs)):
            print(f"[STEP {step}] BROKEN: observation is NaN/inf")
            return

        reward_values.append(float(reward))
        current_episode_len += 1

        for key, value in info.get("reward_terms", {}).items():
            reward_terms[key].append(float(value))

        if terminated or truncated:
            episode_count += 1
            episode_lengths.append(current_episode_len)
            episode_final_distances.append(float(info.get("distance_to_goal", 0.0)))

            if info.get("collision", False):
                collision_count += 1
            if info.get("reached_goal", False):
                goal_count += 1
            if info.get("stuck", False):
                stuck_count += 1
            if truncated:
                truncation_count += 1

            current_episode_len = 0
            obs, info = env.reset(seed=seed + episode_count + 1)

    print("Rollout finished.")
    print(f"Total steps observed            : {steps}")
    print(f"Reward mean/std                 : {_safe_mean(reward_values):.6f} / {_safe_std(reward_values):.6f}")
    print(f"Episodes finished               : {episode_count}")
    print(f"Goals reached                   : {goal_count}")
    print(f"Collisions                      : {collision_count}")
    print(f"Stuck terminations              : {stuck_count}")
    print(f"Truncations                     : {truncation_count}")
    print(f"Mean episode length             : {_safe_mean(episode_lengths):.3f}")
    print(f"Mean final distance to goal     : {_safe_mean(episode_final_distances):.3f}")

    print("\nReward term means:")
    for key in sorted(reward_terms.keys()):
        print(f"  {key:>12}: {_safe_mean(reward_terms[key]):.6f}")

    env.close()


def zero_action_drift_test(steps: int = 100, seed: int = 123) -> None:
    _print_header("ZERO ACTION DRIFT TEST")

    env = make_env()
    env.reset(seed=seed)

    env.obstacles = []
    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.goal = np.array([8.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(float(env.position[0]), float(env.position[1]))]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    start_position = env.position.copy()
    start_heading = float(env.heading)

    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    speeds = []
    headings = []

    last_info = None

    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(zero_action)
        last_info = info
        speeds.append(float(np.linalg.norm(env.velocity)))
        headings.append(float(env.heading))
        if terminated or truncated:
            break

    displacement = float(np.linalg.norm(env.position - start_position))
    heading_change = abs(float(env.heading - start_heading))

    print(f"Steps executed                  : {len(speeds)}")
    print(f"Final displacement              : {displacement:.8f}")
    print(f"Mean speed                      : {_safe_mean(speeds):.8f}")
    print(f"Max speed                       : {max(speeds) if speeds else 0.0:.8f}")
    print(f"Heading change                  : {heading_change:.8f}")
    print(f"Mean heading                    : {_safe_mean(headings):.8f}")
    if last_info is not None:
        print(f"Ended stuck                     : {bool(last_info.get('stuck', False))}")
        print(f"No-progress steps               : {int(last_info.get('no_progress_steps', 0))}")

    env.close()


def constant_forward_test(
    steps: int = 100,
    target_heading: float = 0.0,
    target_speed: float = 1.0,
    seed: int = 123,
) -> None:
    _print_header("CONSTANT FORWARD TRACKING TEST")

    env = make_env()
    env.reset(seed=seed)

    env.obstacles = []
    env.position = np.array([2.0, 5.0], dtype=np.float32)
    env.goal = np.array([9.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.heading = 0.0
    env.angular_velocity = 0.0
    env.trajectory = [(float(env.position[0]), float(env.position[1]))]
    env.prev_distance_to_goal = env._distance_to_goal()
    env.best_distance_to_goal = env.prev_distance_to_goal
    env.no_progress_steps = 0

    action = np.array([target_heading, target_speed], dtype=np.float32)

    speeds = []
    headings = []
    lateral_errors = []
    last_info = None

    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        last_info = info

        speed = float(np.linalg.norm(env.velocity))
        speeds.append(speed)
        headings.append(float(env.heading))
        lateral_errors.append(float(abs(env.position[1] - 5.0)))

        if terminated or truncated:
            break

    print(f"Steps executed                  : {len(speeds)}")
    print(f"Final position                  : {env.position}")
    print(f"Mean speed                      : {_safe_mean(speeds):.6f}")
    print(f"Std speed                       : {_safe_std(speeds):.6f}")
    print(f"Final speed                     : {float(np.linalg.norm(env.velocity)):.6f}")
    print(f"Mean heading                    : {_safe_mean(headings):.6f}")
    print(f"Final heading                   : {float(env.heading):.6f}")
    print(f"Mean lateral error              : {_safe_mean(lateral_errors):.6f}")
    if last_info is not None:
        print(f"Final distance to goal          : {float(last_info.get('distance_to_goal', 0.0)):.6f}")
        print(f"Ended stuck                     : {bool(last_info.get('stuck', False))}")

    env.close()


def sensor_continuity_test(
    heading_start: float = 0.0,
    heading_end: float = math.pi / 2.0,
    steps: int = 40,
    seed: int = 123,
) -> None:
    _print_header("SENSOR CONTINUITY TEST")

    env = make_env()
    env.reset(seed=seed)

    env.position = np.array([5.0, 5.0], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)
    env.angular_velocity = 0.0
    env.obstacles = [
        type("ObstacleLike", (), {"x": 7.0, "y": 5.0, "radius": 0.5})(),
        type("ObstacleLike", (), {"x": 5.0, "y": 7.0, "radius": 0.5})(),
    ]

    reading_deltas = []
    previous = None

    headings = np.linspace(heading_start, heading_end, steps)
    for h in headings:
        env.heading = float(h)
        readings = env._get_sensor_readings()

        if previous is not None:
            delta = float(np.linalg.norm(readings - previous))
            reading_deltas.append(delta)

        previous = readings.copy()

    print(f"Heading sweep start/end         : {heading_start:.6f} -> {heading_end:.6f}")
    print(f"Steps sampled                   : {steps}")
    print(f"Mean sensor delta               : {_safe_mean(reading_deltas):.6f}")
    print(f"Std sensor delta                : {_safe_std(reading_deltas):.6f}")
    print(f"Max sensor delta                : {max(reading_deltas) if reading_deltas else 0.0:.6f}")

    env.close()


def main() -> None:
    random_rollout_diagnostics(steps=3000, seed=123)
    zero_action_drift_test(steps=100, seed=123)
    constant_forward_test(steps=100, target_heading=0.0, target_speed=1.0, seed=123)
    sensor_continuity_test(heading_start=0.0, heading_end=math.pi / 2.0, steps=40, seed=123)


if __name__ == "__main__":
    main()