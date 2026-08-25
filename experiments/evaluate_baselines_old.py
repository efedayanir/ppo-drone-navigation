from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.drone_env import DroneNavEnv


REWARD_TERM_KEYS = [
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
]


@dataclass
class EpisodeResult:
    policy: str
    episode: int
    outcome: str
    total_reward: float
    length: int
    final_distance: float
    path_length: float
    path_efficiency: float
    collision: bool
    reached_goal: bool
    truncated: bool
    stuck: bool
    final_x: float
    final_y: float


class BaselinePolicy:
    name = "base"

    def act(self, env: DroneNavEnv, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    name = "random"

    def act(self, env: DroneNavEnv, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [
                rng.uniform(env.action_space.low[0], env.action_space.high[0]),
                rng.uniform(env.action_space.low[1], env.action_space.high[1]),
            ],
            dtype=np.float32,
        )


class GreedyGoalPolicy(BaselinePolicy):
    name = "greedy_goal"

    def __init__(self, speed_scale: float = 1.0):
        self.speed_scale = float(speed_scale)

    def act(self, env: DroneNavEnv, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        goal_vec = env.goal - env.position
        heading = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        speed = env.max_speed * self.speed_scale
        return np.array([heading, speed], dtype=np.float32)


class ObstacleAwarePolicy(BaselinePolicy):
    name = "obstacle_aware"

    def __init__(
        self,
        speed_scale: float = 0.9,
        repulsion_gain: float = 1.25,
        sensor_threshold: float = 0.65,
        side_bias: float = 0.35,
    ):
        self.speed_scale = float(speed_scale)
        self.repulsion_gain = float(repulsion_gain)
        self.sensor_threshold = float(sensor_threshold)
        self.side_bias = float(side_bias)

    def act(self, env: DroneNavEnv, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        goal_vec = env.goal - env.position
        goal_norm = float(np.linalg.norm(goal_vec))

        if goal_norm > 1e-8:
            desired = goal_vec / goal_norm
        else:
            desired = np.array([math.cos(env.heading), math.sin(env.heading)], dtype=np.float32)

        readings = env._get_sensor_readings()
        repulsion = np.zeros(2, dtype=np.float32)

        for i, reading in enumerate(readings):
            # readings are normalized to [0, 1] in current configs.
            normalized_reading = float(reading)
            if not env.normalize_sensor_readings:
                normalized_reading = normalized_reading / max(env.sensor_max_range, 1e-8)

            if normalized_reading >= self.sensor_threshold:
                continue

            ray_angle = env.heading + (2.0 * math.pi * i / env.num_rays)
            ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)], dtype=np.float32)

            closeness = (self.sensor_threshold - normalized_reading) / max(self.sensor_threshold, 1e-8)
            repulsion -= ray_dir * closeness * self.repulsion_gain

        combined = desired + repulsion

        # If obstacle repulsion nearly cancels the goal vector, turn sideways instead of freezing.
        if float(np.linalg.norm(combined)) < 1e-6:
            combined = np.array(
                [math.cos(env.heading + self.side_bias), math.sin(env.heading + self.side_bias)],
                dtype=np.float32,
            )

        heading = math.atan2(float(combined[1]), float(combined[0]))

        min_reading = float(np.min(readings)) if len(readings) else 1.0
        if not env.normalize_sensor_readings:
            min_reading = min_reading / max(env.sensor_max_range, 1e-8)

        # Slow down near obstacles, but do not stop completely unless extremely close.
        obstacle_speed_factor = np.clip(0.35 + min_reading, 0.25, 1.0)
        speed = env.max_speed * self.speed_scale * float(obstacle_speed_factor)

        return np.array([heading, speed], dtype=np.float32)


class WallAvoidingGreedyPolicy(BaselinePolicy):
    name = "wall_avoiding_greedy"

    def __init__(self, speed_scale: float = 0.9, wall_margin: float = 0.8):
        self.speed_scale = float(speed_scale)
        self.wall_margin = float(wall_margin)

    def act(self, env: DroneNavEnv, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        goal_vec = env.goal - env.position
        goal_norm = float(np.linalg.norm(goal_vec))

        if goal_norm > 1e-8:
            desired = goal_vec / goal_norm
        else:
            desired = np.array([math.cos(env.heading), math.sin(env.heading)], dtype=np.float32)

        wall_push = np.zeros(2, dtype=np.float32)
        x, y = float(env.position[0]), float(env.position[1])

        if x < self.wall_margin:
            wall_push[0] += (self.wall_margin - x) / self.wall_margin
        if env.width - x < self.wall_margin:
            wall_push[0] -= (self.wall_margin - (env.width - x)) / self.wall_margin
        if y < self.wall_margin:
            wall_push[1] += (self.wall_margin - y) / self.wall_margin
        if env.height - y < self.wall_margin:
            wall_push[1] -= (self.wall_margin - (env.height - y)) / self.wall_margin

        combined = desired + 1.5 * wall_push
        heading = math.atan2(float(combined[1]), float(combined[0]))
        speed = env.max_speed * self.speed_scale

        return np.array([heading, speed], dtype=np.float32)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_outcome(info: Dict[str, Any]) -> str:
    if bool(info.get("reached_goal", False)):
        return "success"
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("stuck", False)):
        return "stuck"
    if bool(info.get("truncated", False)):
        return "timeout"
    return "failed"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_episode(
    policy: BaselinePolicy,
    config: dict,
    episode_idx: int,
    seed: int,
) -> tuple[EpisodeResult, List[Dict[str, Any]]]:
    env = DroneNavEnv(config)
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    length = 0
    rows: List[Dict[str, Any]] = []
    last_info: Dict[str, Any] = {}

    while not done:
        action = policy.act(env=env, obs=obs, rng=rng)
        obs, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated or truncated)
        total_reward += float(reward)
        length += 1
        last_info = info

        row: Dict[str, Any] = {
            "policy": policy.name,
            "episode": episode_idx,
            "step": length,
            "reward": float(reward),
            "done": int(done),
            "outcome_so_far": infer_outcome(info) if done else "running",
            "action_heading": float(action[0]),
            "action_speed": float(action[1]),
            "collision": int(bool(info.get("collision", False))),
            "reached_goal": int(bool(info.get("reached_goal", False))),
            "truncated": int(bool(info.get("truncated", False))),
            "stuck": int(bool(info.get("stuck", False))),
            "distance_to_goal": safe_float(info.get("distance_to_goal")),
            "path_length": safe_float(info.get("path_length")),
            "path_efficiency": safe_float(info.get("path_efficiency")),
            "speed": safe_float(info.get("speed")),
            "agent_x": safe_float(info.get("agent_x")),
            "agent_y": safe_float(info.get("agent_y")),
            "goal_x": safe_float(info.get("goal_x")),
            "goal_y": safe_float(info.get("goal_y")),
            "no_progress_steps": int(info.get("no_progress_steps", 0)),
            "best_distance_to_goal": safe_float(info.get("best_distance_to_goal")),
        }

        reward_terms = info.get("reward_terms", {}) or {}
        for key in REWARD_TERM_KEYS:
            row[f"reward_term_{key}"] = safe_float(reward_terms.get(key), 0.0)

        rows.append(row)

    outcome = infer_outcome(last_info)

    result = EpisodeResult(
        policy=policy.name,
        episode=episode_idx,
        outcome=outcome,
        total_reward=total_reward,
        length=length,
        final_distance=safe_float(last_info.get("distance_to_goal")),
        path_length=safe_float(last_info.get("path_length")),
        path_efficiency=safe_float(last_info.get("path_efficiency")),
        collision=bool(last_info.get("collision", False)),
        reached_goal=bool(last_info.get("reached_goal", False)),
        truncated=bool(last_info.get("truncated", False)),
        stuck=bool(last_info.get("stuck", False)),
        final_x=safe_float(last_info.get("agent_x")),
        final_y=safe_float(last_info.get("agent_y")),
    )

    env.close()
    return result, rows


def summarize_policy(policy_name: str, results: List[EpisodeResult], step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(len(results), 1)

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    def std(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.std(values)) if values else 0.0

    summary: Dict[str, Any] = {
        "policy": policy_name,
        "episodes": len(results),
        "success_rate": sum(r.outcome == "success" for r in results) / n,
        "collision_rate": sum(r.outcome == "collision" for r in results) / n,
        "timeout_rate": sum(r.outcome == "timeout" for r in results) / n,
        "stuck_rate": sum(r.outcome == "stuck" for r in results) / n,
        "failed_rate": sum(r.outcome == "failed" for r in results) / n,
        "mean_reward": mean(r.total_reward for r in results),
        "std_reward": std(r.total_reward for r in results),
        "mean_episode_length": mean(r.length for r in results),
        "std_episode_length": std(r.length for r in results),
        "mean_final_distance": mean(r.final_distance for r in results),
        "mean_path_length": mean(r.path_length for r in results),
        "mean_path_efficiency": mean(r.path_efficiency for r in results),
    }

    for key in REWARD_TERM_KEYS:
        summary[f"reward_term_{key}_mean"] = mean(
            safe_float(row.get(f"reward_term_{key}")) for row in step_rows if row["policy"] == policy_name
        )

    return summary


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"BASELINE: {summary['policy']}")
    print("=" * 80)
    print(f"Episodes                : {summary['episodes']}")
    print(f"Success rate            : {summary['success_rate']:.3f}")
    print(f"Collision rate          : {summary['collision_rate']:.3f}")
    print(f"Timeout rate            : {summary['timeout_rate']:.3f}")
    print(f"Stuck rate              : {summary['stuck_rate']:.3f}")
    print(f"Failed rate             : {summary['failed_rate']:.3f}")
    print(f"Mean reward             : {summary['mean_reward']:.3f}")
    print(f"Std reward              : {summary['std_reward']:.3f}")
    print(f"Mean episode length     : {summary['mean_episode_length']:.3f}")
    print(f"Mean final distance     : {summary['mean_final_distance']:.3f}")
    print(f"Mean path length        : {summary['mean_path_length']:.3f}")
    print(f"Mean path efficiency    : {summary['mean_path_efficiency']:.3f}")


def save_outcome_plot(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    policies = [s["policy"] for s in summaries]
    outcomes = ["success_rate", "collision_rate", "timeout_rate", "stuck_rate"]
    x = np.arange(len(policies))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, outcome in enumerate(outcomes):
        values = [float(s[outcome]) for s in summaries]
        ax.bar(x + (i - 1.5) * width, values, width, label=outcome.replace("_rate", ""))

    ax.set_title("Baseline Outcome Breakdown")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_efficiency_plot(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    policies = [s["policy"] for s in summaries]
    values = [float(s["mean_path_efficiency"]) for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(policies, values)
    ax.set_title("Baseline Mean Path Efficiency")
    ax.set_ylabel("Mean path efficiency")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_policies(names: List[str]) -> List[BaselinePolicy]:
    available: Dict[str, BaselinePolicy] = {
        "random": RandomPolicy(),
        "greedy_goal": GreedyGoalPolicy(),
        "obstacle_aware": ObstacleAwarePolicy(),
        "wall_avoiding_greedy": WallAvoidingGreedyPolicy(),
    }

    policies = []
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown policy '{name}'. Available: {sorted(available)}")
        policies.append(available[name])

    return policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-learning baseline policies on DroneNavEnv."
    )

    parser.add_argument("--config", required=True, help="Path to environment config YAML.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "greedy_goal", "obstacle_aware", "wall_avoiding_greedy"],
        help="Baseline policies to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = build_policies(args.policies)

    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for policy in policies:
        print(f"\nEvaluating baseline: {policy.name}")

        policy_results: List[EpisodeResult] = []
        policy_step_rows: List[Dict[str, Any]] = []

        for episode_idx in range(args.episodes):
            result, step_rows = run_episode(
                policy=policy,
                config=config,
                episode_idx=episode_idx,
                seed=args.seed + episode_idx,
            )

            policy_results.append(result)
            policy_step_rows.extend(step_rows)

            all_episode_rows.append(result.__dict__)
            all_step_rows.extend(step_rows)

            print(
                f"[{episode_idx + 1:04d}/{args.episodes:04d}] "
                f"{policy.name:<20} {result.outcome:<9} "
                f"reward={result.total_reward:8.2f} "
                f"steps={result.length:4d} "
                f"eff={result.path_efficiency:.3f} "
                f"dist={result.final_distance:.3f}"
            )

        summary = summarize_policy(policy.name, policy_results, policy_step_rows)
        summaries.append(summary)
        print_summary(summary)

        save_csv(output_dir / f"{policy.name}_episodes.csv", [r.__dict__ for r in policy_results])
        save_csv(output_dir / f"{policy.name}_steps.csv", policy_step_rows)

    save_csv(output_dir / "baseline_episodes.csv", all_episode_rows)
    save_csv(output_dir / "baseline_steps.csv", all_step_rows)
    save_csv(output_dir / "baseline_summary.csv", summaries)

    save_outcome_plot(output_dir / "baseline_outcome_breakdown.png", summaries)
    save_efficiency_plot(output_dir / "baseline_path_efficiency.png", summaries)

    print("\nSaved baseline outputs:")
    print(f"  {output_dir / 'baseline_summary.csv'}")
    print(f"  {output_dir / 'baseline_episodes.csv'}")
    print(f"  {output_dir / 'baseline_steps.csv'}")
    print(f"  {output_dir / 'baseline_outcome_breakdown.png'}")
    print(f"  {output_dir / 'baseline_path_efficiency.png'}")


if __name__ == "__main__":
    main()
