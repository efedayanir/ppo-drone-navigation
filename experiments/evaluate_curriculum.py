from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

MODEL_STAGE_ORDER = ["easy", "medium", "hard"]
EVAL_DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_vec_env(config: dict) -> DummyVecEnv:
    return DummyVecEnv([lambda: DroneNavEnv(config)])


def get_space_signature(config_path: str | Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = load_config(config_path)
    env = DroneNavEnv(config)
    try:
        return env.observation_space.shape, env.action_space.shape
    finally:
        env.close()


def load_model_and_env(
    model_path: str | Path,
    vecnorm_path: str | Path,
    config_path: str | Path,
) -> Tuple[PPO, VecNormalize]:
    config = load_config(config_path)
    base_env = make_vec_env(config)

    env = VecNormalize.load(str(vecnorm_path), base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    return model, env


def infer_outcome(info: dict) -> str:
    if bool(info.get("reached_goal", False)):
        return "success"
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("stuck", False)):
        return "stuck"
    if bool(info.get("truncated", False)):
        return "timeout"
    return "failed"


def evaluate_model_on_config(
    model: PPO,
    env: VecNormalize,
    episodes: int,
    seed_base: int,
    deterministic: bool,
) -> Dict[str, float]:
    outcome_counts = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "stuck": 0,
        "failed": 0,
    }

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    final_distances: List[float] = []
    path_lengths: List[float] = []
    path_efficiencies: List[float] = []

    reward_term_history: Dict[str, List[float]] = {key: [] for key in REWARD_TERM_KEYS}

    for episode_idx in range(episodes):
        env.seed(seed_base + episode_idx)
        obs = env.reset()

        done = False
        ep_reward = 0.0
        ep_len = 0
        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]
            last_info = info

            ep_reward += reward
            ep_len += 1

            reward_terms = info.get("reward_terms", {})
            for key in reward_term_history:
                reward_term_history[key].append(float(reward_terms.get(key, 0.0)))

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if last_info is None:
            outcome_counts["failed"] += 1
            continue

        outcome_counts[infer_outcome(last_info)] += 1

        final_distances.append(float(last_info.get("distance_to_goal", 0.0)))
        path_lengths.append(float(last_info.get("path_length", 0.0)))
        path_efficiencies.append(float(last_info.get("path_efficiency", 0.0)))

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def std(xs: List[float]) -> float:
        return float(np.std(xs)) if xs else 0.0

    denom = max(episodes, 1)
    results = {
        "episodes": float(episodes),
        "success_rate": outcome_counts["success"] / denom,
        "collision_rate": outcome_counts["collision"] / denom,
        "timeout_rate": outcome_counts["timeout"] / denom,
        "stuck_rate": outcome_counts["stuck"] / denom,
        "failed_rate": outcome_counts["failed"] / denom,
        "mean_reward": mean(episode_rewards),
        "std_reward": std(episode_rewards),
        "mean_episode_length": mean(episode_lengths),
        "std_episode_length": std(episode_lengths),
        "mean_final_distance": mean(final_distances),
        "mean_path_length": mean(path_lengths),
        "mean_path_efficiency": mean(path_efficiencies),
    }

    for key, values in reward_term_history.items():
        results[f"reward_term_{key}_mean"] = mean(values)

    return results


def print_result_block(
    model_name: str,
    eval_name: str,
    results: Dict[str, float],
) -> None:
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"EVAL ENV: {eval_name}")
    print("=" * 80)
    print(f"Episodes                : {int(results['episodes'])}")
    print(f"Success rate            : {results['success_rate']:.3f}")
    print(f"Collision rate          : {results['collision_rate']:.3f}")
    print(f"Timeout rate            : {results['timeout_rate']:.3f}")
    print(f"Stuck rate              : {results['stuck_rate']:.3f}")
    print(f"Failed rate             : {results['failed_rate']:.3f}")
    print(f"Mean reward             : {results['mean_reward']:.3f}")
    print(f"Std reward              : {results['std_reward']:.3f}")
    print(f"Mean episode length     : {results['mean_episode_length']:.3f}")
    print(f"Std episode length      : {results['std_episode_length']:.3f}")
    print(f"Mean final distance     : {results['mean_final_distance']:.3f}")
    print(f"Mean path length        : {results['mean_path_length']:.3f}")
    print(f"Mean path efficiency    : {results['mean_path_efficiency']:.3f}")

    print("\nReward term means:")
    for key in REWARD_TERM_KEYS:
        print(f"  {key:>12}: {results[f'reward_term_{key}_mean']:.6f}")


def save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_curriculum_transfer_violations(
    rows: List[Dict[str, object]],
    tolerance: float = 0.0,
) -> List[str]:
    lookup: Dict[tuple[str, str], float] = {}
    for row in rows:
        model_name = row.get("model_name")
        eval_name = row.get("eval_name")
        success_rate = row.get("success_rate")
        if model_name is None or eval_name is None or success_rate is None:
            continue
        try:
            lookup[(str(model_name), str(eval_name))] = float(success_rate)
        except (TypeError, ValueError):
            continue

    violations: List[str] = []

    for model_name in MODEL_STAGE_ORDER:
        available = [
            (eval_name, lookup[(model_name, eval_name)])
            for eval_name in EVAL_DIFFICULTY_ORDER
            if (model_name, eval_name) in lookup
        ]
        for (left_name, left_rate), (right_name, right_rate) in zip(available, available[1:]):
            if left_rate + tolerance < right_rate:
                violations.append(
                    f"{model_name} model success should not improve from {left_name} "
                    f"({left_rate:.3f}) to {right_name} ({right_rate:.3f})."
                )

    for eval_name in ("hard", "hard_eval"):
        available = [
            (model_name, lookup[(model_name, eval_name)])
            for model_name in MODEL_STAGE_ORDER
            if (model_name, eval_name) in lookup
        ]
        for (left_name, left_rate), (right_name, right_rate) in zip(available, available[1:]):
            if right_rate + tolerance < left_rate:
                violations.append(
                    f"{eval_name} eval success should not regress from {left_name} "
                    f"({left_rate:.3f}) to {right_name} ({right_rate:.3f})."
                )

    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate curriculum-stage PPO models across compatible environment configs."
    )

    parser.add_argument("--easy-config", type=str, default="configs/config_easy.yaml")
    parser.add_argument("--medium-config", type=str, default="configs/config_medium.yaml")
    parser.add_argument("--hard-config", type=str, default="configs/config_hard.yaml")
    parser.add_argument(
        "--hard-eval-config",
        type=str,
        default=None,
        help=(
            "Optional extra hard evaluation config. If omitted, only --hard-config is used. "
            "Use this for config_hard_2/3/4/etc. when observation/action spaces are compatible."
        ),
    )

    parser.add_argument(
        "--curriculum-root",
        type=str,
        default="results/curriculum/ppo_curriculum_stable",
        help="Root folder created by train_curriculum.py",
    )

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV path for summarized results.",
    )
    parser.add_argument(
        "--check-transfer",
        action="store_true",
        help=(
            "Warn if success rates violate the expected curriculum transfer patterns: "
            "hard >= medium >= easy on hard evals, and easier envs >= harder envs per model."
        ),
    )
    parser.add_argument(
        "--transfer-tolerance",
        type=float,
        default=0.0,
        help="Allowed success-rate slack for --check-transfer comparisons.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    curriculum_root = Path(args.curriculum_root)

    stage_specs = {
        "easy": {
            "model_path": curriculum_root / "easy" / "models" / "easy_final_model.zip",
            "vecnorm_path": curriculum_root / "easy" / "vecnormalize" / "easy_vecnormalize.pkl",
            "train_config": args.easy_config,
        },
        "medium": {
            "model_path": curriculum_root / "medium" / "models" / "medium_final_model.zip",
            "vecnorm_path": curriculum_root / "medium" / "vecnormalize" / "medium_vecnormalize.pkl",
            "train_config": args.medium_config,
        },
        "hard": {
            "model_path": curriculum_root / "hard" / "models" / "hard_final_model.zip",
            "vecnorm_path": curriculum_root / "hard" / "vecnormalize" / "hard_vecnormalize.pkl",
            "train_config": args.hard_config,
        },
    }

    eval_configs = {
        "easy": args.easy_config,
        "medium": args.medium_config,
        "hard": args.hard_config,
    }

    if args.hard_eval_config is not None:
        eval_configs["hard_eval"] = args.hard_eval_config

    stage_signatures = {
        stage_name: get_space_signature(spec["train_config"])
        for stage_name, spec in stage_specs.items()
    }

    eval_signatures = {
        eval_name: get_space_signature(config_path)
        for eval_name, config_path in eval_configs.items()
    }

    summary_rows: List[Dict[str, object]] = []

    for model_name, spec in stage_specs.items():
        if not spec["model_path"].exists():
            print(f"\n[WARN] Missing model: {spec['model_path']}")
            continue
        if not spec["vecnorm_path"].exists():
            print(f"\n[WARN] Missing vecnorm stats: {spec['vecnorm_path']}")
            continue

        for eval_name, eval_config_path in eval_configs.items():
            if stage_signatures[model_name] != eval_signatures[eval_name]:
                print(
                    f"\n[WARN] Skipping {model_name} -> {eval_name}: "
                    "observation/action spaces are incompatible."
                )
                continue

            model, env = load_model_and_env(
                model_path=spec["model_path"],
                vecnorm_path=spec["vecnorm_path"],
                config_path=eval_config_path,
            )

            try:
                results = evaluate_model_on_config(
                    model=model,
                    env=env,
                    episodes=args.episodes,
                    seed_base=args.seed,
                    deterministic=args.deterministic,
                )
            finally:
                env.close()

            print_result_block(
                model_name=model_name,
                eval_name=eval_name,
                results=results,
            )

            row: Dict[str, object] = {
                "model_name": model_name,
                "eval_name": eval_name,
                "train_config": str(spec["train_config"]),
                "eval_config": str(eval_config_path),
            }
            row.update(results)
            summary_rows.append(row)

    if args.output_csv is not None:
        save_summary_csv(Path(args.output_csv), summary_rows)
        print(f"\nSaved summary CSV: {args.output_csv}")

    if args.check_transfer:
        violations = find_curriculum_transfer_violations(
            summary_rows,
            tolerance=args.transfer_tolerance,
        )
        if violations:
            print("\n[WARN] Curriculum transfer checks found potential regressions:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("\nCurriculum transfer checks passed.")


if __name__ == "__main__":
    main()
