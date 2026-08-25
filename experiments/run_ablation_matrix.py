from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.drone_env import DroneNavEnv
from experiments.evaluate_baselines import (
    REWARD_TERM_KEYS,
    BaselinePolicy,
    build_policies,
    infer_outcome,
    run_episode,
    safe_float,
    save_csv,
    summarize_policy,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    category: str
    description: str
    requires_retraining: bool
    evaluate_by_default: bool
    mutate_config: Callable[[dict], None] | None = None


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clone_config(config: dict) -> dict:
    return copy.deepcopy(config)


def _mutate_progress_distance_normalized(config: dict) -> None:
    shaping = config["reward"].setdefault("shaping", {})
    shaping.pop("progress_normalizer", None)


def _mutate_progress_fixed_normalizer(config: dict) -> None:
    config["reward"].setdefault("shaping", {})["progress_normalizer"] = 1.0


def _mutate_stuck_disabled(config: dict) -> None:
    config["reward"]["stuck_patience"] = 10_000_000
    config["reward"]["stuck_penalty"] = 0.0


def _mutate_previous_action_observation(config: dict) -> None:
    config["environment"].setdefault("observation", {})["include_previous_action"] = True


def _mutate_obstacle_feature_observation(config: dict) -> None:
    observation_cfg = config["environment"].setdefault("observation", {})
    observation_cfg["include_nearest_obstacle_distance"] = True
    observation_cfg["include_nearest_obstacle_bearing"] = True
    observation_cfg["include_min_sensor_distance"] = True


def _mutate_frame_stack(config: dict) -> None:
    config.setdefault("training", {})["frame_stack"] = 4


def build_ablation_specs() -> Dict[str, AblationSpec]:
    specs = [
        AblationSpec(
            name="control",
            category="baseline",
            description="Original config with no ablation.",
            requires_retraining=False,
            evaluate_by_default=True,
            mutate_config=None,
        ),
        AblationSpec(
            name="stuck_disabled_eval",
            category="termination",
            description="Eval-only diagnostic: disables practical stuck termination by using very high patience.",
            requires_retraining=False,
            evaluate_by_default=True,
            mutate_config=_mutate_stuck_disabled,
        ),
        AblationSpec(
            name="progress_fixed_normalizer_retrain",
            category="reward",
            description="Training ablation: force one meter of physical progress to have a fixed reward scale.",
            requires_retraining=True,
            evaluate_by_default=False,
            mutate_config=_mutate_progress_fixed_normalizer,
        ),
        AblationSpec(
            name="progress_distance_normalized_retrain",
            category="reward",
            description="Training ablation: remove fixed progress_normalizer so reward scales by previous distance.",
            requires_retraining=True,
            evaluate_by_default=False,
            mutate_config=_mutate_progress_distance_normalized,
        ),
        AblationSpec(
            name="previous_action_observation_retrain",
            category="observation",
            description="Training ablation: expose previous action in the policy observation.",
            requires_retraining=True,
            evaluate_by_default=False,
            mutate_config=_mutate_previous_action_observation,
        ),
        AblationSpec(
            name="obstacle_feature_observation_retrain",
            category="observation",
            description="Training ablation: expose nearest-obstacle and min-ray summary features.",
            requires_retraining=True,
            evaluate_by_default=False,
            mutate_config=_mutate_obstacle_feature_observation,
        ),
        AblationSpec(
            name="frame_stack_memory_retrain",
            category="memory",
            description="Training ablation: evaluate whether short observation history helps local minima.",
            requires_retraining=True,
            evaluate_by_default=False,
            mutate_config=_mutate_frame_stack,
        ),
    ]
    return {spec.name: spec for spec in specs}


def apply_ablation(config: dict, spec: AblationSpec) -> dict:
    ablated = clone_config(config)
    if spec.mutate_config is not None:
        spec.mutate_config(ablated)
    return ablated


def spec_to_manifest_row(spec: AblationSpec) -> Dict[str, Any]:
    return {
        "ablation": spec.name,
        "category": spec.category,
        "description": spec.description,
        "requires_retraining": int(spec.requires_retraining),
        "evaluate_by_default": int(spec.evaluate_by_default),
    }


def make_skip_row(spec: AblationSpec, reason: str) -> Dict[str, Any]:
    return {
        "ablation": spec.name,
        "category": spec.category,
        "policy": "not_evaluated",
        "evaluated": 0,
        "skip_reason": reason,
        "requires_retraining": int(spec.requires_retraining),
        "episodes": 0,
    }


def summarize_baseline_on_config(
    policy: BaselinePolicy,
    config: dict,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    policy_results = []
    policy_step_rows: List[Dict[str, Any]] = []

    for episode_idx in range(episodes):
        result, step_rows = run_episode(
            policy=policy,
            config=config,
            episode_idx=episode_idx,
            seed=seed + episode_idx,
        )
        policy_results.append(result)
        policy_step_rows.extend(step_rows)

    return summarize_policy(policy.name, policy_results, policy_step_rows)


def evaluate_ppo_on_config(
    config: dict,
    model_path: Path,
    vecnorm_path: Path,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from experiments.evaluate import validate_model_vecnorm_pair

    validate_model_vecnorm_pair(model_path, vecnorm_path)

    base_env = DummyVecEnv([lambda: DroneNavEnv(config)])
    env = VecNormalize.load(str(vecnorm_path), base_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path), env=env)

    outcome_counts = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "stuck": 0,
        "failed": 0,
    }
    rewards: List[float] = []
    lengths: List[int] = []
    final_distances: List[float] = []
    path_lengths: List[float] = []
    path_efficiencies: List[float] = []
    reward_term_history: Dict[str, List[float]] = {key: [] for key in REWARD_TERM_KEYS}

    try:
        for episode_idx in range(episodes):
            env.seed(seed + episode_idx)
            obs = env.reset()
            done = False
            total_reward = 0.0
            length = 0
            last_info: Dict[str, Any] = {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = env.step(action)

                done = bool(dones[0])
                total_reward += float(reward[0])
                length += 1
                last_info = infos[0]

                reward_terms = last_info.get("reward_terms", {}) or {}
                for key in REWARD_TERM_KEYS:
                    reward_term_history[key].append(safe_float(reward_terms.get(key), 0.0))

            outcome_counts[infer_outcome(last_info)] += 1
            rewards.append(total_reward)
            lengths.append(length)
            final_distances.append(safe_float(last_info.get("distance_to_goal")))
            path_lengths.append(safe_float(last_info.get("path_length")))
            path_efficiencies.append(safe_float(last_info.get("path_efficiency")))
    finally:
        env.close()

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    def std(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.std(values)) if values else 0.0

    denom = max(episodes, 1)
    summary: Dict[str, Any] = {
        "policy": "ppo",
        "episodes": episodes,
        "success_rate": outcome_counts["success"] / denom,
        "collision_rate": outcome_counts["collision"] / denom,
        "timeout_rate": outcome_counts["timeout"] / denom,
        "stuck_rate": outcome_counts["stuck"] / denom,
        "failed_rate": outcome_counts["failed"] / denom,
        "mean_reward": mean(rewards),
        "std_reward": std(rewards),
        "mean_episode_length": mean(lengths),
        "std_episode_length": std(lengths),
        "mean_final_distance": mean(final_distances),
        "mean_path_length": mean(path_lengths),
        "mean_path_efficiency": mean(path_efficiencies),
    }

    for key, values in reward_term_history.items():
        summary[f"reward_term_{key}_mean"] = mean(values)

    return summary


def decorate_summary(
    summary: Dict[str, Any],
    spec: AblationSpec,
    config_path: Path,
) -> Dict[str, Any]:
    row = {
        "ablation": spec.name,
        "category": spec.category,
        "description": spec.description,
        "config": str(config_path),
        "evaluated": 1,
        "skip_reason": "",
        "requires_retraining": int(spec.requires_retraining),
    }
    row.update(summary)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact PPO-vs-heuristic ablation matrix for DroneNavEnv."
    )
    parser.add_argument("--config", default="configs/config_hard.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", default="results/ablation_matrix")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["greedy_goal", "obstacle_aware", "wall_avoiding_greedy"],
        help="Handcrafted baselines to evaluate.",
    )
    parser.add_argument("--ppo-model", default=None, help="Optional PPO .zip path.")
    parser.add_argument("--ppo-vecnorm", default=None, help="Optional VecNormalize .pkl path.")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Ablations to include. Defaults to eval-safe ablations plus skipped retrain-required manifest rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(config_path)
    specs = build_ablation_specs()

    selected_names = args.ablations
    if selected_names is None:
        selected_names = [name for name, spec in specs.items() if spec.evaluate_by_default]
        selected_names += [name for name, spec in specs.items() if spec.requires_retraining]

    unknown = [name for name in selected_names if name not in specs]
    if unknown:
        raise ValueError(f"Unknown ablations {unknown}. Available: {sorted(specs)}")

    policies = build_policies(args.policies)
    ppo_model = Path(args.ppo_model) if args.ppo_model else None
    ppo_vecnorm = Path(args.ppo_vecnorm) if args.ppo_vecnorm else None

    if (ppo_model is None) != (ppo_vecnorm is None):
        raise ValueError("Pass both --ppo-model and --ppo-vecnorm, or neither.")

    manifest_rows = [spec_to_manifest_row(specs[name]) for name in selected_names]
    summary_rows: List[Dict[str, Any]] = []

    for name in selected_names:
        spec = specs[name]
        if spec.requires_retraining:
            summary_rows.append(
                make_skip_row(
                    spec,
                    "Requires retraining before PPO-vs-baseline comparison is scientifically valid.",
                )
            )
            continue

        ablated_config = apply_ablation(base_config, spec)
        print(f"\nAblation: {spec.name}")

        for policy in policies:
            print(f"  baseline: {policy.name}")
            summary = summarize_baseline_on_config(
                policy=policy,
                config=ablated_config,
                episodes=args.episodes,
                seed=args.seed,
            )
            summary_rows.append(decorate_summary(summary, spec, config_path))

        if ppo_model is not None and ppo_vecnorm is not None:
            print("  policy: ppo")
            summary = evaluate_ppo_on_config(
                config=ablated_config,
                model_path=ppo_model,
                vecnorm_path=ppo_vecnorm,
                episodes=args.episodes,
                seed=args.seed,
                deterministic=args.deterministic,
            )
            summary_rows.append(decorate_summary(summary, spec, config_path))

    save_csv(output_dir / "ablation_manifest.csv", manifest_rows)
    save_csv(output_dir / "ablation_summary.csv", summary_rows)

    print("\nSaved ablation outputs:")
    print(f"  {output_dir / 'ablation_manifest.csv'}")
    print(f"  {output_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
