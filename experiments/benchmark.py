from __future__ import annotations

import argparse
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


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config: dict) -> DummyVecEnv:
    return DummyVecEnv([lambda: DroneNavEnv(config)])


def load_model_env(
    model_path: str,
    vecnorm_path: str,
    config_path: str,
) -> Tuple[PPO, VecNormalize]:
    config = load_config(config_path)
    base_env = make_env(config)

    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    return model, env


def evaluate(
    model: PPO,
    env: VecNormalize,
    episodes: int,
    seed_base: int,
    deterministic: bool,
) -> Dict[str, float]:
    successes = 0
    collisions = 0
    timeouts = 0
    stucks = 0

    rewards: List[float] = []
    lengths: List[int] = []
    efficiencies: List[float] = []
    final_distances: List[float] = []

    reward_term_history: Dict[str, List[float]] = {
        "collision": [],
        "goal": [],
        "progress": [],
        "step": [],
        "smoothness": [],
        "stall": [],
        "alignment": [],
        "timeout": [],
        "low_speed": [],
        "stuck": [],
        "total": [],
    }

    for ep in range(episodes):
        env.seed(seed_base + ep)
        obs = env.reset()
        done = False

        ep_reward = 0.0
        ep_len = 0
        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, d, infos = env.step(action)

            ep_reward += float(r[0])
            ep_len += 1
            done = bool(d[0])
            last_info = infos[0]

            reward_terms = last_info.get("reward_terms", {})
            for key in reward_term_history:
                reward_term_history[key].append(float(reward_terms.get(key, 0.0)))

        rewards.append(ep_reward)
        lengths.append(ep_len)

        if last_info:
            if last_info.get("reached_goal", False):
                successes += 1
            elif last_info.get("collision", False):
                collisions += 1
            elif last_info.get("stuck", False):
                stucks += 1
            else:
                timeouts += 1

            efficiencies.append(float(last_info.get("path_efficiency", 0.0)))
            final_distances.append(float(last_info.get("distance_to_goal", 0.0)))

    def mean(x: List[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    results = {
        "success_rate": successes / max(episodes, 1),
        "collision_rate": collisions / max(episodes, 1),
        "timeout_rate": timeouts / max(episodes, 1),
        "stuck_rate": stucks / max(episodes, 1),
        "mean_reward": mean(rewards),
        "mean_length": mean(lengths),
        "mean_efficiency": mean(efficiencies),
        "mean_final_distance": mean(final_distances),
    }

    for key, values in reward_term_history.items():
        results[f"reward_term_{key}_mean"] = mean(values)

    return results


def print_comparison(name: str, results: Dict[str, float]) -> None:
    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print(f"Success rate         : {results['success_rate']:.3f}")
    print(f"Collision rate       : {results['collision_rate']:.3f}")
    print(f"Timeout rate         : {results['timeout_rate']:.3f}")
    print(f"Stuck rate           : {results['stuck_rate']:.3f}")
    print(f"Mean reward          : {results['mean_reward']:.3f}")
    print(f"Mean length          : {results['mean_length']:.2f}")
    print(f"Path efficiency      : {results['mean_efficiency']:.3f}")
    print(f"Mean final distance  : {results['mean_final_distance']:.3f}")

    print("\nReward term means:")
    for key in [
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
    ]:
        print(f"  {key:>12}: {results[f'reward_term_{key}_mean']:.6f}")


def parse_args():
    p = argparse.ArgumentParser("Benchmark curriculum vs direct training")

    p.add_argument("--hard-config", default="config_hard.yaml")

    p.add_argument(
        "--curriculum-model",
        default="results/curriculum/ppo_curriculum_stable/hard/models/hard_final_model.zip",
    )
    p.add_argument(
        "--curriculum-vecnorm",
        default="results/curriculum/ppo_curriculum_stable/hard/vecnormalize/hard_vecnormalize.pkl",
    )

    p.add_argument(
        "--direct-model",
        default="results/models/ppo_stable_hard/final_model.zip",
    )
    p.add_argument(
        "--direct-vecnorm",
        default="results/vecnormalize/ppo_stable_hard/vecnormalize.pkl",
    )

    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if Path(args.curriculum_model).exists() and Path(args.curriculum_vecnorm).exists():
        model_c, env_c = load_model_env(
            args.curriculum_model,
            args.curriculum_vecnorm,
            args.hard_config,
        )

        res_c = evaluate(
            model_c,
            env_c,
            args.episodes,
            args.seed,
            args.deterministic,
        )

        print_comparison("CURRICULUM MODEL (EASY -> MEDIUM -> HARD)", res_c)
        env_c.close()
    else:
        print("[WARN] Curriculum model or vecnorm not found.")

    if Path(args.direct_model).exists() and Path(args.direct_vecnorm).exists():
        model_d, env_d = load_model_env(
            args.direct_model,
            args.direct_vecnorm,
            args.hard_config,
        )

        res_d = evaluate(
            model_d,
            env_d,
            args.episodes,
            args.seed,
            args.deterministic,
        )

        print_comparison("DIRECT HARD TRAINING", res_d)
        env_d.close()
    else:
        print("[WARN] Direct model or vecnorm not found.")


if __name__ == "__main__":
    main()
