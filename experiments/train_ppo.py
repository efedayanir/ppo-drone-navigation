from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.drone_env import DroneNavEnv


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_fn(config: dict, rank: int, base_seed: int) -> Callable[[], Monitor]:
    def _make() -> Monitor:
        env = DroneNavEnv(config)
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _make


def build_vec_env(config: dict, n_envs: int, base_seed: int) -> DummyVecEnv:
    env_fns = [make_env_fn(config, rank=i, base_seed=base_seed) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


class NormalizationSyncEvalCallback(EvalCallback):
    """
    Sync VecNormalize stats before evaluation.
    If a new best model is found, also save the matching train-env VecNormalize
    snapshot so `best_model.zip` and `best_vecnormalize.pkl` stay paired.
    """

    def __init__(self, *args, best_vecnorm_save_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_vecnorm_save_path = best_vecnorm_save_path

    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model is not None and self.model.get_env() is not None:
                sync_envs_normalization(self.model.get_env(), self.eval_env)

        continue_training = super()._on_step()

        found_new_best = self.best_mean_reward > previous_best
        if found_new_best and self.best_vecnorm_save_path is not None:
            train_env = None if self.model is None else self.model.get_env()
            if train_env is not None:
                Path(self.best_vecnorm_save_path).parent.mkdir(parents=True, exist_ok=True)
                train_env.save(self.best_vecnorm_save_path)
                print(f"Saved best VecNormalize stats to: {self.best_vecnorm_save_path}")

        return continue_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable PPO training for DroneNavEnv")

    parser.add_argument("--config", type=str, default="config_easy.yaml")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="ppo_stable_run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seeds(args.seed)

    project_root = Path(__file__).resolve().parents[1]

    results_dir = project_root / "results"
    models_dir = results_dir / "models" / args.run_name
    logs_dir = results_dir / "logs"
    eval_dir = results_dir / "eval" / args.run_name
    vecnorm_dir = results_dir / "vecnormalize" / args.run_name

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    vecnorm_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(config, n_envs=args.n_envs, base_seed=args.seed)
    train_env = VecNormalize(
        train_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    eval_env = build_vec_env(config, n_envs=1, base_seed=args.seed + 10_000)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=str(logs_dir),
        learning_rate=2e-5,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.002,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=args.seed,
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(models_dir / "checkpoints"),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    best_vecnorm_path = vecnorm_dir / "best_vecnormalize.pkl"

    eval_callback = NormalizationSyncEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        best_vecnorm_save_path=str(best_vecnorm_path),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"Training with config : {args.config}")
    print(f"Run name             : {args.run_name}")
    print(f"Seed                 : {args.seed}")
    print(f"Timesteps            : {args.timesteps}")
    print(f"Parallel envs        : {args.n_envs}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    final_model_path = models_dir / "final_model.zip"
    vecnorm_path = vecnorm_dir / "vecnormalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(vecnorm_path))

    print(f"Training complete. Final model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {vecnorm_path}")
    print(f"Best-model VecNormalize stats saved to: {best_vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
