from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable, Optional

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
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model is not None and self.model.get_env() is not None:
                sync_envs_normalization(self.model.get_env(), self.eval_env)
        return super()._on_step()


def create_train_env(
    config: dict,
    n_envs: int,
    seed: int,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    base_env = build_vec_env(config=config, n_envs=n_envs, base_seed=seed)

    if vecnorm_path is not None and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = True
        env.norm_obs = True
        env.norm_reward = True
        env.clip_obs = 10.0
        env.clip_reward = 10.0
        return env

    return VecNormalize(
        base_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )


def create_eval_env(
    config: dict,
    seed: int,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    base_env = build_vec_env(config=config, n_envs=1, base_seed=seed)

    if vecnorm_path is not None and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False
        env.clip_obs = 10.0
        return env

    return VecNormalize(
        base_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
    )


def build_model(
    train_env: VecNormalize,
    seed: int,
    device: str,
    model_path: Optional[str] = None,
) -> PPO:
    if model_path is not None and Path(model_path).exists():
        return PPO.load(model_path, env=train_env, device=device)

    return PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=1e-4,
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
        seed=seed,
        device=device,
    )


def train_stage(
    stage_name: str,
    config_path: str,
    timesteps: int,
    run_root: Path,
    n_envs: int,
    seed: int,
    device: str,
    incoming_model_path: Optional[str],
    incoming_vecnorm_path: Optional[str],
    eval_freq: int,
    eval_episodes: int,
    checkpoint_freq: int,
) -> tuple[str, str]:
    print("\n" + "=" * 80)
    print(f"TRAINING STAGE: {stage_name}")
    print("=" * 80)

    config = load_config(config_path)

    stage_dir = run_root / stage_name
    models_dir = stage_dir / "models"
    eval_dir = stage_dir / "eval"
    vecnorm_dir = stage_dir / "vecnormalize"
    tb_dir = stage_dir / "tensorboard"

    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    vecnorm_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_env = create_train_env(
        config=config,
        n_envs=n_envs,
        seed=seed,
        vecnorm_path=incoming_vecnorm_path,
    )

    eval_env = create_eval_env(
        config=config,
        seed=seed + 10_000,
        vecnorm_path=incoming_vecnorm_path,
    )

    model = build_model(
        train_env=train_env,
        seed=seed,
        device=device,
        model_path=incoming_model_path,
    )
    model.tensorboard_log = str(tb_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(models_dir / "checkpoints"),
        name_prefix=f"{stage_name}_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = NormalizationSyncEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(eval_dir),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"Config path           : {config_path}")
    print(f"Timesteps             : {timesteps}")
    print(f"Incoming model        : {incoming_model_path}")
    print(f"Incoming vecnorm      : {incoming_vecnorm_path}")
    print(f"Parallel envs         : {n_envs}")
    print(f"Seed                  : {seed}")

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=stage_name,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_model_path = models_dir / f"{stage_name}_final_model.zip"
    final_vecnorm_path = vecnorm_dir / f"{stage_name}_vecnormalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(final_vecnorm_path))

    print(f"Saved final model     : {final_model_path}")
    print(f"Saved vecnorm stats   : {final_vecnorm_path}")

    train_env.close()
    eval_env.close()

    return str(final_model_path), str(final_vecnorm_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable curriculum training for DroneNavEnv")

    parser.add_argument("--easy-config", type=str, default="config_easy.yaml")
    parser.add_argument("--medium-config", type=str, default="config_medium.yaml")
    parser.add_argument("--hard-config", type=str, default="config_hard.yaml")

    parser.add_argument("--easy-timesteps", type=int, default=300_000)
    parser.add_argument("--medium-timesteps", type=int, default=400_000)
    parser.add_argument("--hard-timesteps", type=int, default=500_000)

    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="ppo_curriculum_stable")

    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    curriculum_root = project_root / "results" / "curriculum" / args.run_name
    curriculum_root.mkdir(parents=True, exist_ok=True)

    model_path = None
    vecnorm_path = None

    model_path, vecnorm_path = train_stage(
        stage_name="easy",
        config_path=args.easy_config,
        timesteps=args.easy_timesteps,
        run_root=curriculum_root,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        incoming_model_path=model_path,
        incoming_vecnorm_path=vecnorm_path,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
    )

    model_path, vecnorm_path = train_stage(
        stage_name="medium",
        config_path=args.medium_config,
        timesteps=args.medium_timesteps,
        run_root=curriculum_root,
        n_envs=args.n_envs,
        seed=args.seed + 1_000,
        device=args.device,
        incoming_model_path=model_path,
        incoming_vecnorm_path=vecnorm_path,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
    )

    model_path, vecnorm_path = train_stage(
        stage_name="hard",
        config_path=args.hard_config,
        timesteps=args.hard_timesteps,
        run_root=curriculum_root,
        n_envs=args.n_envs,
        seed=args.seed + 2_000,
        device=args.device,
        incoming_model_path=model_path,
        incoming_vecnorm_path=vecnorm_path,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
    )

    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final model path      : {model_path}")
    print(f"Final vecnorm path    : {vecnorm_path}")


if __name__ == "__main__":
    main()
