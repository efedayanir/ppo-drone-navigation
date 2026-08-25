from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.drone_env import DroneNavEnv


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config: dict) -> DummyVecEnv:
    return DummyVecEnv([lambda: DroneNavEnv(config)])


def infer_run_name_from_model_path(model_path: Path) -> str | None:
    parts = model_path.parts
    if "models" not in parts:
        return None

    try:
        models_idx = parts.index("models")
        return parts[models_idx + 1]
    except (ValueError, IndexError):
        return None


def infer_vecnorm_path(model_path: Path, explicit_vecnorm: str | None) -> Path:
    if explicit_vecnorm:
        return Path(explicit_vecnorm)

    run_name = infer_run_name_from_model_path(model_path)
    if run_name is None:
        raise FileNotFoundError(
            "Could not infer run name from model path. Pass --vecnorm explicitly."
        )

    vecnorm_dir = PROJECT_ROOT / "results" / "vecnormalize" / run_name
    model_name = model_path.name.lower()

    if "best_model" in model_name or "best_model" in str(model_path).lower():
        candidate = vecnorm_dir / "best_vecnormalize.pkl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "Model path looks like a best-model checkpoint, but the matching "
            f"VecNormalize snapshot was not found at {candidate}. "
            "Retrain with the updated training callback or pass --vecnorm explicitly."
        )

    candidate = vecnorm_dir / "vecnormalize.pkl"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find VecNormalize stats at {candidate}. Pass --vecnorm explicitly."
    )


def validate_model_vecnorm_pair(model_path: Path, vecnorm_path: Path) -> None:
    model_str = str(model_path).lower()
    vecnorm_name = vecnorm_path.name.lower()

    if "best_model" in model_str and vecnorm_name == "vecnormalize.pkl":
        print(
            "[warning] You are evaluating a best-model checkpoint with final "
            "VecNormalize stats. This may understate or distort best-model performance."
        )
    if "final_model" in model_str and vecnorm_name == "best_vecnormalize.pkl":
        print(
            "[warning] You are evaluating a final model with best-checkpoint "
            "VecNormalize stats. This pairing is usually incorrect."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable PPO evaluation for DroneNavEnv")

    parser.add_argument("--config", type=str, default="config_easy.yaml")
    parser.add_argument("--model", type=str, default="results/models/ppo_stable_run/final_model.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="results/eval_artifacts",
        help="Directory for episode CSV files, trajectory plots, heatmaps, and GIFs.",
    )
    parser.add_argument(
        "--save-episode-csv",
        action="store_true",
        help="Save per-step telemetry for every evaluated episode.",
    )
    parser.add_argument(
        "--save-failure-gifs",
        action="store_true",
        help="Save GIFs for failed episodes when the environment supports rgb_array rendering.",
    )
    parser.add_argument(
        "--save-demo-gifs",
        action="store_true",
        help="Save GIFs for the first successful and first failed episodes when rendering is available.",
    )
    parser.add_argument(
        "--max-gifs",
        type=int,
        default=3,
        help="Maximum number of GIFs to save.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="Frames per second for exported GIFs.",
    )
    parser.add_argument(
        "--plot-trajectory",
        action="store_true",
        help="Save x/y trajectory plots if telemetry exposes agent coordinates.",
    )
    parser.add_argument(
        "--plot-hesitation-heatmap",
        action="store_true",
        help="Save a speed-weighted hesitation heatmap if x/y/speed telemetry is available.",
    )

    return parser.parse_args()


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(info: Dict[str, Any], keys: List[str]) -> float | None:
    for key in keys:
        value = _as_float(info.get(key))
        if value is not None:
            return value
    return None


def extract_step_telemetry(
    episode_idx: int,
    step_idx: int,
    action: np.ndarray,
    reward: float,
    done: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    reward_terms = info.get("reward_terms", {}) or {}
    action_arr = np.asarray(action).reshape(-1)

    row: Dict[str, Any] = {
        "episode": episode_idx,
        "step": step_idx,
        "reward": reward,
        "done": int(done),
        "reached_goal": int(bool(info.get("reached_goal", False))),
        "collision": int(bool(info.get("collision", False))),
        "stuck": int(bool(info.get("stuck", False))),
        "truncated": int(bool(info.get("truncated", False))),
        "distance_to_goal": _first_float(info, ["distance_to_goal", "goal_distance", "dist_to_goal"]),
        "path_length": _first_float(info, ["path_length"]),
        "path_efficiency": _first_float(info, ["path_efficiency"]),
        "agent_x": _first_float(info, ["agent_x", "x", "pos_x", "position_x"]),
        "agent_y": _first_float(info, ["agent_y", "y", "pos_y", "position_y"]),
        "goal_x": _first_float(info, ["goal_x", "target_x"]),
        "goal_y": _first_float(info, ["goal_y", "target_y"]),
        "speed": _first_float(info, ["speed", "agent_speed"]),
        "heading": _first_float(info, ["heading", "agent_heading"]),
        "action_0": float(action_arr[0]) if action_arr.size > 0 else None,
        "action_1": float(action_arr[1]) if action_arr.size > 1 else None,
    }

    for key, value in reward_terms.items():
        scalar = _as_float(value)
        if scalar is not None:
            row[f"reward_term_{key}"] = scalar

    return row


def save_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_render_rgb(vec_env: VecNormalize) -> np.ndarray | None:
    try:
        frame = vec_env.envs[0].render(mode="rgb_array")
    except TypeError:
        try:
            frame = vec_env.envs[0].render()
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        return frame
    return None


def save_gif(path: Path, frames: List[np.ndarray], fps: int) -> bool:
    if not frames:
        return False
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[warning] imageio is not installed; skipping GIF export.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return True


def save_trajectory_plot(path: Path, rows: List[Dict[str, Any]], title: str) -> bool:
    xs = [row.get("agent_x") for row in rows]
    ys = [row.get("agent_y") for row in rows]
    points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(points) < 2:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warning] matplotlib is not installed; skipping trajectory plot.")
        return False

    x_vals, y_vals = zip(*points)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, y_vals, linewidth=2)
    plt.scatter([x_vals[0]], [y_vals[0]], marker="o", label="start")
    plt.scatter([x_vals[-1]], [y_vals[-1]], marker="x", label="final")

    goal_x = next((row.get("goal_x") for row in rows if row.get("goal_x") is not None), None)
    goal_y = next((row.get("goal_y") for row in rows if row.get("goal_y") is not None), None)
    if goal_x is not None and goal_y is not None:
        plt.scatter([goal_x], [goal_y], marker="*", label="goal")

    plt.title(title)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def save_hesitation_heatmap(path: Path, rows: List[Dict[str, Any]], title: str) -> bool:
    points = []
    for row in rows:
        x = row.get("agent_x")
        y = row.get("agent_y")
        speed = row.get("speed")
        if x is None or y is None:
            continue
        hesitation = 1.0 / (float(speed) + 1e-3) if speed is not None else 1.0
        points.append((float(x), float(y), hesitation))

    if len(points) < 5:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warning] matplotlib is not installed; skipping hesitation heatmap.")
        return False

    xs, ys, weights = zip(*points)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.hist2d(xs, ys, bins=40, weights=weights)
    plt.colorbar(label="hesitation proxy: 1 / speed")
    plt.title(title)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    model_path = Path(args.model)
    vecnorm_path = infer_vecnorm_path(model_path, args.vecnorm)
    validate_model_vecnorm_pair(model_path, vecnorm_path)

    vec_env = make_env(config)
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env)

    artifacts_dir = Path(args.artifacts_dir)
    csv_dir = artifacts_dir / "episode_csv"
    gif_dir = artifacts_dir / "gifs"
    plot_dir = artifacts_dir / "plots"
    saved_gif_count = 0
    saved_success_demo = False
    saved_failure_demo = False
    all_step_rows: List[Dict[str, Any]] = []

    goal_count = 0
    collision_count = 0
    timeout_count = 0
    stuck_count = 0

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    final_distances: List[float] = []
    path_lengths: List[float] = []
    path_efficiencies: List[float] = []

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

    for episode_idx in range(args.episodes):
        vec_env.seed(args.seed + episode_idx)
        obs = vec_env.reset()

        done = False
        ep_reward = 0.0
        ep_len = 0
        last_info = None
        episode_rows: List[Dict[str, Any]] = []
        frames: List[np.ndarray] = []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if args.save_failure_gifs or args.save_demo_gifs:
                frame = try_render_rgb(vec_env)
                if frame is not None:
                    frames.append(frame)

            obs, rewards, dones, infos = vec_env.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]
            last_info = info

            ep_reward += reward
            ep_len += 1

            step_row = extract_step_telemetry(
                episode_idx=episode_idx,
                step_idx=ep_len,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            episode_rows.append(step_row)

            reward_terms = info.get("reward_terms", {})
            for key in reward_term_history:
                reward_term_history[key].append(float(reward_terms.get(key, 0.0)))

        if args.save_episode_csv:
            save_rows_csv(csv_dir / f"episode_{episode_idx:04d}.csv", episode_rows)
        all_step_rows.extend(episode_rows)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if last_info is None:
            continue

        outcome = "timeout"
        if last_info.get("reached_goal", False):
            goal_count += 1
            outcome = "success"
        elif last_info.get("collision", False):
            collision_count += 1
            outcome = "collision"
        elif last_info.get("stuck", False):
            stuck_count += 1
            outcome = "stuck"
        elif last_info.get("truncated", False):
            timeout_count += 1
            outcome = "timeout"
        else:
            timeout_count += 1
            outcome = "timeout"

        should_save_gif = False
        if args.save_failure_gifs and outcome != "success":
            should_save_gif = True
        if args.save_demo_gifs and outcome == "success" and not saved_success_demo:
            should_save_gif = True
            saved_success_demo = True
        if args.save_demo_gifs and outcome != "success" and not saved_failure_demo:
            should_save_gif = True
            saved_failure_demo = True

        if should_save_gif and saved_gif_count < args.max_gifs:
            gif_path = gif_dir / f"episode_{episode_idx:04d}_{outcome}.gif"
            if save_gif(gif_path, frames, fps=args.gif_fps):
                saved_gif_count += 1
                print(f"Saved GIF: {gif_path}")

        if args.plot_trajectory:
            plot_path = plot_dir / f"episode_{episode_idx:04d}_{outcome}_trajectory.png"
            save_trajectory_plot(plot_path, episode_rows, f"Episode {episode_idx} - {outcome}")

        final_distances.append(float(last_info.get("distance_to_goal", 0.0)))
        path_lengths.append(float(last_info.get("path_length", 0.0)))
        path_efficiencies.append(float(last_info.get("path_efficiency", 0.0)))

    success_rate = goal_count / max(args.episodes, 1)
    collision_rate = collision_count / max(args.episodes, 1)
    timeout_rate = timeout_count / max(args.episodes, 1)
    stuck_rate = stuck_count / max(args.episodes, 1)

    mean_reward = _safe_mean(episode_rewards)
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_episode_length = _safe_mean(episode_lengths)
    mean_final_distance = _safe_mean(final_distances)
    mean_path_length = _safe_mean(path_lengths)
    mean_path_efficiency = _safe_mean(path_efficiencies)

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model path               : {model_path}")
    print(f"VecNormalize path        : {vecnorm_path}")
    print(f"Episodes evaluated       : {args.episodes}")
    print(f"Deterministic policy     : {args.deterministic}")
    print(f"Success rate             : {success_rate:.3f}")
    print(f"Collision rate           : {collision_rate:.3f}")
    print(f"Timeout rate             : {timeout_rate:.3f}")
    print(f"Stuck rate               : {stuck_rate:.3f}")
    print(f"Mean episode reward      : {mean_reward:.3f}")
    print(f"Std episode reward       : {std_reward:.3f}")
    print(f"Mean episode length      : {mean_episode_length:.3f}")
    print(f"Mean final distance      : {mean_final_distance:.3f}")
    print(f"Mean path length         : {mean_path_length:.3f}")
    print(f"Mean path efficiency     : {mean_path_efficiency:.3f}")

    if args.save_episode_csv:
        save_rows_csv(csv_dir / "all_episodes_steps.csv", all_step_rows)
        print(f"Saved telemetry CSVs to: {csv_dir}")

    if args.plot_hesitation_heatmap:
        heatmap_path = plot_dir / "hesitation_heatmap_all_episodes.png"
        if save_hesitation_heatmap(heatmap_path, all_step_rows, "Hesitation Heatmap - All Episodes"):
            print(f"Saved hesitation heatmap: {heatmap_path}")
        else:
            print("[warning] Hesitation heatmap skipped. Missing agent_x/agent_y/speed telemetry in info dict.")

    print("\nReward term means:")
    for key, values in reward_term_history.items():
        print(f"  {key:>12}: {_safe_mean(values):.6f}")

    vec_env.close()


if __name__ == "__main__":
    main()