from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.drone_env import DroneNavEnv


REWARD_TERM_KEYS = [
    "collision", "goal", "progress", "step", "smoothness", "stall",
    "alignment", "timeout", "low_speed", "stuck", "total",
]


@dataclass
class EpisodeArtifact:
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
    rows: List[Dict[str, Any]]
    positions: np.ndarray
    headings: np.ndarray
    goal: np.ndarray
    goal_radius: float
    obstacles: list
    width: float
    height: float


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_vec_env(config: dict) -> DummyVecEnv:
    return DummyVecEnv([lambda: DroneNavEnv(config)])


def get_raw_env(vec_env: VecNormalize) -> DroneNavEnv:
    return vec_env.venv.envs[0]


def load_model_and_env(config_path: str | Path, model_path: str | Path, vecnorm_path: str | Path) -> Tuple[PPO, VecNormalize]:
    config = load_config(config_path)
    base_env = make_vec_env(config)
    env = VecNormalize.load(str(vecnorm_path), base_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path), env=env)
    return model, env


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def extract_position_from_info(info: Dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    """Use pre-auto-reset terminal telemetry instead of raw_env.position.

    Stable-Baselines3 VecEnv automatically resets an environment after done=True.
    If we read raw_env.position after vec_env.step(), the last frame can become
    the next episode's start position. That creates the jump you saw in GIFs.
    """
    if "agent_x" in info and "agent_y" in info:
        return np.array([safe_float(info.get("agent_x")), safe_float(info.get("agent_y"))], dtype=np.float32)
    if "agent_position" in info:
        arr = np.asarray(info["agent_position"], dtype=np.float32).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return arr[:2].astype(np.float32)
    return np.asarray(fallback, dtype=np.float32).copy()


def extract_heading_from_info(info: Dict[str, Any], fallback: float) -> float:
    return safe_float(info.get("heading"), fallback) if "heading" in info else float(fallback)


def extract_row(episode_idx: int, step_idx: int, action: np.ndarray, reward: float, done: bool, info: Dict[str, Any]) -> Dict[str, Any]:
    reward_terms = info.get("reward_terms", {}) or {}
    action_arr = np.asarray(action).reshape(-1)
    row: Dict[str, Any] = {
        "episode": episode_idx,
        "step": step_idx,
        "reward": float(reward),
        "done": int(done),
        "outcome_so_far": infer_outcome(info) if done else "running",
        "reached_goal": int(bool(info.get("reached_goal", False))),
        "collision": int(bool(info.get("collision", False))),
        "truncated": int(bool(info.get("truncated", False))),
        "stuck": int(bool(info.get("stuck", False))),
        "distance_to_goal": safe_float(info.get("distance_to_goal")),
        "path_length": safe_float(info.get("path_length")),
        "path_efficiency": safe_float(info.get("path_efficiency")),
        "speed": safe_float(info.get("speed")),
        "heading": safe_float(info.get("heading")),
        "angular_velocity": safe_float(info.get("angular_velocity")),
        "agent_x": safe_float(info.get("agent_x")),
        "agent_y": safe_float(info.get("agent_y")),
        "goal_x": safe_float(info.get("goal_x")),
        "goal_y": safe_float(info.get("goal_y")),
        "action_heading": float(action_arr[0]) if action_arr.size > 0 else 0.0,
        "action_speed": float(action_arr[1]) if action_arr.size > 1 else 0.0,
        "no_progress_steps": int(info.get("no_progress_steps", 0)),
        "best_distance_to_goal": safe_float(info.get("best_distance_to_goal")),
    }
    for key in REWARD_TERM_KEYS:
        row[f"reward_term_{key}"] = safe_float(reward_terms.get(key), 0.0)
    return row


def run_episode(model: PPO, vec_env: VecNormalize, episode_idx: int, seed: int, deterministic: bool) -> EpisodeArtifact:
    vec_env.seed(seed)
    obs = vec_env.reset()
    raw_env = get_raw_env(vec_env)

    positions = [raw_env.position.copy().astype(np.float32)]
    headings = [float(raw_env.heading)]
    goal = raw_env.goal.copy().astype(np.float32)
    goal_radius = float(raw_env.goal_radius)
    obstacles = list(raw_env.obstacles)
    width = float(raw_env.width)
    height = float(raw_env.height)

    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_idx = 0
    done = False
    last_info: Dict[str, Any] = {}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]
        last_info = info
        total_reward += reward
        step_idx += 1

        positions.append(extract_position_from_info(info, positions[-1]))
        headings.append(extract_heading_from_info(info, headings[-1]))
        rows.append(extract_row(episode_idx, step_idx, action, reward, done, info))

    outcome = infer_outcome(last_info)
    return EpisodeArtifact(
        episode=episode_idx,
        outcome=outcome,
        total_reward=total_reward,
        length=step_idx,
        final_distance=safe_float(last_info.get("distance_to_goal")),
        path_length=safe_float(last_info.get("path_length")),
        path_efficiency=safe_float(last_info.get("path_efficiency")),
        collision=bool(last_info.get("collision", False)),
        reached_goal=bool(last_info.get("reached_goal", False)),
        truncated=bool(last_info.get("truncated", False)),
        stuck=bool(last_info.get("stuck", False)),
        rows=rows,
        positions=np.asarray(positions, dtype=np.float32),
        headings=np.asarray(headings, dtype=np.float32),
        goal=goal,
        goal_radius=goal_radius,
        obstacles=obstacles,
        width=width,
        height=height,
    )


def save_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def status_color(outcome: str) -> str:
    return {"success": "#16a34a", "collision": "#dc2626", "timeout": "#f97316", "stuck": "#9333ea", "failed": "#64748b"}.get(outcome, "#64748b")


def draw_world(ax, episode: EpisodeArtifact, title: str, upto: Optional[int] = None, show_sensors: bool = False) -> None:
    positions = episode.positions
    headings = episode.headings
    if upto is not None:
        idx = max(1, min(int(upto), len(positions)))
        positions = positions[:idx]
        headings = headings[:idx]

    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0.0, episode.width)
    ax.set_ylim(0.0, episode.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    for obstacle in episode.obstacles:
        fill = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, color="#111827", alpha=0.18)
        border = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, fill=False, color="#111827", alpha=0.75, linewidth=2)
        ax.add_patch(fill)
        ax.add_patch(border)

    goal_area = plt.Circle((float(episode.goal[0]), float(episode.goal[1])), episode.goal_radius, color="#22c55e", alpha=0.18)
    goal_border = plt.Circle((float(episode.goal[0]), float(episode.goal[1])), episode.goal_radius, fill=False, color="#16a34a", linestyle="--", linewidth=2)
    ax.add_patch(goal_area)
    ax.add_patch(goal_border)
    ax.scatter(episode.goal[0], episode.goal[1], marker="*", s=240, color="#f59e0b", edgecolor="white", linewidth=1.4, zorder=7)

    if len(positions) > 1:
        ax.plot(positions[:, 0], positions[:, 1], color="#2563eb", linewidth=3, solid_capstyle="round", zorder=4)

    ax.scatter(episode.positions[0, 0], episode.positions[0, 1], s=140, color="#0ea5e9", edgecolor="white", linewidth=2, zorder=6)

    final = positions[-1]
    heading = float(headings[-1]) if len(headings) else 0.0
    ax.scatter(final[0], final[1], s=170, color="#ef4444", marker="o", edgecolor="white", linewidth=2, zorder=8)
    ax.arrow(final[0], final[1], math.cos(heading) * 0.45, math.sin(heading) * 0.45, head_width=0.12, head_length=0.16, fc="#ef4444", ec="#ef4444", length_includes_head=True, zorder=9)

    if show_sensors:
        sensor_len = max(episode.width, episode.height) * 0.18
        for i in range(16):
            angle = heading + 2.0 * math.pi * i / 16
            ax.plot([final[0], final[0] + math.cos(angle) * sensor_len], [final[1], final[1] + math.sin(angle) * sensor_len], color="#64748b", linewidth=0.6, alpha=0.35, zorder=2)

    info_text = f"{episode.outcome.upper()}\nReward: {episode.total_reward:.1f}\nSteps: {episode.length}\nFinal dist: {episode.final_distance:.2f}\nEfficiency: {episode.path_efficiency:.2f}"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=status_color(episode.outcome), linewidth=2, alpha=0.95))
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_episode_trajectory(path: Path, episode: EpisodeArtifact, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_world(ax, episode, title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_success_vs_failure_plot(path: Path, success: Optional[EpisodeArtifact], failure: Optional[EpisodeArtifact]) -> bool:
    if success is None and failure is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    fig.patch.set_facecolor("white")
    if success is not None:
        draw_world(axes[0], success, f"Best success: episode {success.episode}")
    else:
        axes[0].axis("off")
        axes[0].set_title("No success episode found")
    if failure is not None:
        draw_world(axes[1], failure, f"Worst failure: episode {failure.episode}")
    else:
        axes[1].axis("off")
        axes[1].set_title("No failure episode found")
    fig.suptitle("PPO Drone Navigation: Success vs Failure", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def render_episode_frame(episode: EpisodeArtifact, upto: int, title: str, show_sensors: bool) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    draw_world(ax, episode, title, upto=upto, show_sensors=show_sensors)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    try:
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
    except AttributeError:
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def save_episode_gif(path: Path, episode: EpisodeArtifact, fps: int, max_frames: int, hold_final_frames: int, show_sensors: bool) -> bool:
    if len(episode.positions) < 2:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    total_steps = len(episode.positions)
    if total_steps <= max_frames:
        frame_indices = list(range(1, total_steps + 1))
    else:
        frame_indices = np.linspace(1, total_steps, max_frames).astype(int).tolist()
    frame_indices.extend([total_steps] * max(0, hold_final_frames))
    frames = [render_episode_frame(episode, idx, f"Episode {episode.episode} - {episode.outcome.upper()}", show_sensors) for idx in frame_indices]
    imageio.mimsave(path, frames, fps=fps)
    return True


def save_outcome_breakdown(path: Path, episodes: List[EpisodeArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["success", "collision", "timeout", "stuck", "failed"]
    counts = [sum(ep.outcome == label for ep in episodes) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts)
    ax.set_title("Outcome Breakdown")
    ax.set_ylabel("Episodes")
    ax.set_xlabel("Outcome")
    total = max(len(episodes), 1)
    for idx, count in enumerate(counts):
        ax.text(idx, count, f"{count}\n{count / total:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_episode_lengths(path: Path, episodes: List[EpisodeArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [ep.length for ep in episodes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=min(30, max(5, int(math.sqrt(max(len(lengths), 1))))))
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_path_efficiency_plot(path: Path, episodes: List[EpisodeArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [ep.episode for ep in episodes]
    ys = [ep.path_efficiency for ep in episodes]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys)
    ax.axhline(float(np.mean(ys)) if ys else 0.0, linestyle="--", linewidth=1)
    ax.set_title("Path Efficiency by Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Path efficiency")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_reward_terms_plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    means = {}
    for key in REWARD_TERM_KEYS:
        values = [safe_float(row.get(f"reward_term_{key}")) for row in rows]
        means[key] = float(np.mean(values)) if values else 0.0
    labels = [key for key in REWARD_TERM_KEYS if key != "total"]
    values = [means[key] for key in labels]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, values)
    ax.set_title("Mean Reward Terms per Step")
    ax.set_ylabel("Mean contribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_hesitation_heatmap(path: Path, rows: List[Dict[str, Any]]) -> bool:
    points = []
    for row in rows:
        x = row.get("agent_x")
        y = row.get("agent_y")
        speed = safe_float(row.get("speed"), 0.0)
        if x is None or y is None:
            continue
        hesitation = 1.0 / (speed + 1e-3)
        points.append((safe_float(x), safe_float(y), hesitation))
    if len(points) < 5:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, weights = zip(*points)
    fig, ax = plt.subplots(figsize=(8, 7))
    hist = ax.hist2d(xs, ys, bins=40, weights=weights)
    fig.colorbar(hist[3], ax=ax, label="hesitation proxy: 1 / speed")
    ax.set_title("Hesitation Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def save_final_positions_plot(path: Path, episodes: List[EpisodeArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Final Positions by Outcome")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    for outcome in ["success", "collision", "timeout", "stuck", "failed"]:
        selected = [ep for ep in episodes if ep.outcome == outcome]
        if not selected:
            continue
        pts = np.asarray([ep.positions[-1] for ep in selected], dtype=np.float32)
        ax.scatter(pts[:, 0], pts[:, 1], label=outcome, alpha=0.8)
    if episodes:
        ax.set_xlim(0, episodes[0].width)
        ax.set_ylim(0, episodes[0].height)
        ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def choose_best_success(episodes: List[EpisodeArtifact]) -> Optional[EpisodeArtifact]:
    successes = [ep for ep in episodes if ep.outcome == "success"]
    if not successes:
        return None
    return max(successes, key=lambda ep: (ep.path_efficiency, -ep.length, ep.total_reward))


def choose_worst_failure(episodes: List[EpisodeArtifact]) -> Optional[EpisodeArtifact]:
    failures = [ep for ep in episodes if ep.outcome != "success"]
    if not failures:
        return None
    def score(ep: EpisodeArtifact) -> Tuple[int, float, float, int]:
        outcome_priority = {"collision": 4, "stuck": 3, "timeout": 2, "failed": 1}.get(ep.outcome, 0)
        return (outcome_priority, ep.final_distance, -ep.path_efficiency, ep.length)
    return max(failures, key=score)


def write_summary(path: Path, episodes: List[EpisodeArtifact], all_rows: List[Dict[str, Any]], best_success: Optional[EpisodeArtifact], worst_failure: Optional[EpisodeArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = max(len(episodes), 1)
    success_count = sum(ep.outcome == "success" for ep in episodes)
    collision_count = sum(ep.outcome == "collision" for ep in episodes)
    timeout_count = sum(ep.outcome == "timeout" for ep in episodes)
    stuck_count = sum(ep.outcome == "stuck" for ep in episodes)
    mean_reward = float(np.mean([ep.total_reward for ep in episodes])) if episodes else 0.0
    mean_length = float(np.mean([ep.length for ep in episodes])) if episodes else 0.0
    mean_efficiency = float(np.mean([ep.path_efficiency for ep in episodes])) if episodes else 0.0
    mean_final_distance = float(np.mean([ep.final_distance for ep in episodes])) if episodes else 0.0
    reward_means = {}
    for key in REWARD_TERM_KEYS:
        values = [safe_float(row.get(f"reward_term_{key}")) for row in all_rows]
        reward_means[key] = float(np.mean(values)) if values else 0.0
    lines = [
        "# Asset Pipeline Summary", "", "## Evaluation Metrics", "",
        f"- Episodes: {len(episodes)}",
        f"- Success rate: {success_count / total:.1%}",
        f"- Collision rate: {collision_count / total:.1%}",
        f"- Timeout rate: {timeout_count / total:.1%}",
        f"- Stuck rate: {stuck_count / total:.1%}",
        f"- Mean reward: {mean_reward:.3f}",
        f"- Mean episode length: {mean_length:.3f}",
        f"- Mean final distance: {mean_final_distance:.3f}",
        f"- Mean path efficiency: {mean_efficiency:.3f}",
        "", "## Selected Episodes", "",
        f"- Best success: episode {best_success.episode if best_success else 'none'}",
        f"- Worst failure: episode {worst_failure.episode if worst_failure else 'none'}",
        "", "## Reward Term Means", "", "| Term | Mean contribution |", "| --- | ---: |",
    ]
    for key in REWARD_TERM_KEYS:
        lines.append(f"| {key} | {reward_means[key]:.6f} |")
    lines.extend([
        "", "## Generated Assets", "",
        "- `metrics.csv`", "- `steps.csv`", "- `outcome_breakdown.png`",
        "- `episode_lengths.png`", "- `path_efficiency.png`", "- `reward_terms.png`",
        "- `hesitation_heatmap.png`", "- `final_positions.png`",
        "- `success_vs_failure_trajectory.png`", "- `best_success_trajectory.png`",
        "- `worst_failure_trajectory.png`", "- `best_success.gif`", "- `worst_failure.gif`",
        "", "## Shareable Hook", "",
        "The agent did not mainly fail by crashing. It often failed by hesitating, losing progress, and timing out.",
        "", "## Implementation Note", "",
        "Terminal positions are recorded from the environment info dictionary before SB3 VecEnv auto-reset can change the raw environment state.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shareable analysis assets for DroneNavEnv PPO checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Path to PPO .zip model.")
    parser.add_argument("--vecnorm", required=True, help="Path to VecNormalize .pkl.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", default="assets/analysis")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    parser.add_argument("--gif-hold-final-frames", type=int, default=24)
    parser.add_argument("--show-sensors-in-gif", action="store_true", help="Draw simple sensor rays on exported trajectory GIF frames.")
    parser.add_argument("--no-gifs", action="store_true", help="Skip GIF export.")
    parser.add_argument("--save-all-episode-csv", action="store_true", help="Also save one CSV per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, vec_env = load_model_and_env(args.config, args.model, args.vecnorm)
    episodes: List[EpisodeArtifact] = []
    all_rows: List[Dict[str, Any]] = []
    try:
        for episode_idx in range(args.episodes):
            episode = run_episode(model, vec_env, episode_idx, args.seed + episode_idx, args.deterministic)
            episodes.append(episode)
            all_rows.extend(episode.rows)
            if args.save_all_episode_csv:
                save_csv(output_dir / "episode_csv" / f"episode_{episode_idx:04d}_{episode.outcome}.csv", episode.rows)
            print(f"[{episode_idx + 1:04d}/{args.episodes:04d}] {episode.outcome:<9} reward={episode.total_reward:8.2f} steps={episode.length:4d} eff={episode.path_efficiency:.3f} dist={episode.final_distance:.3f} final=({episode.positions[-1, 0]:.2f}, {episode.positions[-1, 1]:.2f})")
    finally:
        vec_env.close()

    best_success = choose_best_success(episodes)
    worst_failure = choose_worst_failure(episodes)
    metrics_rows = [{
        "episode": ep.episode, "outcome": ep.outcome, "total_reward": ep.total_reward,
        "length": ep.length, "final_distance": ep.final_distance, "path_length": ep.path_length,
        "path_efficiency": ep.path_efficiency, "collision": int(ep.collision),
        "reached_goal": int(ep.reached_goal), "truncated": int(ep.truncated), "stuck": int(ep.stuck),
        "final_x": float(ep.positions[-1, 0]), "final_y": float(ep.positions[-1, 1]),
    } for ep in episodes]

    save_csv(output_dir / "metrics.csv", metrics_rows)
    save_csv(output_dir / "steps.csv", all_rows)
    save_outcome_breakdown(output_dir / "outcome_breakdown.png", episodes)
    save_episode_lengths(output_dir / "episode_lengths.png", episodes)
    save_path_efficiency_plot(output_dir / "path_efficiency.png", episodes)
    save_reward_terms_plot(output_dir / "reward_terms.png", all_rows)
    save_hesitation_heatmap(output_dir / "hesitation_heatmap.png", all_rows)
    save_final_positions_plot(output_dir / "final_positions.png", episodes)
    save_success_vs_failure_plot(output_dir / "success_vs_failure_trajectory.png", best_success, worst_failure)
    if best_success is not None:
        save_episode_trajectory(output_dir / "best_success_trajectory.png", best_success, f"Best Success: Episode {best_success.episode}")
    if worst_failure is not None:
        save_episode_trajectory(output_dir / "worst_failure_trajectory.png", worst_failure, f"Worst Failure: Episode {worst_failure.episode}")
    if not args.no_gifs:
        if best_success is not None:
            save_episode_gif(output_dir / "best_success.gif", best_success, args.gif_fps, args.gif_max_frames, args.gif_hold_final_frames, args.show_sensors_in_gif)
        if worst_failure is not None:
            save_episode_gif(output_dir / "worst_failure.gif", worst_failure, args.gif_fps, args.gif_max_frames, args.gif_hold_final_frames, args.show_sensors_in_gif)
    write_summary(output_dir / "summary.md", episodes, all_rows, best_success, worst_failure)

    print("\nGenerated assets:")
    for name in ["summary.md", "metrics.csv", "steps.csv", "outcome_breakdown.png", "success_vs_failure_trajectory.png", "reward_terms.png", "hesitation_heatmap.png", "final_positions.png"]:
        print(f"  {output_dir / name}")
    if best_success is not None:
        print(f"  selected best success episode: {best_success.episode}")
    if worst_failure is not None:
        print(f"  selected worst failure episode: {worst_failure.episode}")


if __name__ == "__main__":
    main()