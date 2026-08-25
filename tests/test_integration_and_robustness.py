import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.drone_env import DroneNavEnv
from env.obstacles import Obstacle
from tests.test_utils import BASE_CONFIG, make_env


def _exact_color_count(frame: np.ndarray, rgb: tuple[int, int, int]) -> int:
    color = np.array(rgb, dtype=np.uint8)
    return int(np.count_nonzero(np.all(frame == color, axis=-1)))


def test_render_rgb_array_returns_valid_image():
    env = make_env()
    env.reset(
        seed=123,
        options={
            "position": [2.0, 2.0],
            "goal": [8.0, 8.0],
            "obstacles": [Obstacle(x=5.0, y=5.0, radius=0.7)],
        },
    )

    frame = env.render(mode="rgb_array")

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert np.all(np.isfinite(frame))
    assert _exact_color_count(frame, (220, 20, 60)) > 50
    assert _exact_color_count(frame, (60, 180, 75)) > 50
    assert _exact_color_count(frame, (70, 70, 70)) > 50

    env.close()


def test_terminated_and_truncated_are_not_both_true():
    env = make_env()
    obs, _ = env.reset(seed=123)

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert not (terminated and truncated)

        if terminated or truncated:
            obs, _ = env.reset(seed=456)

    env.close()


def test_long_random_rollout_stays_finite():
    env = make_env()
    obs, _ = env.reset(seed=123)

    for step in range(10_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.all(np.isfinite(env.position))
        assert np.all(np.isfinite(env.velocity))
        assert np.isfinite(env.heading)
        assert np.isfinite(env.angular_velocity)

        if terminated or truncated:
            obs, _ = env.reset(seed=123 + step)

    env.close()


def test_same_seed_and_actions_produce_same_trajectory():
    env1 = make_env()
    env2 = make_env()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    rng = np.random.default_rng(999)

    for _ in range(100):
        action = np.array(
            [
                rng.uniform(env1.action_space.low[0], env1.action_space.high[0]),
                rng.uniform(env1.action_space.low[1], env1.action_space.high[1]),
            ],
            dtype=np.float32,
        )

        obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
        obs2, reward2, terminated2, truncated2, info2 = env2.step(action)

        assert np.allclose(obs1, obs2)
        assert np.isclose(reward1, reward2)
        assert terminated1 == terminated2
        assert truncated1 == truncated2
        assert np.allclose(env1.position, env2.position)
        assert np.allclose(env1.velocity, env2.velocity)
        assert np.isclose(env1.heading, env2.heading)

        if terminated1 or truncated1:
            break

    env1.close()
    env2.close()


def test_sb3_ppo_vecnormalize_smoke_test():
    config = copy.deepcopy(BASE_CONFIG)
    config["environment"]["episode"]["max_steps"] = 20

    def make_single_env():
        return DroneNavEnv(config)

    vec_env = DummyVecEnv([make_single_env])
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.995,
        verbose=0,
        seed=123,
    )

    model.learn(total_timesteps=32)

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(reward))

    vec_env.close()


def test_curriculum_configs_have_compatible_observation_and_action_spaces():
    config_paths = [
        Path("configs/config_easy.yaml"),
        Path("configs/config_medium.yaml"),
        Path("configs/config_hard.yaml"),
    ]

    existing_paths = [p for p in config_paths if p.exists()]
    if len(existing_paths) < 3:
        pytest.skip("Curriculum config files are not available in this test environment.")

    signatures = []

    for path in existing_paths:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        env = DroneNavEnv(config)
        signatures.append((env.observation_space.shape, env.action_space.shape))
        env.close()

    assert len(set(signatures)) == 1
