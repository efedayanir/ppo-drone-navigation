import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dynamics import apply_dynamics
from env.geometry import (
    check_collision,
    check_goal,
    distance_to_goal,
    point_is_safe,
    wrap_angle,
)
from env.obstacles import Obstacle, generate_obstacles
from env.reward import compute_reward_with_info
from env.sampling import sample_start_and_goal
from env.sensors import cast_ray, get_sensor_readings


class DroneNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        env_cfg = config["environment"]
        drone_cfg = env_cfg["drone"]
        sensor_cfg = env_cfg["sensors"]
        episode_cfg = env_cfg["episode"]
        control_cfg = config["control"]
        self.reward_config = config["reward"]
        shaping_cfg = self.reward_config.get("shaping", {})

        self.width = float(env_cfg["world"]["width"])
        self.height = float(env_cfg["world"]["height"])
        self.world_diag = float(np.linalg.norm([self.width, self.height]))

        self.drone_radius = float(drone_cfg["radius"])
        self.max_speed = float(drone_cfg["max_speed"])
        self.max_acceleration = float(drone_cfg["max_acceleration"])
        self.max_angular_speed = float(drone_cfg["max_angular_speed"])
        self.initial_heading = float(drone_cfg.get("initial_heading", 0.0))

        self.num_rays = int(sensor_cfg["num_rays"])
        self.sensor_max_range = float(sensor_cfg["max_range"])
        self.sensor_noise_std = float(sensor_cfg["noise_std"])
        self.add_sensor_noise = bool(sensor_cfg["add_noise"])
        self.normalize_sensor_readings = bool(sensor_cfg["normalize_readings"])

        self.dt = float(episode_cfg["dt"])
        self.max_steps = int(episode_cfg["max_steps"])
        self.terminate_on_collision = bool(episode_cfg["terminate_on_collision"])
        self.terminate_on_goal = bool(episode_cfg["terminate_on_goal"])

        self.goal_radius = float(env_cfg["goal"]["radius"])

        self.heading_kp = float(control_cfg["heading"]["kp"])
        self.heading_kd = float(control_cfg["heading"]["kd"])
        self.speed_kp = float(control_cfg["speed"]["kp"])
        self.speed_kd = float(control_cfg["speed"]["kd"])
        self.stuck_progress_epsilon = float(self.reward_config.get("stuck_progress_epsilon", 0.003))
        self.stuck_speed_threshold = float(
            self.reward_config.get(
                "stuck_speed_threshold",
                shaping_cfg.get("stall_speed_threshold", 0.05),
            )
        )
        self.stuck_patience = int(self.reward_config.get("stuck_patience", 80))

        if self.normalize_sensor_readings:
            sensor_low = np.zeros(self.num_rays, dtype=np.float32)
            sensor_high = np.ones(self.num_rays, dtype=np.float32)
        else:
            sensor_low = np.zeros(self.num_rays, dtype=np.float32)
            sensor_high = np.full(self.num_rays, self.sensor_max_range, dtype=np.float32)

        low = np.concatenate(
            [
                np.array([-1.0, -1.0], dtype=np.float32),  # goal delta
                np.array([-1.0, -1.0], dtype=np.float32),  # velocity
                np.array([-1.0, -1.0], dtype=np.float32),  # heading cos/sin
                sensor_low,
            ]
        )
        high = np.concatenate(
            [
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                sensor_high,
            ]
        )

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(2 + 2 + 2 + self.num_rays,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-math.pi, 0.0], dtype=np.float32),
            high=np.array([math.pi, self.max_speed], dtype=np.float32),
            dtype=np.float32,
        )

        self.np_random = None
        self.obstacles: List[Obstacle] = []
        self.trajectory: List[Tuple[float, float]] = []

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.angular_velocity = 0.0
        self.goal = np.zeros(2, dtype=np.float32)

        self.prev_distance_to_goal = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)

        self.last_reward_terms: Dict[str, float] = {}
        self.no_progress_steps = 0
        self.best_distance_to_goal = 0.0
        self.step_count = 0

        self._screen = None
        self._clock = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            default_seed = int(self.config.get("project", {}).get("seed", 42))
            self.np_random = np.random.default_rng(default_seed)

        self.step_count = 0
        self.no_progress_steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.angular_velocity = 0.0
        reset_options = options or {}

        self.heading = self.initial_heading
        self.trajectory = []

        self.last_reward_terms = {
            "collision": 0.0,
            "goal": 0.0,
            "progress": 0.0,
            "step": 0.0,
            "smoothness": 0.0,
            "stall": 0.0,
            "alignment": 0.0,
            "timeout": 0.0,
            "low_speed": 0.0,
            "stuck": 0.0,
            "total": 0.0,
        }

        self.obstacles = self._generate_obstacles()
        if "obstacles" in reset_options:
            self.obstacles = self._coerce_reset_obstacles(reset_options["obstacles"])

        self.position, self.goal = self._sample_start_and_goal()
        if "position" in reset_options:
            self.position = self._coerce_reset_point(reset_options["position"], "position")
        if "goal" in reset_options:
            self.goal = self._coerce_reset_point(reset_options["goal"], "goal")
        if "heading" in reset_options:
            heading = float(reset_options["heading"])
            if not math.isfinite(heading):
                raise ValueError("reset option 'heading' must be finite.")
            self.heading = self._wrap_angle(heading)
        if "velocity" in reset_options:
            self.velocity = self._coerce_reset_vector(reset_options["velocity"], "velocity")
        if "angular_velocity" in reset_options:
            angular_velocity = float(reset_options["angular_velocity"])
            if not math.isfinite(angular_velocity):
                raise ValueError("reset option 'angular_velocity' must be finite.")
            self.angular_velocity = angular_velocity

        self.position = self.position.astype(np.float32)
        self.goal = self.goal.astype(np.float32)

        self.trajectory.append((float(self.position[0]), float(self.position[1])))
        self.prev_distance_to_goal = self._distance_to_goal()
        self.best_distance_to_goal = self.prev_distance_to_goal

        obs = self._get_observation()
        info = self._get_info(
            collision=False,
            reached_goal=False,
            truncated=False,
            stuck=False,
        )
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        old_prev_distance = self.prev_distance_to_goal
        old_prev_action = self.prev_action.copy()

        self.position, self.velocity, self.heading, self.angular_velocity = apply_dynamics(
            position=self.position,
            velocity=self.velocity,
            heading=self.heading,
            angular_velocity=self.angular_velocity,
            action=action,
            dt=self.dt,
            max_speed=self.max_speed,
            max_acceleration=self.max_acceleration,
            max_angular_speed=self.max_angular_speed,
            heading_kp=self.heading_kp,
            heading_kd=self.heading_kd,
            speed_kp=self.speed_kp,
            speed_kd=self.speed_kd,
            wrap_angle_fn=self._wrap_angle,
        )

        self.position[0] = float(np.clip(self.position[0], 0.0, self.width))
        self.position[1] = float(np.clip(self.position[1], 0.0, self.height))

        self.trajectory.append((float(self.position[0]), float(self.position[1])))
        self.step_count += 1

        collision = self._check_collision()
        reached_goal = self._check_goal()
        timeout = self.step_count >= self.max_steps

        current_distance = self._distance_to_goal()
        progress = old_prev_distance - current_distance
        speed = float(np.linalg.norm(self.velocity))

        if progress > self.stuck_progress_epsilon:
            self.no_progress_steps = 0
        elif speed < self.stuck_speed_threshold:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = max(0, self.no_progress_steps - 1)

        self.best_distance_to_goal = min(self.best_distance_to_goal, current_distance)
        stuck = self.no_progress_steps >= self.stuck_patience

        terminated = False
        if collision and self.terminate_on_collision:
            terminated = True
        if reached_goal and self.terminate_on_goal:
            terminated = True
        if stuck:
            terminated = True

        truncated = bool(timeout and not terminated)

        reward, reward_terms = compute_reward_with_info(
            action=action,
            velocity=self.velocity,
            prev_action=old_prev_action,
            prev_distance_to_goal=old_prev_distance,
            current_distance=current_distance,
            collision=collision,
            reached_goal=reached_goal,
            truncated=truncated,
            config=self.reward_config,
            goal_vector=(self.goal - self.position),
        )

        stuck_penalty = float(self.reward_config.get("stuck_penalty", 0.0)) if stuck else 0.0
        reward_terms["stuck"] = -stuck_penalty
        reward = float(reward - stuck_penalty)
        reward_terms["total"] = float(reward)

        self.prev_action = action.copy()
        self.last_action = action.copy()
        self.prev_distance_to_goal = current_distance
        self.last_reward_terms = dict(reward_terms)

        obs = self._get_observation()
        info = self._get_info(
            collision=collision,
            reached_goal=reached_goal,
            truncated=truncated,
            stuck=stuck,
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: Optional[str] = None):
        if mode is not None:
            previous_render_mode = self.render_mode
            self.render_mode = mode
        else:
            previous_render_mode = None
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("pygame is required for rendering. Install it or use render_mode=None.") from exc

        canvas_size = 700
        margin = 20
        scale_x = (canvas_size - 2 * margin) / self.width
        scale_y = (canvas_size - 2 * margin) / self.height

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((canvas_size, canvas_size))
                pygame.display.set_caption("Drone Navigation Environment")
            else:
                self._screen = pygame.Surface((canvas_size, canvas_size))
            self._clock = pygame.time.Clock()

        surface = self._screen
        surface.fill((245, 245, 245))

        def world_to_screen(point: np.ndarray) -> Tuple[int, int]:
            x = int(margin + point[0] * scale_x)
            y = int(canvas_size - (margin + point[1] * scale_y))
            return x, y

        for obstacle in self.obstacles:
            center = world_to_screen(np.array([obstacle.x, obstacle.y], dtype=np.float32))
            radius_px = int(obstacle.radius * min(scale_x, scale_y))
            pygame.draw.circle(surface, (70, 70, 70), center, radius_px)

        goal_center = world_to_screen(self.goal)
        goal_radius_px = int(self.goal_radius * min(scale_x, scale_y))
        pygame.draw.circle(surface, (60, 180, 75), goal_center, goal_radius_px)

        if len(self.trajectory) > 1:
            pygame.draw.lines(
                surface,
                (100, 149, 237),
                False,
                [world_to_screen(np.array(p, dtype=np.float32)) for p in self.trajectory],
                2,
            )

        drone_center = world_to_screen(self.position)
        drone_radius_px = int(self.drone_radius * min(scale_x, scale_y))
        pygame.draw.circle(surface, (220, 20, 60), drone_center, drone_radius_px)

        nose = self.position + np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32) * 0.4
        pygame.draw.line(surface, (0, 0, 0), drone_center, world_to_screen(nose), 3)

        sensor_readings = self._get_sensor_readings()
        for i, dist in enumerate(sensor_readings):
            angle = self.heading + (2.0 * math.pi * i / self.num_rays)
            ray_len = dist * self.sensor_max_range if self.normalize_sensor_readings else dist
            endpoint = self.position + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * ray_len
            pygame.draw.line(surface, (180, 180, 180), drone_center, world_to_screen(endpoint), 1)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            if previous_render_mode is not None:
                self.render_mode = previous_render_mode
            return None

        rgb = pygame.surfarray.array3d(surface)
        rgb = np.transpose(rgb, (1, 0, 2))
        if previous_render_mode is not None:
            self.render_mode = previous_render_mode
        return rgb

    def close(self):
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
        self._screen = None
        self._clock = None

    def _generate_obstacles(self) -> List[Obstacle]:
        return generate_obstacles(
            rng=self.np_random,
            width=self.width,
            height=self.height,
            config=self.config["environment"]["obstacles"],
        )

    def _coerce_reset_point(self, value: Any, name: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float32)
        if point.shape != (2,):
            raise ValueError(f"reset option '{name}' must be a 2D point.")
        if not np.all(np.isfinite(point)):
            raise ValueError(f"reset option '{name}' must contain finite values.")
        if not (0.0 <= float(point[0]) <= self.width and 0.0 <= float(point[1]) <= self.height):
            raise ValueError(f"reset option '{name}' must be inside the world bounds.")
        return point

    def _coerce_reset_vector(self, value: Any, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (2,):
            raise ValueError(f"reset option '{name}' must be a 2D vector.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"reset option '{name}' must contain finite values.")
        return vector

    def _coerce_reset_obstacles(self, value: Any) -> List[Obstacle]:
        if value is None:
            return []

        obstacles: List[Obstacle] = []
        for obstacle in value:
            if isinstance(obstacle, Obstacle):
                candidate = Obstacle(
                    x=float(obstacle.x),
                    y=float(obstacle.y),
                    radius=float(obstacle.radius),
                )
            elif isinstance(obstacle, dict):
                candidate = Obstacle(
                    x=float(obstacle["x"]),
                    y=float(obstacle["y"]),
                    radius=float(obstacle["radius"]),
                )
            else:
                x, y, radius = obstacle
                candidate = Obstacle(x=float(x), y=float(y), radius=float(radius))

            if not all(np.isfinite([candidate.x, candidate.y, candidate.radius])):
                raise ValueError("reset option 'obstacles' must contain finite values.")
            if candidate.radius < 0.0:
                raise ValueError("reset option 'obstacles' cannot contain negative radii.")
            obstacles.append(candidate)

        return obstacles

    def _sample_start_and_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        sampling_cfg = {
            "safe_spawn_margin": self.config["environment"]["obstacles"].get("safe_spawn_margin", self.drone_radius),
            "safe_goal_margin": self.config["environment"]["obstacles"].get("safe_goal_margin", self.goal_radius),
            "min_start_goal_distance": self.config["environment"]["goal"].get("min_start_goal_distance", 2.0),
            "max_sampling_tries": self.config["environment"]["goal"].get("max_sampling_tries", 100),
        }
        return sample_start_and_goal(
            rng=self.np_random,
            width=self.width,
            height=self.height,
            drone_radius=self.drone_radius,
            goal_radius=self.goal_radius,
            config=sampling_cfg,
            point_is_safe_fn=self._point_is_safe,
        )

    def _point_is_safe(self, point: np.ndarray, radius: float) -> bool:
        return point_is_safe(
            point,
            radius,
            self.width,
            self.height,
            self.obstacles,
        )

    def _get_observation(self) -> np.ndarray:
        goal_delta = self.goal - self.position
        goal_delta_norm = goal_delta / max(self.world_diag, 1e-8)

        velocity_norm = self.velocity / max(self.max_speed, 1e-8)
        velocity_norm = np.clip(velocity_norm, -1.0, 1.0)

        heading_repr = np.array(
            [math.cos(self.heading), math.sin(self.heading)],
            dtype=np.float32,
        )

        sensor_readings = self._get_sensor_readings()

        obs = np.concatenate(
            [
                goal_delta_norm.astype(np.float32),
                velocity_norm.astype(np.float32),
                heading_repr.astype(np.float32),
                sensor_readings.astype(np.float32),
            ]
        ).astype(np.float32)

        return obs

    def _get_sensor_readings(self) -> np.ndarray:
        return get_sensor_readings(
            position=self.position,
            heading=self.heading,
            num_rays=self.num_rays,
            sensor_max_range=self.sensor_max_range,
            normalize_sensor_readings=self.normalize_sensor_readings,
            add_sensor_noise=self.add_sensor_noise,
            sensor_noise_std=self.sensor_noise_std,
            rng=self.np_random,
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
        )

    def _cast_ray(self, angle: float) -> float:
        return cast_ray(
            position=self.position,
            angle=angle,
            sensor_max_range=self.sensor_max_range,
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
        )

    def _compute_reward(self, action: np.ndarray, collision: bool, reached_goal: bool) -> float:
        current_distance = self._distance_to_goal()
        reward, _ = compute_reward_with_info(
            action=np.asarray(action, dtype=np.float32),
            velocity=self.velocity,
            prev_action=self.prev_action,
            prev_distance_to_goal=self.prev_distance_to_goal,
            current_distance=current_distance,
            collision=collision,
            reached_goal=reached_goal,
            truncated=False,
            config=self.reward_config,
            goal_vector=(self.goal - self.position),
        )
        return float(reward)

    def _distance_to_goal(self) -> float:
        return distance_to_goal(self.position, self.goal)

    def _check_collision(self) -> bool:
        return check_collision(
            self.position,
            self.drone_radius,
            self.width,
            self.height,
            self.obstacles,
        )

    def _check_goal(self) -> bool:
        return check_goal(self.position, self.goal, self.goal_radius)

    def _get_info(
        self,
        collision: bool,
        reached_goal: bool,
        truncated: bool,
        stuck: bool = False,
    ) -> Dict[str, Any]:
        path_length = 0.0
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                p0 = np.array(self.trajectory[i - 1], dtype=np.float32)
                p1 = np.array(self.trajectory[i], dtype=np.float32)
                path_length += float(np.linalg.norm(p1 - p0))

        start_to_goal_distance = 0.0
        if self.trajectory:
            start = np.array(self.trajectory[0], dtype=np.float32)
            start_to_goal_distance = float(np.linalg.norm(self.goal - start))

        if reached_goal and path_length > 1e-8:
            path_efficiency = min(1.0, start_to_goal_distance / path_length)
        else:
            path_efficiency = 0.0

        dist = self._distance_to_goal()
        speed = float(np.linalg.norm(self.velocity))

        return {
            "collision": bool(collision),
            "reached_goal": bool(reached_goal),
            "truncated": bool(truncated),
            "stuck": bool(stuck),
            "distance_to_goal": float(dist),
            "distance_to_goal_normalized": float(dist / max(self.world_diag, 1e-8)),
            "speed": speed,
            "speed_normalized": float(speed / max(self.max_speed, 1e-8)),
            "heading": float(self.heading),
            "angular_velocity": float(self.angular_velocity),
            "step_count": int(self.step_count),
            "path_length": float(path_length),
            "path_efficiency": float(path_efficiency),
            "goal_position": self.goal.copy(),
            "agent_position": self.position.copy(),
            "agent_x": float(self.position[0]),
            "agent_y": float(self.position[1]),
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "velocity_x": float(self.velocity[0]),
            "velocity_y": float(self.velocity[1]),
            "last_action": self.last_action.copy(),
            "action_heading": float(self.last_action[0]),
            "action_speed": float(self.last_action[1]),
            "reward_terms": dict(self.last_reward_terms),
            "no_progress_steps": int(self.no_progress_steps),
            "best_distance_to_goal": float(self.best_distance_to_goal),
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return wrap_angle(angle)
