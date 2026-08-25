from env.drone_env import DroneNavEnv

BASE_CONFIG = {
    "project": {"seed": 42},
    "environment": {
        "world": {"width": 10.0, "height": 10.0},
        "drone": {
            "radius": 0.2,
            "max_speed": 1.5,
            "max_acceleration": 1.0,
            "max_angular_speed": 2.0,
            "initial_heading": 0.0,
        },
        "obstacles": {
            "enabled": True,
            "count_min": 0,
            "count_max": 0,
            "radius_min": 0.3,
            "radius_max": 0.8,
            "safe_spawn_margin": 0.8,
            "safe_goal_margin": 0.8,
            "safe_obstacle_margin": 0.2,
        },
        "goal": {
            "radius": 0.3,
            "min_start_goal_distance": 2.0,
        },
        "sensors": {
            "num_rays": 8,
            "max_range": 3.0,
            "noise_std": 0.0,
            "add_noise": False,
            "normalize_readings": True,
        },
        "episode": {
            "dt": 0.1,
            "max_steps": 300,
            "terminate_on_collision": True,
            "terminate_on_goal": True,
        },
        "observation": {
            "include_goal_relative": True,
            "include_velocity": True,
            "include_heading": True,
            "include_angular_velocity": False,
            "include_previous_action": False,
            "normalize_goal_delta": True,
            "normalize_velocity": True,
        },
        "action": {
            "heading_min": -3.141592653589793,
            "heading_max": 3.141592653589793,
            "speed_min": 0.0,
            "speed_max": 1.5,
        },
    },
    "control": {
        "type": "pd",
        "heading": {
            "kp": 2.5,
            "kd": 0.4,
            "ki": 0.0,
            "integral_limit": 0.5,
        },
        "speed": {
            "kp": 1.8,
            "kd": 0.2,
            "ki": 0.0,
            "integral_limit": 0.5,
        },
        "limits": {
            "max_linear_acceleration": 1.0,
            "max_angular_velocity": 2.0,
        },
    },
    "reward": {
        "goal_reward": 100.0,
        "collision_penalty": 50.0,
        "timeout_penalty": 20.0,
        "progress_weight": 16.0,
        "step_penalty": 0.01,
        "smoothness_weight": 0.02,
        "stall_penalty": 1.0,
        "low_speed_penalty": 0.25,
        "stuck_penalty": 8.0,
        "stuck_progress_epsilon": 0.003,
        "stuck_speed_threshold": 0.05,
        "stuck_patience": 80,
        "action_speed_max": 1.5,
        "shaping": {
            "clip_progress": True,
            "progress_clip_min": -0.5,
            "progress_clip_max": 0.5,
            "progress_normalizer": 1.0,
            "stall_progress_threshold": 1e-3,
            "stall_speed_threshold": 5e-2,
            "stall_distance_gate": 0.5,
            "alignment_weight": 1.0,
            "min_speed_reward_gate": 0.10,
        },
    },
}


def make_env():
    return DroneNavEnv(BASE_CONFIG)
