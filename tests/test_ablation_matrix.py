import copy

from tests.test_utils import BASE_CONFIG

from experiments.run_ablation_matrix import (
    apply_ablation,
    build_ablation_specs,
    make_skip_row,
    summarize_baseline_on_config,
)
from experiments.evaluate_baselines import build_policies


def test_ablation_specs_mark_eval_safe_and_retrain_required_cases():
    specs = build_ablation_specs()

    assert specs["control"].evaluate_by_default is True
    assert specs["control"].requires_retraining is False
    assert specs["stuck_disabled_eval"].requires_retraining is False
    assert specs["progress_distance_normalized_retrain"].requires_retraining is True
    assert specs["previous_action_observation_retrain"].category == "observation"
    assert specs["frame_stack_memory_retrain"].category == "memory"


def test_progress_distance_normalized_ablation_removes_fixed_normalizer_without_mutating_original():
    config = copy.deepcopy(BASE_CONFIG)
    config["reward"]["shaping"]["progress_normalizer"] = 1.0
    spec = build_ablation_specs()["progress_distance_normalized_retrain"]

    ablated = apply_ablation(config, spec)

    assert "progress_normalizer" not in ablated["reward"]["shaping"]
    assert config["reward"]["shaping"]["progress_normalizer"] == 1.0


def test_stuck_disabled_ablation_uses_eval_safe_high_patience():
    config = copy.deepcopy(BASE_CONFIG)
    spec = build_ablation_specs()["stuck_disabled_eval"]

    ablated = apply_ablation(config, spec)

    assert ablated["reward"]["stuck_patience"] > config["reward"]["stuck_patience"]
    assert ablated["reward"]["stuck_penalty"] == 0.0


def test_skip_row_records_retraining_requirement():
    spec = build_ablation_specs()["obstacle_feature_observation_retrain"]

    row = make_skip_row(spec, "needs retraining")

    assert row["ablation"] == "obstacle_feature_observation_retrain"
    assert row["evaluated"] == 0
    assert row["requires_retraining"] == 1
    assert row["skip_reason"] == "needs retraining"


def test_summarize_baseline_on_config_smoke():
    config = copy.deepcopy(BASE_CONFIG)
    config["environment"]["episode"]["max_steps"] = 5
    config["environment"]["obstacles"]["enabled"] = False
    policy = build_policies(["greedy_goal"])[0]

    summary = summarize_baseline_on_config(
        policy=policy,
        config=config,
        episodes=1,
        seed=123,
    )

    assert summary["policy"] == "greedy_goal"
    assert summary["episodes"] == 1
    assert summary["success_rate"] + summary["timeout_rate"] + summary["collision_rate"] + summary["stuck_rate"] + summary["failed_rate"] == 1.0
