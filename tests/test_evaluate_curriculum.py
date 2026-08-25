import experiments.evaluate_curriculum as evaluate_curriculum


def _row(model_name: str, eval_name: str, success_rate: float) -> dict:
    return {
        "model_name": model_name,
        "eval_name": eval_name,
        "success_rate": success_rate,
    }


def test_curriculum_transfer_guard_accepts_expected_monotonic_matrix():
    rows = [
        _row("easy", "easy", 0.90),
        _row("easy", "medium", 0.60),
        _row("easy", "hard", 0.20),
        _row("medium", "easy", 0.92),
        _row("medium", "medium", 0.75),
        _row("medium", "hard", 0.40),
        _row("hard", "easy", 0.94),
        _row("hard", "medium", 0.80),
        _row("hard", "hard", 0.55),
    ]

    assert evaluate_curriculum.find_curriculum_transfer_violations(rows) == []


def test_curriculum_transfer_guard_flags_harder_env_improving_easy_model():
    rows = [
        _row("easy", "easy", 0.50),
        _row("easy", "medium", 0.65),
        _row("easy", "hard", 0.20),
    ]

    violations = evaluate_curriculum.find_curriculum_transfer_violations(rows)

    assert len(violations) == 1
    assert "easy model success should not improve from easy" in violations[0]


def test_curriculum_transfer_guard_flags_hard_eval_stage_regression():
    rows = [
        _row("easy", "hard", 0.55),
        _row("medium", "hard", 0.40),
        _row("hard", "hard", 0.70),
    ]

    violations = evaluate_curriculum.find_curriculum_transfer_violations(rows)

    assert len(violations) == 1
    assert "hard eval success should not regress from easy" in violations[0]


def test_curriculum_transfer_guard_honors_tolerance():
    rows = [
        _row("easy", "easy", 0.50),
        _row("easy", "medium", 0.505),
    ]

    assert (
        evaluate_curriculum.find_curriculum_transfer_violations(rows, tolerance=0.01)
        == []
    )
