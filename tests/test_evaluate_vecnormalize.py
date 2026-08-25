from pathlib import Path

import pytest

import experiments.evaluate as evaluate


def test_validate_model_vecnorm_pair_warns_for_best_model_with_final_vecnormalize(capsys):
    evaluate.validate_model_vecnorm_pair(
        Path("results/models/run/best_model/best_model.zip"),
        Path("results/vecnormalize/run/vecnormalize.pkl"),
    )

    captured = capsys.readouterr()
    assert "best-model checkpoint with final VecNormalize stats" in captured.out


def test_validate_model_vecnorm_pair_warns_for_final_model_with_best_vecnormalize(capsys):
    evaluate.validate_model_vecnorm_pair(
        Path("results/models/run/final_model.zip"),
        Path("results/vecnormalize/run/best_vecnormalize.pkl"),
    )

    captured = capsys.readouterr()
    assert "final model with best-checkpoint VecNormalize stats" in captured.out


def test_validate_model_vecnorm_pair_allows_matching_names_without_warning(capsys):
    evaluate.validate_model_vecnorm_pair(
        Path("results/models/run/best_model/best_model.zip"),
        Path("results/vecnormalize/run/best_vecnormalize.pkl"),
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_infer_vecnorm_path_prefers_best_snapshot_for_best_model(tmp_path, monkeypatch):
    model_path = tmp_path / "results" / "models" / "run_a" / "best_model" / "best_model.zip"
    model_path.parent.mkdir(parents=True)
    model_path.touch()

    vecnorm_path = tmp_path / "results" / "vecnormalize" / "run_a" / "best_vecnormalize.pkl"
    vecnorm_path.parent.mkdir(parents=True)
    vecnorm_path.touch()

    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)

    assert evaluate.infer_vecnorm_path(model_path, explicit_vecnorm=None) == vecnorm_path


def test_infer_vecnorm_path_rejects_best_model_without_best_snapshot(tmp_path, monkeypatch):
    model_path = tmp_path / "results" / "models" / "run_a" / "best_model" / "best_model.zip"
    model_path.parent.mkdir(parents=True)
    model_path.touch()

    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="best-model checkpoint"):
        evaluate.infer_vecnorm_path(model_path, explicit_vecnorm=None)
