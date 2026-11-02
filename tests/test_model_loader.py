"""Tests for champion artifact loading."""

import joblib

from src import model_loader


def test_load_model_returns_serialized_artifact(tmp_path, monkeypatch):
    artifact_path = tmp_path / "model.joblib"
    expected = {"name": "test-model"}
    joblib.dump(expected, artifact_path)
    monkeypatch.setattr(model_loader, "MODEL_PATH", artifact_path)

    assert model_loader.load_model() == expected


def test_load_model_returns_none_when_artifact_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "MODEL_PATH", tmp_path / "missing.joblib")

    assert model_loader.load_model() is None
