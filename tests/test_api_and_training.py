"""Inference service probes and training-data validation."""

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from training.make_dataset import build
from training.pipeline import DataValidationError, QualityGateError, preprocess, quality_gate, train, validate


def test_validation_rejects_corrupt_data(tmp_path):
    path = tmp_path / "bad.csv"
    build("corrupt").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="missing label column"):
        validate(path)


def test_quality_gate_blocks_weak_model():
    with pytest.raises(QualityGateError):
        quality_gate({"f1": 0.5}, 0.9)


def test_api_readiness_and_prediction(tmp_path, monkeypatch):
    raw = tmp_path / "d.csv"
    build("v1").to_csv(raw, index=False)
    preprocess(raw, tmp_path / "p", 0.2, 42)
    model = tmp_path / "m.joblib"
    train(tmp_path / "p", model, {"model": "logistic_regression", "seed": 42})

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    import api.main as api_main

    api_main = importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503  # readiness fails without a model

    monkeypatch.setenv("MODEL_PATH", str(model))
    api_main = importlib.reload(api_main)
    row = pd.read_csv(raw).drop(columns=["target"]).iloc[0].to_dict()
    with TestClient(api_main.app) as client:
        assert client.get("/ready").status_code == 200
        body = client.post("/predict", json={"rows": [row]}).json()
        assert body["predictions"][0] in (0, 1)
