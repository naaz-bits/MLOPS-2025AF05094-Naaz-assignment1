"""API tests. Uses a tiny model trained on the fly so they run without data/."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.data import FEATURES
from src.features import build_pipeline

PATIENT = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
           "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
           "ca": 0, "thal": 6}


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(1)
    X = pd.DataFrame([{**PATIENT, "cp": int(rng.integers(1, 5)),
                       "age": int(rng.integers(30, 77)),
                       "ca": float(rng.integers(0, 4))} for _ in range(40)])[FEATURES]
    y = (X["cp"] == 4).astype(int)
    joblib.dump(build_pipeline(LogisticRegression()).fit(X, y), model_dir / "model.joblib")
    (model_dir / "model_meta.json").write_text(
        json.dumps({"model_name": "test_lr", "features": FEATURES}))

    monkeypatch_module.setattr("api.main.MODEL_DIR", model_dir)
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_single(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "test_lr"
    res = body["results"][0]
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1.0
    assert "x-request-id" in r.headers


def test_predict_batch_with_missing_optional_fields(client):
    p2 = {k: v for k, v in PATIENT.items() if k not in ("ca", "thal")}
    r = client.post("/predict", json=[PATIENT, p2])
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


@pytest.mark.parametrize("bad", [
    {**PATIENT, "cp": 9},          # invalid category
    {**PATIENT, "age": -4},        # out of range
    {k: v for k, v in PATIENT.items() if k != "chol"},  # missing required field
])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json=PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
