import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.features import build_pipeline, build_preprocessor


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "age": rng.integers(30, 77, n), "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n), "thalach": rng.integers(90, 200, n),
        "oldpeak": rng.uniform(0, 5, n).round(1), "cp": rng.integers(1, 5, n),
        "restecg": rng.integers(0, 3, n), "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float), "thal": rng.choice([3, 6, 7], n).astype(float),
        "sex": rng.integers(0, 2, n), "fbs": rng.integers(0, 2, n), "exang": rng.integers(0, 2, n),
    })[FEATURES]
    X.loc[0, "ca"] = np.nan
    X.loc[1, "thal"] = np.nan
    y = (X["cp"] == 4).astype(int) | (X["exang"] == 1).astype(int)
    return X, y


def test_preprocessor_handles_missing_values(sample):
    X, _ = sample
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_preprocessor_output_width(sample):
    X, _ = sample
    out = build_preprocessor().fit_transform(X)
    n_onehot = sum(X[c].dropna().nunique() for c in CATEGORICAL_FEATURES)
    assert out.shape == (len(X), len(NUMERIC_FEATURES) + n_onehot + len(BINARY_FEATURES))


def test_numeric_features_are_scaled(sample):
    X, _ = sample
    out = build_preprocessor().fit_transform(X)[:, :len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-8)


def test_unseen_category_does_not_crash(sample):
    X, y = sample
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["thal"] = 99  # never seen during training
    assert pipe.predict_proba(row).shape == (1, 2)


def test_pipeline_learns_signal(sample):
    X, y = sample
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert (pipe.predict(X) == y).mean() > 0.8
