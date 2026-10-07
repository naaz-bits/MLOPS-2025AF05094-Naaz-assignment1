"""Preprocessing pipeline shared by training and serving.

Everything (imputation, scaling, one-hot encoding) lives inside a single
sklearn ColumnTransformer, so the saved model file is the *whole* pipeline and
the API never has to re-implement preprocessing.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # unseen codes at inference time become all-zero rather than crashing
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
