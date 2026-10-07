"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m src.data            # download raw file + write cleaned CSV
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Feature groups drive the preprocessing pipeline (see src/features.py)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def download(url: str = UCI_URL, dest: Path = RAW_PATH) -> Path:
    """Download the raw Cleveland file if it is not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Raw data already present at %s", dest)
        return dest
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw file; UCI marks missing values with '?'."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning. Imputation is left to the model pipeline to avoid leakage.

    - drop exact duplicate rows
    - collapse the 0-4 severity score `num` into a binary target
    - cast categorical codes to integers where present (nullable Int64)
    - null out physiologically impossible values (e.g. cholesterol / BP of 0)
    """
    df = df.drop_duplicates().copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        df[col] = df[col].round().astype("Int64")

    for col in ("trestbps", "chol"):
        df.loc[df[col] <= 0, col] = np.nan

    return df.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = clean(load_raw(download()))
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info("Wrote %d rows x %d cols to %s", *df.shape, CLEAN_PATH)
    logger.info("Missing values:\n%s", df.isna().sum()[lambda s: s > 0])
    logger.info("Class balance:\n%s", df[TARGET].value_counts())


if __name__ == "__main__":
    main()
