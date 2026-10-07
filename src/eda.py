"""Exploratory data analysis: writes figures to reports/figures/.

Usage:
    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import (  # noqa: E402
    CATEGORICAL_FEATURES, CLEAN_PATH, NUMERIC_FEATURES, ROOT, TARGET,
)

FIG_DIR = ROOT / "reports" / "figures"
PALETTE = {0: "#4C9F70", 1: "#C8553D"}
LABELS = {0: "No disease", 1: "Disease"}


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def class_balance(df: pd.DataFrame) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, n in zip(bars, counts.values, strict=True):
        ax.annotate(f"{n} ({n / counts.sum():.0%})",
                    (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "class_balance.png")


def numeric_histograms(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(18, 3.6))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        for cls in (0, 1):
            ax.hist(df.loc[df[TARGET] == cls, col].dropna(), bins=20,
                    alpha=0.6, color=PALETTE[cls], label=LABELS[cls])
        ax.set_title(col)
    axes[0].legend()
    fig.suptitle("Numeric feature distributions by outcome")
    return _save(fig, "numeric_histograms.png")


def categorical_rates(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(CATEGORICAL_FEATURES), figsize=(18, 3.6))
    for ax, col in zip(axes, CATEGORICAL_FEATURES, strict=True):
        rate = df.groupby(col)[TARGET].mean()
        ax.bar(rate.index.astype(int).astype(str), rate.values, color="#C8553D")
        ax.set_ylim(0, 1)
        ax.set_title(f"{col}: disease rate")
    return _save(fig, "categorical_disease_rate.png")


def correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df.astype(float).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation (Pearson)")
    return _save(fig, "correlation_heatmap.png")


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)
    print(df.describe().T.round(2))
    for fn in (class_balance, numeric_histograms, categorical_rates,
               correlation_heatmap):
        print("saved", fn(df))


if __name__ == "__main__":
    main()
