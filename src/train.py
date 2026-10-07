"""Train, tune and compare classifiers; track everything in MLflow.

Usage:
    python -m src.train
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from src.data import CLEAN_PATH, FEATURES, ROOT, TARGET  # noqa: E402
from src.features import build_pipeline  # noqa: E402

SEED = 42
MODEL_DIR = ROOT / "models"
EXPERIMENT = "heart-disease-classification"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Candidate models and their hyper-parameter grids (keys prefixed for Pipeline)
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1.0, 10.0],
         "model__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 5]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(CLEAN_PATH)
    return df[FEATURES], df[TARGET]


def test_metrics(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def log_plots(model, X, y, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=axes[1])
    fig.suptitle(f"{name} – hold-out test set")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/{name}_roc_confusion.png")
    plt.close(fig)

    # feature importance / coefficients, mapped back to transformed names
    est = model.named_steps["model"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    weights = getattr(est, "feature_importances_", None)
    if weights is None:
        weights = est.coef_[0]
    imp = pd.Series(weights, index=names).sort_values(key=abs).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#3B6EA5")
    ax.set_title(f"{name} – top features")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/{name}_feature_importance.png")
    plt.close(fig)


def main() -> None:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # cross-validated metrics of the tuned model on the training split
            cv_scores = cross_validate(best, X_train, y_train, cv=cv,
                                       scoring=SCORING)
            metrics = {f"cv_{m}_mean": cv_scores[f"test_{m}"].mean()
                       for m in SCORING}
            metrics |= {f"cv_{m}_std": cv_scores[f"test_{m}"].std()
                        for m in SCORING}
            metrics |= test_metrics(best, X_test, y_test)

            mlflow.set_tag("model_family", name)
            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_param("grid_size", len(search.cv_results_["params"]))
            mlflow.log_metrics(metrics)
            pd.DataFrame(search.cv_results_).to_csv("/tmp/cv_results.csv",
                                                    index=False)
            mlflow.log_artifact("/tmp/cv_results.csv", "tuning")
            log_plots(best, X_test, y_test, name)
            mlflow.sklearn.log_model(best, name="model",
                                     input_example=X_test.head(3),
                                     # we produced these objects ourselves
                                     skops_trusted_types=[
                                         "numpy.dtype",
                                         "sklearn.tree._tree.Tree"])

            results[name] = {"run_id": run.info.run_id, "model": best,
                             "params": search.best_params_, **metrics}
            print(f"{name:22s} CV ROC-AUC={metrics['cv_roc_auc_mean']:.3f}"
                  f"±{metrics['cv_roc_auc_std']:.3f}  "
                  f"test ROC-AUC={metrics['test_roc_auc']:.3f}  "
                  f"test acc={metrics['test_accuracy']:.3f}")

    # Selection uses CV score (not the test set) to avoid optimistic bias
    winner = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best = results[winner]
    print(f"\nSelected model: {winner}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "model.joblib")
    meta = {
        "model_name": winner,
        "mlflow_run_id": best["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "best_params": {k: (v if v is None or isinstance(v, (int, float, str))
                            else str(v)) for k, v in best["params"].items()},
        "metrics": {k: round(v, 4) for k, v in best.items()
                    if k.startswith(("cv_", "test_"))},
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    comparison = pd.DataFrame({n: {k: v for k, v in r.items()
                                   if k.startswith(("cv_", "test_"))}
                               for n, r in results.items()}).T.round(4)
    comparison.to_csv(ROOT / "reports" / "model_comparison.csv")
    print(comparison[["cv_roc_auc_mean", "cv_accuracy_mean", "test_roc_auc",
                      "test_accuracy", "test_precision", "test_recall"]])


if __name__ == "__main__":
    main()
