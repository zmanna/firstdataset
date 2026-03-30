from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_qsar_biodegradation
from .week7_gnn import run_descriptor_graph_prototype_on_split


@dataclass(frozen=True)
class Week8FoldResult:
    model_name: str
    environment_id: int
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]


def _evaluate_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> tuple[dict[str, float], list[list[int]]]:
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "rb_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics, matrix


def _build_proxy_environments(X: np.ndarray, y: np.ndarray, n_environments: int, random_state: int) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    environment_labels = np.empty(len(y), dtype=int)
    for class_id in np.unique(y):
        class_mask = y == class_id
        class_points = X_scaled[class_mask]
        kmeans = KMeans(n_clusters=n_environments, random_state=random_state, n_init=20)
        environment_labels[class_mask] = kmeans.fit_predict(class_points)
    return environment_labels


def _fit_and_eval_sklearn_model(
    model_name: str,
    model: Pipeline | RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], list[list[int]]]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return _evaluate_binary_predictions(y_test, predictions, scores)


def run_cross_environment_validation(
    *,
    n_environments: int = 3,
    random_state: int = 42,
) -> tuple[list[Week8FoldResult], pd.DataFrame]:
    bundle = load_qsar_biodegradation(target_as_category=False)
    X = bundle.X.to_numpy(dtype=np.float64)
    y = (bundle.y.to_numpy(dtype=np.int64) == 2).astype(int)

    environment_labels = _build_proxy_environments(X, y, n_environments, random_state)

    results: list[Week8FoldResult] = []
    for environment_id in range(n_environments):
        train_mask = environment_labels != environment_id
        test_mask = environment_labels == environment_id

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models: list[tuple[str, Pipeline | RandomForestClassifier]] = [
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
                    ]
                ),
            ),
            (
                "random_forest_classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "feedforward_neural_network",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPClassifier(
                                hidden_layer_sizes=(128, 64, 32),
                                activation="relu",
                                solver="adam",
                                alpha=1e-4,
                                batch_size=32,
                                learning_rate_init=1e-3,
                                max_iter=500,
                                early_stopping=True,
                                validation_fraction=0.1,
                                n_iter_no_change=20,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
        ]

        for model_name, model in models:
            metrics, matrix = _fit_and_eval_sklearn_model(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )
            results.append(
                Week8FoldResult(
                    model_name=model_name,
                    environment_id=environment_id,
                    metrics=metrics,
                    confusion_matrix=matrix,
                )
            )

        gnn_result = run_descriptor_graph_prototype_on_split(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            random_state=random_state,
        )
        results.append(
            Week8FoldResult(
                model_name=gnn_result.model_name,
                environment_id=environment_id,
                metrics=gnn_result.metrics,
                confusion_matrix=gnn_result.confusion_matrix,
            )
        )

    rows = []
    for result in results:
        row = {"model_name": result.model_name, "environment_id": result.environment_id}
        row.update(result.metrics)
        rows.append(row)
    summary = pd.DataFrame(rows)
    return results, summary


def write_week8_charts(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    mean_scores = summary.groupby("model_name")[["accuracy", "rb_recall", "roc_auc"]].mean().sort_values(
        "accuracy", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_scores.plot(kind="bar", ax=ax)
    ax.set_title("Week 8 Mean Cross-Environment Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path_1 = output_path / "week8_mean_scores.png"
    fig.savefig(path_1, dpi=200)
    plt.close(fig)
    chart_paths.append(path_1)

    pivot = summary.pivot(index="environment_id", columns="model_name", values="rb_recall")
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Week 8 RB Recall by Held-Out Environment")
    ax.set_xlabel("Model")
    ax.set_ylabel("Held-Out Environment")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Env {idx}" for idx in pivot.index])
    fig.colorbar(image, ax=ax, label="RB Recall")
    fig.tight_layout()
    path_2 = output_path / "week8_rb_recall_heatmap.png"
    fig.savefig(path_2, dpi=200)
    plt.close(fig)
    chart_paths.append(path_2)

    return chart_paths
