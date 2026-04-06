from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .data import load_qsar_biodegradation


@dataclass(frozen=True)
class FoldDiagnostics:
    fold_id: int
    train_size: int
    test_size: int
    train_nrb: int
    train_rb: int
    test_nrb: int
    test_rb: int


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.bincount(y)
    minority_count = int(class_counts.min())
    if minority_count < 2 or class_counts[0] == class_counts[1]:
        return X, y

    sampler = SMOTE(
        random_state=random_state,
        k_neighbors=min(k_neighbors, minority_count - 1),
    )
    return sampler.fit_resample(X, y)


def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> tuple[dict[str, float], list[list[int]]]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "rb_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics, matrix


def run_week9_validation(*, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = load_qsar_biodegradation(target_as_category=False)
    X = bundle.X.to_numpy(dtype=np.float64)
    y = (bundle.y.to_numpy(dtype=np.int64) == 2).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    diagnostics_rows: list[dict[str, int]] = []
    result_rows: list[dict[str, float | int | str]] = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        diagnostics_rows.append(
            {
                "fold_id": fold_id,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_nrb": int((y_train == 0).sum()),
                "train_rb": int((y_train == 1).sum()),
                "test_nrb": int((y_test == 0).sum()),
                "test_rb": int((y_test == 1).sum()),
            }
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        for sampling_name, (X_fold_train, y_fold_train) in {
            "baseline": (X_train_scaled, y_train),
            "smote": apply_smote(X_train_scaled, y_train, random_state=random_state + fold_id),
        }.items():
            models = {
                "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
                "random_forest_classifier": RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
                "feedforward_neural_network": MLPClassifier(
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
            }

            for model_name, model in models.items():
                model.fit(X_fold_train, y_fold_train)
                predictions = model.predict(X_test_scaled)
                scores = model.predict_proba(X_test_scaled)[:, 1]
                metrics, matrix = _evaluate_binary(y_test, predictions, scores)

                row: dict[str, float | int | str] = {
                    "fold_id": fold_id,
                    "sampling": sampling_name,
                    "model_name": model_name,
                    "tn": matrix[0][0],
                    "fp": matrix[0][1],
                    "fn": matrix[1][0],
                    "tp": matrix[1][1],
                }
                row.update(metrics)
                result_rows.append(row)

    return pd.DataFrame(diagnostics_rows), pd.DataFrame(result_rows)
