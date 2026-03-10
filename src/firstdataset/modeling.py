from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import split_qsar_biodegradation, split_tabular_regression_dataset


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    task_type: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ClassificationResult:
    model_name: str
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]


def run_qsar_classification_baselines(random_state: int = 42) -> list[BaselineResult]:
    split = split_qsar_biodegradation(random_state=random_state)

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }

    results: list[BaselineResult] = []
    for name, model in models.items():
        model.fit(split.X_train, split.y_train)
        predictions = model.predict(split.X_test)
        metrics = {
            "accuracy": float(accuracy_score(split.y_test, predictions)),
            "macro_f1": float(f1_score(split.y_test, predictions, average="macro")),
        }
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(split.X_test)
            positive_index = list(model.classes_).index("RB")
            y_true = (split.y_test == "RB").astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_true, probas[:, positive_index]))
        results.append(BaselineResult(model_name=name, task_type="classification", metrics=metrics))
    return results


def run_qsar_fnn_classifier(random_state: int = 42) -> ClassificationResult:
    split = split_qsar_biodegradation(random_state=random_state, target_as_category=False)
    y_train = (split.y_train == 2).astype(int)
    y_test = (split.y_test == 2).astype(int)
    model = Pipeline(
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
    )

    model.fit(split.X_train, y_train)
    predictions = model.predict(split.X_test)
    probabilities = model.predict_proba(split.X_test)
    positive_index = list(model.classes_).index(1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=1)),
        "recall": float(recall_score(y_test, predictions, pos_label=1)),
        "f1_score": float(f1_score(y_test, predictions, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, probabilities[:, positive_index])),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return ClassificationResult(
        model_name="feedforward_neural_network",
        metrics=metrics,
        confusion_matrix=matrix,
    )


def run_regression_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int = 42,
) -> list[BaselineResult]:
    models = {
        "linear_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }

    results: list[BaselineResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, predictions)),
        }
        results.append(BaselineResult(model_name=name, task_type="regression", metrics=metrics))
    return results


def run_regression_baselines_from_csv(
    csv_path: str,
    *,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_missing: bool = True,
) -> list[BaselineResult]:
    split = split_tabular_regression_dataset(
        csv_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        drop_missing=drop_missing,
    )
    return run_regression_baselines(
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
        random_state=random_state,
    )
