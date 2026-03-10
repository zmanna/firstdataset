from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import split_qsar_biodegradation, split_tabular_regression_dataset


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    task_type: str
    metrics: dict[str, float]


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
