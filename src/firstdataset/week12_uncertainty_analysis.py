from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .week8_validation import build_proxy_environments_for_arrays
from .week10_features import build_week10_feature_bundle
from .week11_analysis import build_week11_feature_sets, compute_feature_rankings


SELECTIVE_COVERAGE_LEVELS = (1.0, 0.9, 0.75, 0.5, 0.25)


@dataclass(frozen=True)
class Week12FeatureMap:
    feature_frames: dict[str, pd.DataFrame]
    y_binary: np.ndarray


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, 1e-6, 1.0 - 1e-6)


def _confidence_from_scores(y_score: np.ndarray) -> np.ndarray:
    return np.maximum(y_score, 1.0 - y_score)


def _entropy_from_scores(y_score: np.ndarray) -> np.ndarray:
    p = _clip_probabilities(y_score)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bins[1:-1], right=True)
    total = len(y_true)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        accuracy = np.mean(y_true[mask] == y_pred[mask])
        avg_confidence = float(confidence[mask].mean())
        error += (mask.sum() / total) * abs(avg_confidence - accuracy)
    return float(error)


def _build_models(random_state: int) -> dict[str, object]:
    return {
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


def _build_feature_map(*, random_state: int = 42) -> Week12FeatureMap:
    bundle = build_week10_feature_bundle()
    ranking = compute_feature_rankings(random_state=random_state)
    feature_sets = build_week11_feature_sets(ranking)
    feature_frames = {
        "full_enhanced": bundle.enhanced_X[feature_sets.full_enhanced],
        "top_ranked": bundle.enhanced_X[feature_sets.top_ranked],
        "proxy_only": bundle.enhanced_X[feature_sets.proxy_only],
        "reduced_hybrid": bundle.enhanced_X[feature_sets.reduced_hybrid],
    }
    return Week12FeatureMap(feature_frames=feature_frames, y_binary=bundle.y_binary)


def _prediction_frame(
    *,
    feature_set: str,
    model_name: str,
    evaluation_type: str,
    split_id: int,
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    confidence = _confidence_from_scores(y_score)
    uncertainty = 1.0 - confidence
    entropy = _entropy_from_scores(y_score)
    correct = (y_true == y_pred).astype(int)
    return pd.DataFrame(
        {
            "feature_set": feature_set,
            "model_name": model_name,
            "evaluation_type": evaluation_type,
            "split_id": split_id,
            "sample_index": sample_indices.astype(int),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "probability_rb": y_score.astype(float),
            "confidence": confidence.astype(float),
            "uncertainty": uncertainty.astype(float),
            "entropy_uncertainty": entropy.astype(float),
            "correct": correct.astype(int),
        }
    )


def _summarize_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    y_score = frame["probability_rb"].to_numpy(dtype=float)
    confidence = frame["confidence"].to_numpy(dtype=float)
    correct_mask = frame["correct"] == 1
    incorrect_mask = ~correct_mask

    if incorrect_mask.any():
        mean_uncertainty_incorrect = float(frame.loc[incorrect_mask, "uncertainty"].mean())
        mean_entropy_incorrect = float(frame.loc[incorrect_mask, "entropy_uncertainty"].mean())
        mean_confidence_incorrect = float(frame.loc[incorrect_mask, "confidence"].mean())
    else:
        mean_uncertainty_incorrect = 0.0
        mean_entropy_incorrect = 0.0
        mean_confidence_incorrect = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "rb_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - y_score, y_score]), labels=[0, 1])),
        "ece": _expected_calibration_error(y_true, y_pred, confidence),
        "mean_confidence": float(frame["confidence"].mean()),
        "mean_uncertainty": float(frame["uncertainty"].mean()),
        "mean_entropy": float(frame["entropy_uncertainty"].mean()),
        "mean_uncertainty_correct": float(frame.loc[correct_mask, "uncertainty"].mean()),
        "mean_uncertainty_incorrect": mean_uncertainty_incorrect,
        "mean_entropy_correct": float(frame.loc[correct_mask, "entropy_uncertainty"].mean()),
        "mean_entropy_incorrect": mean_entropy_incorrect,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "incorrect_count": float((~correct_mask).sum()),
    }


def _selective_prediction_rows(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    ordered = frame.sort_values("confidence", ascending=False).reset_index(drop=True)
    rows: list[dict[str, float | str]] = []
    for coverage in SELECTIVE_COVERAGE_LEVELS:
        keep = max(1, int(np.ceil(len(ordered) * coverage)))
        subset = ordered.iloc[:keep]
        rows.append(
            {
                "feature_set": str(frame["feature_set"].iloc[0]),
                "model_name": str(frame["model_name"].iloc[0]),
                "evaluation_type": str(frame["evaluation_type"].iloc[0]),
                "coverage": float(coverage),
                "kept_predictions": int(keep),
                "accuracy": float((subset["y_true"] == subset["y_pred"]).mean()),
                "rb_recall": float(
                    recall_score(
                        subset["y_true"],
                        subset["y_pred"],
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "mean_confidence": float(subset["confidence"].mean()),
                "mean_uncertainty": float(subset["uncertainty"].mean()),
            }
        )
    return rows


def run_week12_uncertainty_analysis(
    *,
    random_state: int = 42,
    model_names: tuple[str, ...] = (
        "random_forest_classifier",
        "logistic_regression",
        "feedforward_neural_network",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_map = _build_feature_map(random_state=random_state)
    available_models = _build_models(random_state)
    selected_models = {name: available_models[name] for name in model_names}

    prediction_frames: list[pd.DataFrame] = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for feature_set, frame in feature_map.feature_frames.items():
        X = frame.to_numpy(dtype=np.float64)
        y = feature_map.y_binary

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            for model_name, model in selected_models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_score = model.predict_proba(X_test_scaled)[:, 1]
                prediction_frames.append(
                    _prediction_frame(
                        feature_set=feature_set,
                        model_name=model_name,
                        evaluation_type="stratified_cv",
                        split_id=fold_id,
                        sample_indices=test_idx,
                        y_true=y_test,
                        y_pred=y_pred,
                        y_score=y_score,
                    )
                )

        environment_labels = build_proxy_environments_for_arrays(
            X,
            y,
            n_environments=3,
            random_state=random_state,
        )
        for environment_id in range(3):
            train_mask = environment_labels != environment_id
            test_mask = environment_labels == environment_id
            X_train_raw = X[train_mask]
            X_test_raw = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            for model_name, model in selected_models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_score = model.predict_proba(X_test_scaled)[:, 1]
                sample_indices = np.flatnonzero(test_mask)
                prediction_frames.append(
                    _prediction_frame(
                        feature_set=feature_set,
                        model_name=model_name,
                        evaluation_type="cross_environment",
                        split_id=environment_id,
                        sample_indices=sample_indices,
                        y_true=y_test,
                        y_pred=y_pred,
                        y_score=y_score,
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)

    metrics_rows: list[dict[str, float | str]] = []
    selective_rows: list[dict[str, float | str]] = []
    cross_env_rows: list[dict[str, float | str]] = []

    grouped = predictions.groupby(["feature_set", "model_name", "evaluation_type"], sort=False)
    for (feature_set, model_name, evaluation_type), frame in grouped:
        metrics = _summarize_prediction_frame(frame)
        row: dict[str, float | str] = {
            "feature_set": feature_set,
            "model_name": model_name,
            "evaluation_type": evaluation_type,
        }
        row.update(metrics)
        metrics_rows.append(row)
        selective_rows.extend(_selective_prediction_rows(frame))
        if evaluation_type == "cross_environment":
            cross_env_rows.append(row.copy())

    metrics_df = pd.DataFrame(metrics_rows)
    selective_df = pd.DataFrame(selective_rows)
    cross_env_df = pd.DataFrame(cross_env_rows)
    return predictions, metrics_df, selective_df, cross_env_df


def write_week12_charts(
    predictions: pd.DataFrame,
    selective: pd.DataFrame,
    output_dir: str | Path,
    *,
    primary_model: str = "random_forest_classifier",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    rf_predictions = predictions[
        (predictions["model_name"] == primary_model)
        & (predictions["evaluation_type"] == "stratified_cv")
    ].copy()
    feature_sets = list(dict.fromkeys(rf_predictions["feature_set"].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feature_set in zip(axes.flatten(), feature_sets):
        frame = rf_predictions[rf_predictions["feature_set"] == feature_set]
        frac_pos, mean_pred = calibration_curve(
            frame["y_true"],
            frame["probability_rb"],
            n_bins=10,
            strategy="quantile",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, marker="o", color="#4C78A8", label=feature_set)
        ax.set_title(feature_set.replace("_", " ").title())
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed RB frequency")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left")
    fig.tight_layout()
    calibration_path = output_path / "week12_calibration_curve.png"
    fig.savefig(calibration_path, dpi=200)
    plt.close(fig)
    chart_paths.append(calibration_path)

    rf_selective = selective[
        (selective["model_name"] == primary_model)
        & (selective["evaluation_type"] == "stratified_cv")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_set in feature_sets:
        frame = rf_selective[rf_selective["feature_set"] == feature_set].sort_values("coverage")
        ax.plot(frame["coverage"], frame["accuracy"], marker="o", label=feature_set)
    ax.set_title("Week 12 Selective Prediction Accuracy")
    ax.set_xlabel("Coverage kept")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    selective_path = output_path / "week12_selective_accuracy.png"
    fig.savefig(selective_path, dpi=200)
    plt.close(fig)
    chart_paths.append(selective_path)

    summary_rows = []
    for feature_set in feature_sets:
        frame = rf_predictions[rf_predictions["feature_set"] == feature_set]
        correct = frame[frame["correct"] == 1]["uncertainty"].mean()
        incorrect = frame[frame["correct"] == 0]["uncertainty"].mean()
        summary_rows.append(
            {
                "feature_set": feature_set,
                "correct_uncertainty": float(correct),
                "incorrect_uncertainty": float(incorrect),
            }
        )
    uncertainty_df = pd.DataFrame(summary_rows)
    x = np.arange(len(uncertainty_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, uncertainty_df["correct_uncertainty"], width=width, label="Correct")
    ax.bar(x + width / 2, uncertainty_df["incorrect_uncertainty"], width=width, label="Incorrect")
    ax.set_title("Week 12 Uncertainty on Correct vs Incorrect Predictions")
    ax.set_ylabel("Mean uncertainty")
    ax.set_xticks(x)
    ax.set_xticklabels(uncertainty_df["feature_set"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    uncertainty_path = output_path / "week12_uncertainty_correct_vs_incorrect.png"
    fig.savefig(uncertainty_path, dpi=200)
    plt.close(fig)
    chart_paths.append(uncertainty_path)

    return chart_paths
