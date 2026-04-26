from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .week8_validation import run_cross_environment_validation_for_arrays
from .week9_validation import run_week9_validation_for_arrays
from .week10_features import TIER1_BASELINE_FEATURES, TIER2_PROXY_FEATURES, build_week10_feature_bundle


@dataclass(frozen=True)
class Week11FeatureSets:
    full_enhanced: list[str]
    top_ranked: list[str]
    proxy_only: list[str]
    reduced_hybrid: list[str]


def compute_feature_rankings(*, random_state: int = 42) -> pd.DataFrame:
    bundle = build_week10_feature_bundle()
    X = bundle.enhanced_X
    y = bundle.y_binary

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    forest.fit(X_train, y_train)
    rf_importance = forest.feature_importances_

    perm = permutation_importance(
        forest,
        X_test,
        y_test,
        n_repeats=15,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mi = mutual_info_classif(X_scaled, y, random_state=random_state)

    ranking = pd.DataFrame(
        {
            "feature_name": X.columns,
            "rf_importance": rf_importance,
            "permutation_importance": perm.importances_mean,
            "mutual_information": mi,
        }
    )
    ranking["rf_rank"] = ranking["rf_importance"].rank(ascending=False, method="average")
    ranking["perm_rank"] = ranking["permutation_importance"].rank(ascending=False, method="average")
    ranking["mi_rank"] = ranking["mutual_information"].rank(ascending=False, method="average")
    ranking["combined_rank"] = ranking[["rf_rank", "perm_rank", "mi_rank"]].mean(axis=1)
    ranking = ranking.sort_values("combined_rank").reset_index(drop=True)
    return ranking


def build_week11_feature_sets(ranking: pd.DataFrame) -> Week11FeatureSets:
    top_ranked = ranking["feature_name"].head(15).tolist()

    baseline_ranked = ranking[ranking["feature_name"].isin(TIER1_BASELINE_FEATURES)]["feature_name"].head(8).tolist()
    proxy_ranked = ranking[ranking["feature_name"].isin(TIER2_PROXY_FEATURES)]["feature_name"].head(4).tolist()
    reduced_hybrid = list(dict.fromkeys(baseline_ranked + proxy_ranked))

    return Week11FeatureSets(
        full_enhanced=TIER1_BASELINE_FEATURES + TIER2_PROXY_FEATURES,
        top_ranked=top_ranked,
        proxy_only=list(TIER2_PROXY_FEATURES),
        reduced_hybrid=reduced_hybrid,
    )


def evaluate_feature_sets(*, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Week11FeatureSets]:
    bundle = build_week10_feature_bundle()
    ranking = compute_feature_rankings(random_state=random_state)
    feature_sets = build_week11_feature_sets(ranking)

    diagnostics_frames: list[pd.DataFrame] = []
    results_frames: list[pd.DataFrame] = []
    cross_env_rows: list[dict[str, float | str]] = []

    feature_map = {
        "full_enhanced": bundle.enhanced_X[feature_sets.full_enhanced],
        "top_ranked": bundle.enhanced_X[feature_sets.top_ranked],
        "proxy_only": bundle.enhanced_X[feature_sets.proxy_only],
        "reduced_hybrid": bundle.enhanced_X[feature_sets.reduced_hybrid],
    }

    for feature_set_name, frame in feature_map.items():
        diagnostics, results = run_week9_validation_for_arrays(
            frame.to_numpy(dtype=np.float64),
            bundle.y_binary,
            random_state=random_state,
        )
        diagnostics_frames.append(diagnostics.assign(feature_set=feature_set_name))
        results_frames.append(results.assign(feature_set=feature_set_name))

        _, cross_env_summary = run_cross_environment_validation_for_arrays(
            frame.to_numpy(dtype=np.float64),
            bundle.y_binary,
            random_state=random_state,
        )
        rf_cross = cross_env_summary[cross_env_summary["model_name"] == "random_forest_classifier"]
        cross_env_rows.append(
            {
                "feature_set": feature_set_name,
                "cross_env_rf_accuracy": float(rf_cross["accuracy"].mean()),
                "cross_env_rf_roc_auc": float(rf_cross["roc_auc"].mean()),
                "cross_env_rf_rb_recall": float(rf_cross["rb_recall"].mean()),
            }
        )

    diagnostics_all = pd.concat(diagnostics_frames, ignore_index=True)
    results_all = pd.concat(results_frames, ignore_index=True)
    cross_env_df = pd.DataFrame(cross_env_rows)
    return diagnostics_all, results_all, cross_env_df, feature_sets


def write_week11_charts(
    ranking: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    top_features = ranking.head(15).copy()
    top_features = top_features.sort_values("combined_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature_name"], top_features["rf_importance"], color="#4C78A8")
    ax.set_title("Week 11 Top Features by Random Forest Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path_1 = output_path / "week11_top_feature_importance.png"
    fig.savefig(path_1, dpi=200)
    plt.close(fig)
    chart_paths.append(path_1)

    summary = (
        results.groupby(["feature_set", "sampling", "model_name"])[["accuracy", "rb_recall", "roc_auc"]]
        .mean()
        .reset_index()
    )
    focused = summary[summary["sampling"] == "smote"]
    labels = [f"{row.feature_set}\n{row.model_name}" for row in focused.itertuples()]
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x, focused["accuracy"], width=0.6, color="#59A14F")
    axes[0].set_title("Week 11 Accuracy with SMOTE")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    axes[1].bar(x, focused["rb_recall"], width=0.6, color="#E15759")
    axes[1].set_title("Week 11 RB Recall with SMOTE")
    axes[1].set_ylabel("RB Recall")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    path_2 = output_path / "week11_feature_set_comparison.png"
    fig.savefig(path_2, dpi=200)
    plt.close(fig)
    chart_paths.append(path_2)

    return chart_paths
