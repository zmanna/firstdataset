from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.week11_analysis import (
    build_week11_feature_sets,
    compute_feature_rankings,
    evaluate_feature_sets,
    write_week11_charts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "week11_charts"
RANKING_CSV = REPORTS_DIR / "week11_feature_rankings.csv"
RESULTS_CSV = REPORTS_DIR / "week11_feature_set_results.csv"
DIAGNOSTICS_CSV = REPORTS_DIR / "week11_feature_set_diagnostics.csv"
GENERALIZATION_CSV = REPORTS_DIR / "week11_feature_set_generalization.csv"
FEATURES_JSON = REPORTS_DIR / "week11_feature_sets.json"
TXT_PATH = PROJECT_ROOT / "WEEK11_FEATURE_ANALYSIS_SUMMARY.txt"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ranking = compute_feature_rankings()
    diagnostics, results, generalization, feature_sets = evaluate_feature_sets()
    chart_paths = write_week11_charts(ranking, results, CHARTS_DIR)

    ranking.to_csv(RANKING_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    generalization.to_csv(GENERALIZATION_CSV, index=False)

    feature_sets_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_enhanced": feature_sets.full_enhanced,
        "top_ranked": feature_sets.top_ranked,
        "proxy_only": feature_sets.proxy_only,
        "reduced_hybrid": feature_sets.reduced_hybrid,
    }
    FEATURES_JSON.write_text(json.dumps(feature_sets_payload, indent=2) + "\n")

    summary = (
        results.groupby(["feature_set", "sampling", "model_name"])[["accuracy", "precision", "recall", "f1_score", "roc_auc", "rb_recall"]]
        .mean()
        .reset_index()
    )

    lines = [
        "Week 11 Feature Importance and Selection Summary",
        "",
        "Top ranked features:",
    ]
    for _, row in ranking.head(15).iterrows():
        lines.append(
            f"  - {row['feature_name']} | rf={row['rf_importance']:.4f} | perm={row['permutation_importance']:.4f} | mi={row['mutual_information']:.4f}"
        )
    lines.extend(
        [
            "",
            "Feature sets evaluated:",
            f"  full_enhanced ({len(feature_sets.full_enhanced)} features)",
            f"  top_ranked ({len(feature_sets.top_ranked)} features)",
            f"  proxy_only ({len(feature_sets.proxy_only)} features)",
            f"  reduced_hybrid ({len(feature_sets.reduced_hybrid)} features)",
            "",
            "Cross-environment Random Forest summary by feature set:",
        ]
    )
    for _, row in generalization.iterrows():
        lines.append(f"{row['feature_set']}")
        lines.append(f"  cross_env_rf_accuracy: {row['cross_env_rf_accuracy']:.4f}")
        lines.append(f"  cross_env_rf_roc_auc: {row['cross_env_rf_roc_auc']:.4f}")
        lines.append(f"  cross_env_rf_rb_recall: {row['cross_env_rf_rb_recall']:.4f}")
        lines.append("")
    lines.extend(
        [
            "Mean metrics by feature set, sampling, and model:",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(f"{row['feature_set']} | {row['sampling']} | {row['model_name']}")
        lines.append(f"  accuracy: {row['accuracy']:.4f}")
        lines.append(f"  precision: {row['precision']:.4f}")
        lines.append(f"  recall: {row['recall']:.4f}")
        lines.append(f"  f1_score: {row['f1_score']:.4f}")
        lines.append(f"  roc_auc: {row['roc_auc']:.4f}")
        lines.append(f"  rb_recall: {row['rb_recall']:.4f}")
        lines.append("")
    lines.append("Charts:")
    for path in chart_paths:
        lines.append(f"  {path}")
    TXT_PATH.write_text("\n".join(lines).rstrip() + "\n")

    print(f"Wrote Week 11 rankings CSV to {RANKING_CSV}")
    print(f"Wrote Week 11 diagnostics CSV to {DIAGNOSTICS_CSV}")
    print(f"Wrote Week 11 results CSV to {RESULTS_CSV}")
    print(f"Wrote Week 11 generalization CSV to {GENERALIZATION_CSV}")
    print(f"Wrote Week 11 feature set JSON to {FEATURES_JSON}")
    print(f"Wrote Week 11 summary to {TXT_PATH}")
    for path in chart_paths:
        print(f"Wrote Week 11 chart to {path}")


if __name__ == "__main__":
    main()
