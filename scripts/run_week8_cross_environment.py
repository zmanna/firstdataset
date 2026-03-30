from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.week8_validation import run_cross_environment_validation, write_week8_charts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "week8_charts"
JSON_PATH = REPORTS_DIR / "week8_cross_environment_metrics.json"
CSV_PATH = REPORTS_DIR / "week8_cross_environment_summary.csv"
MD_PATH = PROJECT_ROOT / "WEEK8_CROSS_ENVIRONMENT_VALIDATION.md"


def dataframe_to_markdown(frame) -> str:
    columns = [str(column) for column in frame.columns]
    index_name = frame.index.name or "model_name"
    header = "| " + " | ".join([index_name] + columns) + " |"
    separator = "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"
    rows = []
    for index, values in frame.iterrows():
        formatted = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values.tolist()]
        rows.append("| " + " | ".join([str(index)] + formatted) + " |")
    return "\n".join([header, separator] + rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results, summary = run_cross_environment_validation()
    chart_paths = write_week8_charts(summary, CHARTS_DIR)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsar_biodegradation",
        "evaluation_type": "cross_environment_validation_proxy",
        "environment_definition": "stratified kmeans clustering on standardized descriptor vectors with 3 proxy environments",
        "fold_results": [
            {
                "model_name": result.model_name,
                "environment_id": result.environment_id,
                "metrics": result.metrics,
                "confusion_matrix": result.confusion_matrix,
            }
            for result in fold_results
        ],
        "charts": [str(path) for path in chart_paths],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    summary.to_csv(CSV_PATH, index=False)

    mean_scores = summary.groupby("model_name")[["accuracy", "f1_score", "roc_auc", "rb_recall"]].mean()
    rb_drop = summary.groupby("model_name")["rb_recall"].agg(["min", "max", "mean"])

    lines = [
        "# Week 8 Cross-Environment Validation",
        "",
        "## Objective",
        "Test whether the patterns learned by the current models generalize under a shifted data distribution.",
        "",
        "## Environment Definition",
        "Proxy environments were defined by running stratified k-means clustering with 3 clusters on standardized descriptor vectors, preserving both classes across the held-out environments.",
        "",
        "## Validation Setup",
        "- Leave-one-environment-out evaluation",
        "- Train on two clusters, test on the held-out cluster",
        "- Models: Logistic Regression, Random Forest, Week 6 FNN, Week 7 descriptor-graph prototype",
        "",
        "## Mean Performance Across Held-Out Environments",
        "",
        dataframe_to_markdown(mean_scores.round(4)),
        "",
        "## RB Recall Stability",
        "",
        dataframe_to_markdown(rb_drop.round(4)),
        "",
        "## Charts",
    ]
    for path in chart_paths:
        lines.append(f"- {path}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "These results show how performance changes when models are tested outside the descriptor region they were trained on. The most important signal is whether RB recall collapses under distribution shift.",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote Week 8 JSON to {JSON_PATH}")
    print(f"Wrote Week 8 CSV to {CSV_PATH}")
    print(f"Wrote Week 8 markdown to {MD_PATH}")


if __name__ == "__main__":
    main()
