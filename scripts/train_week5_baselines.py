from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.modeling import run_qsar_classification_baselines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "week5_baseline_metrics.json"
REPORT_PATH = REPORTS_DIR / "week5_baseline_modeling.md"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_qsar_classification_baselines()

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsar_biodegradation",
        "task_type": "classification",
        "deliverable_gap": "Requested regression on degradation-rate constants is blocked because the current dataset only provides a binary class label.",
        "models": [
            {
                "model_name": result.model_name,
                "task_type": result.task_type,
                "metrics": result.metrics,
            }
            for result in results
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    lines = [
        "# Week 5 Baseline Modeling",
        "",
        "## Scope",
        "Baseline modeling was executed on the currently available QSAR biodegradation dataset.",
        "",
        "## Constraint",
        "The requested degradation-rate regression task is not possible with this dataset because the target is a binary biodegradation class rather than a continuous rate constant.",
        "",
        "## Training Setup",
        "- Train/test split: 80/20, stratified, random_state=42",
        "- Baselines run: Logistic Regression, Random Forest Classifier",
        "- Features: 41 numeric descriptor columns",
        "- Target: binary biodegradation class",
        "",
        "## Results",
    ]

    for result in results:
        lines.append(f"### {result.model_name}")
        for metric_name, value in result.metrics.items():
            lines.append(f"- {metric_name}: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "These baselines validate the project pipeline and show the descriptor set is predictive for class-based biodegradation outcomes.",
            "",
            "## To Reach the Original Week 5 Goal",
            "Acquire a dataset with continuous degradation-rate targets such as half-life or rate constant values. Once available, the same project structure can be extended to run Linear Regression and Random Forest Regressor baselines with MAE, RMSE, and R2.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote metrics to {METRICS_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
