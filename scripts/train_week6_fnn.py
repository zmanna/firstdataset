from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.modeling import run_qsar_classification_baselines, run_qsar_fnn_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORTS_DIR / "week6_fnn_metrics.json"
TXT_PATH = PROJECT_ROOT / "WEEK6_FNN_RESULTS.txt"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results = run_qsar_classification_baselines()
    fnn_result = run_qsar_fnn_classifier()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsar_biodegradation",
        "task_type": "classification",
        "train_test_split": {
            "test_size": 0.2,
            "stratified": True,
            "random_state": 42,
        },
        "baselines": [
            {
                "model_name": result.model_name,
                "metrics": result.metrics,
            }
            for result in baseline_results
        ],
        "fnn": {
            "model_name": fnn_result.model_name,
            "metrics": fnn_result.metrics,
            "confusion_matrix_labels": ["NRB", "RB"],
            "confusion_matrix": fnn_result.confusion_matrix,
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Week 6 FNN Results",
        "",
        "Dataset: QSAR biodegradation",
        "Task: binary classification (RB vs NRB)",
        "Train/test split: 80/20 stratified, random_state=42",
        "",
        "Baseline comparison:",
    ]
    for result in baseline_results:
        lines.append(result.model_name)
        for metric_name, value in result.metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            "Feedforward neural network:",
            fnn_result.model_name,
        ]
    )
    for metric_name, value in fnn_result.metrics.items():
        lines.append(f"  {metric_name}: {value:.4f}")
    lines.append("")
    lines.append("Confusion matrix (rows=true [NRB, RB], cols=predicted [NRB, RB]):")
    for row in fnn_result.confusion_matrix:
        lines.append(f"  {row}")
    lines.append("")

    TXT_PATH.write_text("\n".join(lines))

    print(f"Wrote Week 6 JSON to {JSON_PATH}")
    print(f"Wrote Week 6 text results to {TXT_PATH}")


if __name__ == "__main__":
    main()
