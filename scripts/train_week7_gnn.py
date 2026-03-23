from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.modeling import run_qsar_classification_baselines, run_qsar_fnn_classifier
from firstdataset.week7_gnn import run_week7_descriptor_graph_prototype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORTS_DIR / "week7_gnn_metrics.json"
TXT_PATH = PROJECT_ROOT / "WEEK7_GNN_RESULTS.txt"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results = run_qsar_classification_baselines()
    fnn_result = run_qsar_fnn_classifier()
    gnn_result = run_week7_descriptor_graph_prototype()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsar_biodegradation",
        "task_type": "classification",
        "prototype_scope": "Descriptor-graph GNN prototype adapted to the available descriptor dataset.",
        "baselines": [
            {"model_name": result.model_name, "metrics": result.metrics}
            for result in baseline_results
        ],
        "fnn": {
            "model_name": fnn_result.model_name,
            "metrics": fnn_result.metrics,
        },
        "week7_gnn": {
            "model_name": gnn_result.model_name,
            "metrics": gnn_result.metrics,
            "confusion_matrix_labels": ["NRB", "RB"],
            "confusion_matrix": gnn_result.confusion_matrix,
            "graph_info": gnn_result.graph_info,
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Week 7 GNN Prototype Results",
        "",
        "Dataset: QSAR biodegradation",
        "Prototype: descriptor-graph message-passing network",
        "Note: this is an adapted graph prototype because the current dataset has descriptor vectors, not polymer atom/bond graphs.",
        "",
        "Week 5 baselines:",
    ]
    for result in baseline_results:
        lines.append(result.model_name)
        for metric_name, value in result.metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            "Week 6 FNN:",
            fnn_result.model_name,
        ]
    )
    for metric_name, value in fnn_result.metrics.items():
        lines.append(f"  {metric_name}: {value:.4f}")
    lines.append("")

    lines.extend(
        [
            "Week 7 graph prototype:",
            gnn_result.model_name,
        ]
    )
    for metric_name, value in gnn_result.metrics.items():
        lines.append(f"  {metric_name}: {value:.4f}")
    lines.append("")
    lines.append("Confusion matrix (rows=true [NRB, RB], cols=predicted [NRB, RB]):")
    for row in gnn_result.confusion_matrix:
        lines.append(f"  {row}")
    lines.append("")
    lines.append("Graph info:")
    for key, value in gnn_result.graph_info.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    TXT_PATH.write_text("\n".join(lines))
    print(f"Wrote Week 7 JSON to {JSON_PATH}")
    print(f"Wrote Week 7 text results to {TXT_PATH}")


if __name__ == "__main__":
    main()
