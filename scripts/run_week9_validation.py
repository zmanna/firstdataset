from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.week9_validation import run_week9_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORTS_DIR / "week9_validation_metrics.json"
DIAGNOSTICS_CSV = REPORTS_DIR / "week9_fold_diagnostics.csv"
RESULTS_CSV = REPORTS_DIR / "week9_model_results.csv"
TXT_PATH = PROJECT_ROOT / "WEEK9_VALIDATION_SUMMARY.txt"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics, results = run_week9_validation()

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)

    summary = (
        results.groupby(["sampling", "model_name"])[["accuracy", "precision", "recall", "f1_score", "roc_auc", "rb_recall"]]
        .mean()
        .reset_index()
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsar_biodegradation",
        "validation_type": "5_fold_stratified_cross_validation",
        "diagnostics_csv": str(DIAGNOSTICS_CSV),
        "results_csv": str(RESULTS_CSV),
        "summary": summary.to_dict(orient="records"),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Week 9 Validation Summary",
        "",
        "Validation: 5-fold stratified cross-validation",
        "Sampling conditions: baseline, smote",
        "",
        "Fold diagnostics file:",
        f"  {DIAGNOSTICS_CSV}",
        "",
        "Mean metrics by model and sampling:",
    ]
    for _, row in summary.iterrows():
        lines.append(f"{row['sampling']} | {row['model_name']}")
        lines.append(f"  accuracy: {row['accuracy']:.4f}")
        lines.append(f"  precision: {row['precision']:.4f}")
        lines.append(f"  recall: {row['recall']:.4f}")
        lines.append(f"  f1_score: {row['f1_score']:.4f}")
        lines.append(f"  roc_auc: {row['roc_auc']:.4f}")
        lines.append(f"  rb_recall: {row['rb_recall']:.4f}")
        lines.append("")
    TXT_PATH.write_text("\n".join(lines).rstrip() + "\n")

    print(f"Wrote Week 9 JSON to {JSON_PATH}")
    print(f"Wrote Week 9 diagnostics CSV to {DIAGNOSTICS_CSV}")
    print(f"Wrote Week 9 results CSV to {RESULTS_CSV}")
    print(f"Wrote Week 9 text summary to {TXT_PATH}")


if __name__ == "__main__":
    main()
