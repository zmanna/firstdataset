from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.modeling import run_regression_baselines_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline regression models on a tabular CSV dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--target", required=True, help="Numeric target column name.")
    parser.add_argument(
        "--output",
        default="regression_scores.json",
        help="Path for the output JSON file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Disable dropping rows with missing values before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_regression_baselines_from_csv(
        args.csv,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_missing=not args.keep_missing,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(Path(args.csv).resolve()),
        "target_column": args.target,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "models": [
            {
                "model_name": result.model_name,
                "task_type": result.task_type,
                "metrics": result.metrics,
            }
            for result in results
        ],
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote regression scores to {output_path}")


if __name__ == "__main__":
    main()
