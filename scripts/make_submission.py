from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from barec.submission import build_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Codabench prediction.zip submission.")
    parser.add_argument("--predictions", required=True, help="CSV containing Sentence ID/ID and Prediction columns.")
    parser.add_argument("--output-dir", default="submission", help="Directory for prediction and prediction.zip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_file, zip_file = build_submission(args.predictions, args.output_dir)
    print(f"Wrote Codabench prediction file: {prediction_file}")
    print(f"Wrote Codabench zip file: {zip_file}")


if __name__ == "__main__":
    main()
