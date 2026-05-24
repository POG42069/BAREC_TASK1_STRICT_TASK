from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "CAMeL-Lab/BAREC-Corpus-v1.0"
REQUIRED_COLUMNS = {"ID", "D3Tok", "Readability_Level_19"}
SPLIT_TO_FILE = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho bước tải dataset."""
    parser = argparse.ArgumentParser(description="Download the BAREC Corpus v1.0 dataset.")
    parser.add_argument("--output-dir", default="data/barec-corpus-v1", help="Directory to write split CSV files.")
    parser.add_argument("--dataset-id", default=DATASET_ID, help="Hugging Face dataset id.")
    return parser.parse_args()


def main() -> None:
    """Tải BAREC Corpus và ghi từng split ra CSV trong thư mục project."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)

    for split, filename in SPLIT_TO_FILE.items():
        if split not in dataset:
            raise ValueError(f"Dataset is missing expected split: {split}")

        # Mỗi split phải có ID câu, input D3Tok, và nhãn 19 mức.
        frame = dataset[split].to_pandas()
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Split {split} is missing required columns: {', '.join(missing)}")

        output_path = output_dir / filename
        frame.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Wrote {len(frame):,} rows to {output_path}")

    print("Smoke check passed: ID, D3Tok, and Readability_Level_19 are present.")


if __name__ == "__main__":
    main()
