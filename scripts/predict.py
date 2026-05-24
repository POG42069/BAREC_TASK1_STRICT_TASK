from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextDataset(Dataset):
    """Dataset nhỏ để tokenizer từng câu khi chạy inference."""

    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize một câu và trả tensor cho DataLoader."""
        return self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Ghép các sample đơn lẻ thành một batch tensor."""
    keys = batch[0].keys()
    return {key: torch.cat([item[key] for item in batch], dim=0) for key in keys}


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho bước dự đoán bằng model đã train."""
    parser = argparse.ArgumentParser(description="Predict BAREC readability labels with a trained model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing the trained model.")
    parser.add_argument("--input-file", required=True, help="CSV file with ID and input text columns.")
    parser.add_argument("--output-file", required=True, help="CSV file to write Sentence ID and Prediction.")
    parser.add_argument("--input-column", default="D3Tok", help="Text column to use for prediction.")
    parser.add_argument("--id-column", default="ID", help="Sentence id column.")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    return parser.parse_args()


def main() -> None:
    """Load model đã train, dự đoán nhãn 1..19, và ghi CSV kết quả."""
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    for column in (args.id_column, args.input_column):
        if column not in frame.columns:
            raise ValueError(f"Input file is missing required column: {column}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = frame[args.input_column].fillna("").astype(str).tolist()
    dataset = TextDataset(texts, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    predictions: list[int] = []
    # Tắt gradient để inference nhanh hơn và tiết kiệm bộ nhớ.
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            labels = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(label + 1 for label in labels)
            print(f"Predicted batch {step}/{len(loader)}")

    output = pd.DataFrame({"Sentence ID": frame[args.id_column], "Prediction": predictions})
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(output):,} predictions to {output_path}")
    print("Prediction label counts:")
    print(output["Prediction"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
