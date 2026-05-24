from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERTv2 on BAREC Task 1 strict track.")
    parser.add_argument("--data-dir", default="data/barec-corpus-v1", help="Directory containing train.csv/dev.csv.")
    parser.add_argument("--model-name", default="aubmindlab/bert-base-arabertv2", help="Pretrained model checkpoint.")
    parser.add_argument("--input-column", default="D3Tok", help="Input text column.")
    parser.add_argument("--label-column", default="Readability_Level_19", help="Label column with values 1..19.")
    parser.add_argument("--output-dir", default="models/arabertv2-d3tok-ce", help="Model output directory.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device evaluation batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Print training loss every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def read_split(data_dir: Path, filename: str, input_column: str, label_column: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in ("ID", input_column, label_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame[["ID", input_column, label_column]].copy()
    frame[input_column] = frame[input_column].fillna("").astype(str)
    frame[label_column] = frame[label_column].astype(int)

    invalid = sorted(frame.loc[~frame[label_column].between(1, 19), label_column].unique())
    if invalid:
        raise ValueError(f"{path} has labels outside 1..19: {invalid}")

    frame["labels"] = frame[label_column] - 1
    return frame


def tokenize_dataset(frame: pd.DataFrame, tokenizer, input_column: str, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame[["ID", input_column, "labels"]], preserve_index=False)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[input_column], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=[input_column])


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "qwk": cohen_kappa_score(labels + 1, predictions + 1, weights="quadratic"),
    }


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BAREC splits")
    train_frame = read_split(data_dir, "train.csv", args.input_column, args.label_column)
    dev_frame = read_split(data_dir, "dev.csv", args.input_column, args.label_column)
    print(f"Train rows: {len(train_frame):,}")
    print(f"Dev rows: {len(dev_frame):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=19,
        id2label={index: str(index + 1) for index in range(19)},
        label2id={str(index + 1): index for index in range(19)},
    )

    train_dataset = tokenize_dataset(train_frame, tokenizer, args.input_column, args.max_length)
    dev_dataset = tokenize_dataset(dev_frame, tokenizer, args.input_column, args.max_length)
    training_args = make_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training. Loss is logged every logging step; scores are logged on evaluation.")
    train_result = trainer.train()
    print(f"Final training loss: {train_result.training_loss:.6f}")

    metrics = trainer.evaluate()
    print("Final evaluation metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": args.model_name,
        "input_column": args.input_column,
        "label_column": args.label_column,
        "num_labels": 19,
        "loss": "cross_entropy",
        "label_export_offset": 1,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model and metadata to {output_dir}")


if __name__ == "__main__":
    main()
