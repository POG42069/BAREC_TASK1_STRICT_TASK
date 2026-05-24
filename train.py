from __future__ import annotations

"""Huấn luyện AraBERTv2 và tạo luôn file zip nộp BAREC Task 1.

Bạn chỉ cần chạy:

    python train.py

Pipeline mặc định:
1. Đọc dataset có sẵn trong `data/barec-corpus-v1`.
2. Fine-tune `aubmindlab/bert-base-arabertv2` bằng cột `D3Tok`.
3. In loss trong lúc train và in score trên tập dev.
4. Dự đoán tập test.
5. Tạo `submission/prediction.zip` để nộp Codabench.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BAREC_7_DICT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7}
BAREC_5_DICT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5}
BAREC_3_DICT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3}


def configure_utf8_console() -> None:
    """Ép console dùng UTF-8 để log tiếng Việt không lỗi trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Khai báo tham số dòng lệnh; nếu không truyền gì thì chạy cấu hình mặc định."""
    parser = argparse.ArgumentParser(description="Train AraBERTv2 and create BAREC Codabench submission zip.")
    parser.add_argument("--data-dir", default="data/barec-corpus-v1", help="Thư mục chứa train.csv, dev.csv, test.csv.")
    parser.add_argument("--model-name", default="aubmindlab/bert-base-arabertv2", help="Checkpoint pretrained AraBERTv2.")
    parser.add_argument("--input-column", default="D3Tok", help="Cột văn bản đã được D3Tok trong dataset.")
    parser.add_argument("--label-column", default="Readability_Level_19", help="Cột nhãn 19 mức, giá trị 1..19.")
    parser.add_argument("--output-dir", default="models/arabertv2-d3tok-ce", help="Nơi lưu model sau khi train.")
    parser.add_argument("--outputs-dir", default="outputs", help="Nơi lưu file dự đoán CSV trung gian.")
    parser.add_argument("--submission-dir", default="submission", help="Nơi lưu prediction và prediction.zip.")
    parser.add_argument("--max-length", type=int, default=256, help="Độ dài tối đa sau tokenizer.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Số epoch train.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Batch size train trên mỗi device.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size eval/predict trên mỗi device.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay cho AdamW.")
    parser.add_argument("--logging-steps", type=int, default=50, help="In training loss mỗi N step.")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập kết quả.")
    return parser.parse_args()


def read_labeled_split(data_dir: Path, filename: str, input_column: str, label_column: str) -> pd.DataFrame:
    """Đọc split có nhãn và đổi label từ 1..19 sang 0..18 cho Hugging Face Trainer."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in ("ID", input_column, label_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} thiếu cột bắt buộc: {', '.join(missing)}")

    frame = frame[["ID", input_column, label_column]].copy()
    frame[input_column] = frame[input_column].fillna("").astype(str)
    frame[label_column] = frame[label_column].astype(int)

    invalid = sorted(frame.loc[~frame[label_column].between(1, 19), label_column].unique())
    if invalid:
        raise ValueError(f"{path} có nhãn nằm ngoài 1..19: {invalid}")

    # Trainer dùng nhãn bắt đầu từ 0, còn Codabench yêu cầu xuất nhãn 1..19.
    frame["labels"] = frame[label_column] - 1
    return frame


def read_prediction_split(data_dir: Path, filename: str, input_column: str) -> pd.DataFrame:
    """Đọc split cần dự đoán; chỉ cần ID và cột input D3Tok."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in ("ID", input_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} thiếu cột bắt buộc: {', '.join(missing)}")

    frame = frame[["ID", input_column]].copy()
    frame[input_column] = frame[input_column].fillna("").astype(str)
    return frame


def tokenize_labeled_dataset(frame: pd.DataFrame, tokenizer, input_column: str, max_length: int) -> Dataset:
    """Tokenize dữ liệu có nhãn để train/evaluate."""
    dataset = Dataset.from_pandas(frame[["ID", input_column, "labels"]], preserve_index=False)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        # Không padding ở đây; DataCollatorWithPadding sẽ padding động theo batch để tiết kiệm bộ nhớ.
        return tokenizer(batch[input_column], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=[input_column])


def tokenize_prediction_dataset(frame: pd.DataFrame, tokenizer, input_column: str, max_length: int) -> Dataset:
    """Tokenize dữ liệu không cần nhãn để sinh prediction."""
    dataset = Dataset.from_pandas(frame[["ID", input_column]], preserve_index=False)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[input_column], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=[input_column])


def official_like_metrics(labels_1_to_19: list[int], predictions_1_to_19: list[int]) -> dict[str, float]:
    """Tính các metric giống eval.py của ban tổ chức cho sentence-level task."""
    acc_margin_1 = np.mean([abs(pred - label) <= 1 for pred, label in zip(predictions_1_to_19, labels_1_to_19)])

    labels_7 = [BAREC_7_DICT[label] for label in labels_1_to_19]
    predictions_7 = [BAREC_7_DICT[pred] for pred in predictions_1_to_19]
    labels_5 = [BAREC_5_DICT[label] for label in labels_1_to_19]
    predictions_5 = [BAREC_5_DICT[pred] for pred in predictions_1_to_19]
    labels_3 = [BAREC_3_DICT[label] for label in labels_1_to_19]
    predictions_3 = [BAREC_3_DICT[pred] for pred in predictions_1_to_19]

    return {
        "accuracy": accuracy_score(labels_1_to_19, predictions_1_to_19),
        "accuracy_margin_1": float(acc_margin_1),
        "avg_abs_dist": mean_absolute_error(labels_1_to_19, predictions_1_to_19),
        "qwk": cohen_kappa_score(labels_1_to_19, predictions_1_to_19, weights="quadratic"),
        "accuracy_7": accuracy_score(labels_7, predictions_7),
        "accuracy_5": accuracy_score(labels_5, predictions_5),
        "accuracy_3": accuracy_score(labels_3, predictions_3),
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    """Metric được Trainer gọi sau mỗi lần evaluate; label nội bộ là 0..18."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    labels_1_to_19 = [int(label) + 1 for label in labels]
    predictions_1_to_19 = [int(prediction) + 1 for prediction in predictions]
    metrics = official_like_metrics(labels_1_to_19, predictions_1_to_19)
    metrics["macro_f1"] = f1_score(labels, predictions, average="macro")
    return metrics


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Tạo TrainingArguments tương thích cả tên cũ và mới của transformers."""
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
        "metric_for_best_model": "qwk",
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


def write_predictions(ids: pd.Series, predictions_0_to_18: np.ndarray, output_path: Path) -> pd.DataFrame:
    """Ghi file CSV dự đoán với format `Sentence ID,Prediction` và nhãn 1..19."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_1_to_19 = [int(label) + 1 for label in predictions_0_to_18]
    output = pd.DataFrame({"Sentence ID": ids, "Prediction": predictions_1_to_19})
    output.to_csv(output_path, index=False, encoding="utf-8")
    return output


def build_submission(predictions: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Tạo file `prediction` và nén thành `prediction.zip` để nộp Codabench."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = {"Sentence ID", "Prediction"}
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"Thiếu cột submission bắt buộc: {', '.join(missing)}")

    normalized = predictions[["Sentence ID", "Prediction"]].copy()
    normalized["Prediction"] = normalized["Prediction"].astype(int)
    invalid = normalized.loc[~normalized["Prediction"].between(1, 19), "Prediction"].unique()
    if len(invalid) > 0:
        raise ValueError(f"Prediction phải nằm trong 1..19, nhưng có giá trị lỗi: {sorted(invalid)}")

    prediction_file = output_dir / "prediction"
    zip_file = output_dir / "prediction.zip"
    normalized.to_csv(prediction_file, index=False, encoding="utf-8")
    with ZipFile(zip_file, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(prediction_file, arcname="prediction")
    return prediction_file, zip_file


def print_official_metrics(metrics: dict[str, float]) -> None:
    """In metric theo format dễ đọc, gần giống eval.py của ban tổ chức."""
    print(f"Accuracy: {metrics['accuracy'] * 100:.4f}%")
    print(f"Accuracy +/-1: {metrics['accuracy_margin_1'] * 100:.4f}%")
    print(f"Average absolute distance: {metrics['avg_abs_dist']:.6f}")
    print(f"Quadratic Cohen's Kappa: {metrics['qwk'] * 100:.4f}%")
    print(f"Accuracy (7 levels): {metrics['accuracy_7'] * 100:.4f}%")
    print(f"Accuracy (5 levels): {metrics['accuracy_5'] * 100:.4f}%")
    print(f"Accuracy (3 levels): {metrics['accuracy_3'] * 100:.4f}%")


def main() -> None:
    """Chạy toàn bộ quy trình từ train đến tạo file zip nộp bài."""
    configure_utf8_console()
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    outputs_dir = Path(args.outputs_dir)
    submission_dir = Path(args.submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("Bắt đầu đọc dữ liệu BAREC có sẵn trong project.")
    train_frame = read_labeled_split(data_dir, "train.csv", args.input_column, args.label_column)
    dev_frame = read_labeled_split(data_dir, "dev.csv", args.input_column, args.label_column)
    test_frame = read_prediction_split(data_dir, "test.csv", args.input_column)
    print(f"Train rows: {len(train_frame):,}")
    print(f"Dev rows: {len(dev_frame):,}")
    print(f"Test rows: {len(test_frame):,}")

    print(f"Tải tokenizer và model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=19,
        id2label={index: str(index + 1) for index in range(19)},
        label2id={str(index + 1): index for index in range(19)},
    )

    train_dataset = tokenize_labeled_dataset(train_frame, tokenizer, args.input_column, args.max_length)
    dev_dataset = tokenize_labeled_dataset(dev_frame, tokenizer, args.input_column, args.max_length)
    test_dataset = tokenize_prediction_dataset(test_frame, tokenizer, args.input_column, args.max_length)

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu training. Loss sẽ được in theo logging_steps; score dev sẽ được in sau mỗi epoch.")
    train_result = trainer.train()
    print(f"Final training loss: {train_result.training_loss:.6f}")

    print("Evaluate lại trên dev bằng best checkpoint.")
    final_metrics = trainer.evaluate(dev_dataset)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))

    print("Lưu model đã train.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model_name": args.model_name,
        "input_column": args.input_column,
        "label_column": args.label_column,
        "num_labels": 19,
        "loss": "cross_entropy",
        "submission_zip": str(submission_dir / "prediction.zip"),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Sinh dự đoán trên dev để in score theo eval.py.")
    dev_prediction_output = trainer.predict(dev_dataset)
    dev_predictions_0_to_18 = np.argmax(dev_prediction_output.predictions, axis=-1)
    dev_predictions = write_predictions(dev_frame["ID"], dev_predictions_0_to_18, outputs_dir / "dev_predictions.csv")
    dev_metrics = official_like_metrics(
        labels_1_to_19=dev_frame[args.label_column].astype(int).tolist(),
        predictions_1_to_19=dev_predictions["Prediction"].astype(int).tolist(),
    )
    print_official_metrics(dev_metrics)

    print("Sinh dự đoán trên test và tạo file nộp Codabench.")
    test_prediction_output = trainer.predict(test_dataset)
    test_predictions_0_to_18 = np.argmax(test_prediction_output.predictions, axis=-1)
    test_predictions = write_predictions(test_frame["ID"], test_predictions_0_to_18, outputs_dir / "test_predictions.csv")
    print("Phân bố nhãn test dự đoán:")
    print(test_predictions["Prediction"].value_counts().sort_index().to_string())

    prediction_file, zip_file = build_submission(test_predictions, submission_dir)
    print(f"Đã tạo file prediction: {prediction_file}")
    print(f"Đã tạo file zip để nộp: {zip_file}")


if __name__ == "__main__":
    main()
