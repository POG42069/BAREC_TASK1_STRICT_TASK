from __future__ import annotations

"""Script đánh giá do ban tổ chức cung cấp, có bổ sung chú thích tiếng Việt.

Ví dụ dùng cho task sentence-level trên dev:

    python eval.py --output outputs/dev_predictions.csv --split Dev --task Sent

Lưu ý: script này tải ground truth từ Hugging Face theo dataset shared-task 2026.
Pipeline `train.py` cũng tự in các metric tương tự trên dev bằng dữ liệu local
trong `data/barec-corpus-v1`.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error


# Các mapping này gom 19 mức chi tiết về 7/5/3 mức để tính accuracy phụ.
barec_7_dict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7}
barec_5_dict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5}
barec_3_dict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3}


def configure_utf8_console() -> None:
    """Ép console dùng UTF-8 để các chú thích/log tiếng Việt không gây lỗi."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Evaluation script for readability tasks.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--split", type=str, choices=["Dev", "Test"], required=True, help="split: Dev or Test.")
    parser.add_argument("--task", type=str, choices=["Sent", "Doc"], required=True, help="task: Sent (Sentence-level) or Doc (Document-level).")
    args = parser.parse_args()

    output_path = args.output
    split = args.split
    task_name = "Sentence" if args.task == "Sent" else "Document"
    split_name = "validation" if split == "Dev" else "test"

    print(f"Evaluating {task_name}-level readability on {split} split using {output_path}")

    # Đọc file prediction của đội thi. Với task Sent, cột ID phải là `Sentence ID`.
    pred_df = pd.read_csv(output_path, header=0)
    pred_ids = [str(id) for id in list(pred_df[task_name + " ID"])]
    predictions = list(pred_df["Prediction"])
    pred_dict = dict(zip(pred_ids, predictions))

    # Tải ground truth tương ứng từ Hugging Face theo script gốc.
    barec = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2026-" + args.task.lower(), split=split_name)
    gold_ids = [str(id) for id in list(barec["ID"])]
    labels = list(barec["Readability_Level_19"])

    # Kiểm tra prediction có đúng tập ID hay không trước khi tính điểm.
    if set(gold_ids) != set(pred_ids):
        raise ValueError("The output CSV file does not match the split: IDs in predictions and gold data do not align.")

    aligned_predictions = [pred_dict[id] for id in gold_ids]
    aligned_labels = labels

    aligned_predictions = [int(p) for p in aligned_predictions]
    aligned_labels = [int(l) for l in aligned_labels]

    acc = accuracy_score(aligned_labels, aligned_predictions)
    acc_margin_1 = np.mean([abs(p - l) <= 1 for p, l in zip(aligned_predictions, aligned_labels)])
    avg_abs_dist = mean_absolute_error(aligned_labels, aligned_predictions)
    qwk = cohen_kappa_score(aligned_labels, aligned_predictions, weights="quadratic")

    aligned_labels_7 = [barec_7_dict[l] for l in aligned_labels]
    aligned_predictions_7 = [barec_7_dict[p] for p in aligned_predictions]
    acc_7 = accuracy_score(aligned_labels_7, aligned_predictions_7)

    aligned_labels_5 = [barec_5_dict[l] for l in aligned_labels]
    aligned_predictions_5 = [barec_5_dict[p] for p in aligned_predictions]
    acc_5 = accuracy_score(aligned_labels_5, aligned_predictions_5)

    aligned_labels_3 = [barec_3_dict[l] for l in aligned_labels]
    aligned_predictions_3 = [barec_3_dict[p] for p in aligned_predictions]
    acc_3 = accuracy_score(aligned_labels_3, aligned_predictions_3)

    print(f"Accuracy: {acc * 100:.4f}%")
    print(f"Accuracy +/-1: {acc_margin_1 * 100:.4f}%")
    print(f"Average absolute distance: {avg_abs_dist:.6f}")
    print(f"Quadratic Cohen's Kappa: {qwk * 100:.4f}%")
    print(f"Accuracy (7 levels): {acc_7 * 100:.4f}%")
    print(f"Accuracy (5 levels): {acc_5 * 100:.4f}%")
    print(f"Accuracy (3 levels): {acc_3 * 100:.4f}%")
    print("Evaluation completed successfully.")
