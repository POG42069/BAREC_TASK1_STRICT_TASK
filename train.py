from __future__ import annotations

"""File chạy chính của project.

Người dùng chỉ cần chạy:

    python train.py

File này chuyển quyền thực thi sang `scripts/train.py`, nơi chứa toàn bộ pipeline:
train model, in loss/score, dự đoán test, và tạo `submission/prediction.zip`.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    # Dùng runpy để giữ toàn bộ tham số dòng lệnh, ví dụ: python train.py --epochs 1
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "train.py"), run_name="__main__")
