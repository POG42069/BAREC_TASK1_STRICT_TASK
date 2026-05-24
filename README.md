# BAREC Task 1 Strict Track Baseline

Project này được tổ chức để chạy đơn giản nhất:

- Dataset nằm sẵn trong thư mục `data/barec-corpus-v1`.
- Chỉ có một file pipeline chính: `train.py`.
- Chạy `python train.py` sẽ train model và tạo file zip để nộp.

## Cấu trúc project

```text
data/barec-corpus-v1/
  train.csv
  dev.csv
  test.csv
train.py
eval.py
requirements.txt
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train Và Tạo Submission

Chạy:

```powershell
python train.py
```

File này sẽ tự làm toàn bộ pipeline:

1. Đọc `data/barec-corpus-v1/train.csv`, `dev.csv`, `test.csv`.
2. Fine-tune `aubmindlab/bert-base-arabertv2` bằng cột `D3Tok`.
3. In training loss trong lúc chạy.
4. In score trên dev: Accuracy, Accuracy +/-1, Average absolute distance, QWK, Accuracy 7/5/3 levels.
5. Dự đoán test.
6. Tạo file nộp tại `submission/prediction.zip`.

Ví dụ chạy thử ít epoch:

```powershell
python train.py --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

## Eval Script

`eval.py` là script đánh giá do ban tổ chức cung cấp, đã thêm chú thích tiếng Việt.

Ví dụ:

```powershell
python eval.py --output outputs/dev_predictions.csv --split Dev --task Sent
```

Pipeline trong `train.py` đã tự in các metric tương tự trên dev local, nên `eval.py` chủ yếu để đối chiếu với script chính thức.

## Strict Track

Không thêm SAMER, từ điển ngoài, dữ liệu sinh bởi LLM, synthetic labels, hay bất kỳ training data ngoài BAREC Corpus. Project chỉ dùng BAREC Corpus trong `data/barec-corpus-v1` và pretrained AraBERTv2.
