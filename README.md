# BAREC Task 1 Strict Track Baseline

Project này triển khai baseline cho BAREC Shared Task sentence-level strict track.

- Model pretrained: `aubmindlab/bert-base-arabertv2`
- Input: cột `D3Tok` có sẵn trong BAREC Corpus
- Loss: Cross-entropy cho phân loại 19 mức
- Output nộp Codabench: `submission/prediction.zip`

## Cấu trúc quan trọng

```text
data/barec-corpus-v1/
  train.csv
  dev.csv
  test.csv
train.py
eval.py
scripts/
src/
```

Thư mục `data/barec-corpus-v1` đã nằm sẵn trong project. Không cần tải lại dataset trước khi train.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy Training Và Tạo File Nộp

Chỉ cần chạy:

```powershell
python train.py
```

Lệnh này sẽ tự làm toàn bộ pipeline:

1. Đọc `data/barec-corpus-v1/train.csv`, `dev.csv`, `test.csv`.
2. Fine-tune AraBERTv2 bằng cột `D3Tok`.
3. In training loss trong lúc chạy.
4. In score trên dev: Accuracy, Accuracy +/-1, Average absolute distance, QWK, Accuracy 7/5/3 levels.
5. Dự đoán test.
6. Tạo `submission/prediction` và `submission/prediction.zip`.

File zip cuối cùng để nộp nằm ở:

```text
submission/prediction.zip
```

## Tùy Chỉnh Nhanh

Ví dụ train ít epoch hơn để thử pipeline:

```powershell
python train.py --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

Các tham số mặc định chính:

- `--data-dir data/barec-corpus-v1`
- `--model-name aubmindlab/bert-base-arabertv2`
- `--input-column D3Tok`
- `--label-column Readability_Level_19`
- `--output-dir models/arabertv2-d3tok-ce`
- `--submission-dir submission`

## Eval Script Của Ban Tổ Chức

File `eval.py` được thêm ở root project theo script ban tổ chức cung cấp, có chú thích tiếng Việt.

Ví dụ:

```powershell
python eval.py --output outputs/dev_predictions.csv --split Dev --task Sent
```

Lưu ý: script này tải ground truth từ Hugging Face theo dataset shared-task 2025 như bản gốc. Trong pipeline `python train.py`, project cũng tự in các metric tương tự bằng dev local trong `data/barec-corpus-v1`.

## Strict Track

Không thêm SAMER, từ điển ngoài, dữ liệu sinh bởi LLM, synthetic labels, hay bất kỳ training data ngoài BAREC Corpus. Project chỉ dùng BAREC Corpus trong `data/barec-corpus-v1` và pretrained AraBERTv2.
