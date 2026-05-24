# BAREC Task 1 Strict Track Baseline

Project này được tổ chức để chạy đơn giản nhất:

- Dataset BAREC 2026 sentence-level được `train.py` tự tải từ `CAMeL-Lab/BAREC-Shared-Task-2026-sent` khi cần.
- Dataset gốc trên Hugging Face không có `D3Tok`, nên `train.py` tự gọi CAMEL Tools để tạo cột này rồi lưu vào `data/barec-corpus-v1`.
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

1. Kiểm tra dataset local trong `data/barec-corpus-v1`.
2. Nếu dataset local chưa đúng bản 2026 hoặc thiếu `D3Tok`, tự tải dataset 2026 từ Hugging Face.
3. Tự gọi CAMEL Tools để tạo `D3Tok` cho từng câu và lưu lại CSV trong project.
4. Fine-tune `aubmindlab/bert-base-arabertv2` bằng cột `D3Tok`.
5. In training loss trong lúc chạy.
6. In score trên dev: Accuracy, Accuracy +/-1, Average absolute distance, QWK, Accuracy 7/5/3 levels.
7. Dự đoán test.
8. Tạo file nộp tại `submission/prediction.zip`.

Ví dụ chạy thử ít epoch:

```powershell
python train.py --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

Nếu chỉ muốn tải dataset 2026 và tạo D3Tok, chưa train:

```powershell
python train.py --prepare-data-only
```

## Eval Script

`eval.py` là script đánh giá do ban tổ chức cung cấp, đã thêm chú thích tiếng Việt.

Ví dụ:

```powershell
python eval.py --output outputs/dev_predictions.csv --split Dev --task Sent
```

Pipeline trong `train.py` đã tự in các metric tương tự trên dev local, nên `eval.py` chủ yếu để đối chiếu với script chính thức.

## D3Tok

`train.py` dùng cột `D3Tok` làm input. Dataset 2026 gốc không có cột này, nên script tự tạo `D3Tok` bằng:

- `camel_tools.tokenizers.word.simple_word_tokenize`
- `camel_tools.disambig.mle.MLEDisambiguator.pretrained("calima-msa-r13")`
- `camel_tools.tokenizers.morphological.MorphologicalTokenizer(..., "d3tok", split=True)`

Nếu máy chưa có data package của CAMEL Tools, chạy:

```powershell
python -m camel_tools.cli.camel_data --install disambig-mle-calima-msa-r13
```

## Strict Track

Không thêm SAMER, từ điển ngoài, dữ liệu sinh bởi LLM, synthetic labels, hay bất kỳ training data ngoài BAREC Corpus. Project chỉ dùng BAREC Corpus trong `data/barec-corpus-v1` và pretrained AraBERTv2.
