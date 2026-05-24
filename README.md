# BAREC Task 1 Strict Track Baseline

This repository implements the BAREC Shared Task sentence-level strict-track baseline:

- pretrained model: `aubmindlab/bert-base-arabertv2`
- input variant: `D3Tok`
- loss: standard multi-class cross-entropy
- labels: 19 readability levels, exported as integers `1..19`

Strict-track note: do not add external readability corpora, lexicons, synthetic labels, LLM-generated data, or any other outside training data. The scripts use only the BAREC Corpus and the pretrained AraBERTv2 checkpoint.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download Data

```powershell
python scripts/download_data.py --output-dir data/barec-corpus-v1
```

The script downloads `CAMeL-Lab/BAREC-Corpus-v1.0` from Hugging Face and writes `train.csv`, `dev.csv`, and `test.csv`. The dataset already includes the `D3Tok` input column required for Baseline II.

## Train

Do not run training until the project owner confirms.

```powershell
python scripts/train.py `
  --data-dir data/barec-corpus-v1 `
  --model-name aubmindlab/bert-base-arabertv2 `
  --input-column D3Tok `
  --label-column Readability_Level_19 `
  --output-dir models/arabertv2-d3tok-ce
```

During training, the script prints training loss at logging steps and evaluation scores after each evaluation pass: accuracy, macro-F1, and quadratic weighted kappa.

## Predict

```powershell
python scripts/predict.py `
  --model-dir models/arabertv2-d3tok-ce `
  --input-file data/barec-corpus-v1/test.csv `
  --output-file outputs/test_predictions.csv
```

## Make Codabench Submission

```powershell
python scripts/make_submission.py `
  --predictions outputs/test_predictions.csv `
  --output-dir submission
```

This creates:

- `submission/prediction`
- `submission/prediction.zip`

The `prediction` file uses the Codabench format selected for this project:

```csv
Sentence ID,Prediction
10100290001,7
```
