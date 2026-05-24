from __future__ import annotations

import zipfile

import pandas as pd

from barec.submission import build_submission, normalize_predictions


def test_normalize_predictions_accepts_id_aliases() -> None:
    frame = pd.DataFrame({"ID": [1, 2], "Prediction": [1, 19]})

    normalized = normalize_predictions(frame)

    assert list(normalized.columns) == ["Sentence ID", "Prediction"]
    assert normalized.to_dict(orient="records") == [
        {"Sentence ID": 1, "Prediction": 1},
        {"Sentence ID": 2, "Prediction": 19},
    ]


def test_build_submission_writes_codabench_files(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame({"Sentence ID": [10, 20], "Prediction": [7, 12]}).to_csv(predictions_path, index=False)

    prediction_file, zip_file = build_submission(predictions_path, tmp_path / "submission")

    assert prediction_file.name == "prediction"
    assert zip_file.name == "prediction.zip"
    assert prediction_file.read_text(encoding="utf-8").splitlines() == [
        "Sentence ID,Prediction",
        "10,7",
        "20,12",
    ]
    with zipfile.ZipFile(zip_file) as archive:
        assert archive.namelist() == ["prediction"]
        assert archive.read("prediction").decode("utf-8").splitlines()[0] == "Sentence ID,Prediction"
