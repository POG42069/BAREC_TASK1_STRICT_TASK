from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ID_ALIASES = ("Sentence ID", "ID", "sentence_id", "id")
PREDICTION_ALIASES = ("Prediction", "prediction", "label", "predicted_label")


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Tìm tên cột đầu tiên tồn tại trong DataFrame."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa prediction về đúng 2 cột Codabench cần: `Sentence ID`, `Prediction`."""
    id_column = _first_existing(predictions.columns, ID_ALIASES)
    prediction_column = _first_existing(predictions.columns, PREDICTION_ALIASES)

    if id_column is None:
        raise ValueError(f"Missing sentence id column. Expected one of: {', '.join(ID_ALIASES)}")
    if prediction_column is None:
        raise ValueError(f"Missing prediction column. Expected one of: {', '.join(PREDICTION_ALIASES)}")

    normalized = predictions[[id_column, prediction_column]].rename(
        columns={id_column: "Sentence ID", prediction_column: "Prediction"}
    )

    if normalized["Sentence ID"].isna().any():
        raise ValueError("Sentence ID contains missing values")
    if normalized["Prediction"].isna().any():
        raise ValueError("Prediction contains missing values")

    # Codabench yêu cầu nhãn dự đoán là số nguyên trong khoảng 1..19.
    try:
        normalized["Prediction"] = normalized["Prediction"].astype(int)
    except ValueError as exc:
        raise ValueError("Prediction must contain integer labels") from exc

    invalid = normalized.loc[~normalized["Prediction"].between(1, 19), "Prediction"].unique()
    if len(invalid) > 0:
        invalid_values = ", ".join(str(value) for value in sorted(invalid))
        raise ValueError(f"Prediction labels must be in 1..19. Invalid values: {invalid_values}")

    return normalized


def build_submission(predictions_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Tạo file `prediction` và nén thành `prediction.zip` để nộp Codabench."""
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    normalized = normalize_predictions(predictions)

    prediction_file = output_dir / "prediction"
    zip_file = output_dir / "prediction.zip"

    normalized.to_csv(prediction_file, index=False, encoding="utf-8")
    with ZipFile(zip_file, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(prediction_file, arcname="prediction")

    return prediction_file, zip_file
