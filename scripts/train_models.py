from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "student-mat.csv"

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FULL_MODEL_PATH = MODELS_DIR / "full.joblib"
EARLY_MODEL_PATH = MODELS_DIR / "early.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


@dataclass(frozen=True)
class ModelMeta:
    trained_at_utc: str
    dataset: str
    target: str
    mode: str
    features: list[str]
    mae: float
    r2: float


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_one(df: pd.DataFrame, features: list[str], mode: str, out_path: Path) -> ModelMeta:
    X = df[features]
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    joblib.dump(pipe, out_path)

    return ModelMeta(
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset=RAW_PATH.relative_to(ROOT).as_posix(),
        target="G3",
        mode=mode,
        features=features,
        mae=mae,
        r2=r2,
    )


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_PATH}\n"
            "Place student-mat.csv in data/raw/."
        )

    df = pd.read_csv(RAW_PATH, sep=";")
    EARLY_FEATURES = [
    "sex", "age", "address",
    "Medu", "Fedu",
    "studytime", "failures", "absences",
    "schoolsup", "famsup",
    "higher", "internet",
    "Dalc", "Walc", "health"
    ]
    FULL_FEATURES = EARLY_FEATURES + ["G1", "G2"]

    meta_full = train_one(df, FULL_FEATURES, "full", FULL_MODEL_PATH)
    meta_early = train_one(df, EARLY_FEATURES, "early", EARLY_MODEL_PATH)

    payload = {"full": asdict(meta_full), "early": asdict(meta_early)}
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved:", FULL_MODEL_PATH)
    print("Saved:", EARLY_MODEL_PATH)
    print("Saved:", METADATA_PATH)
    print("Full MAE:", meta_full.mae, "R2:", meta_full.r2)
    print("Early MAE:", meta_early.mae, "R2:", meta_early.r2)


if __name__ == "__main__":
    main()
