"""
API Blueprint for GradeGuru.

This module defines the REST API endpoints responsible for health checks
and student performance prediction. It loads pre-trained machine learning
models and exposes them via Flask Blueprint routes.

Endpoints:
- GET /api/health: Service health check.
- POST /api/predict: Predict final grade (G3) and academic risk band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

Mode = Literal["full", "early"]

#: Flask Blueprint for all API routes
api = Blueprint("api", __name__, url_prefix="/api")

#: Project root directory (one level above this file)
ROOT = Path(__file__).resolve().parents[1]

#: Directory containing trained ML models
MODELS_DIR = ROOT / "models"

#: Load trained models at application startup
full_model = joblib.load(MODELS_DIR / "full.joblib")
early_model = joblib.load(MODELS_DIR / "early.joblib")


def risk_band(predicted_g3: float) -> str:
    """
    Categorize a predicted final grade into a risk band.

    Args:
        predicted_g3 (float): Predicted final grade (G3).

    Returns:
        str: Risk category ("High", "Medium", or "Low").
    """
    if predicted_g3 < 10:
        return "High"
    if predicted_g3 < 14:
        return "Medium"
    return "Low"


@api.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        flask.Response: JSON response indicating API availability.
    """
    return jsonify({"status": "ok"})


@api.post("/predict")
def predict():
    """
    Predict student academic performance using a trained ML model.

    This endpoint accepts JSON input where keys correspond to feature
    names from the UCI Student Performance dataset. An optional "mode"
    field selects the prediction model:
        - "early": Early-warning model (default)
        - "full": Full model including prior period grades

    Expected JSON example:
        {
            "mode": "early",
            "studytime": 2,
            "failures": 0,
            "absences": 4
        }

    Returns:
        flask.Response: JSON response containing the predicted final grade
        and the associated academic risk band.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}

    mode = data.pop("mode", "early")
    if mode not in ("full", "early"):
        return jsonify(
            {"error": {"message": "mode must be 'full' or 'early'"}}
        ), 400

    model = full_model if mode == "full" else early_model

    try:
        X = pd.DataFrame([data])
        pred = float(model.predict(X)[0])
        return jsonify(
            {
                "status": "success",
                "mode": mode,
                "predicted_g3": round(pred, 2),
                "risk_band": risk_band(pred),
            }
        )
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 400
