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

EARLY_FEATURES = [
    "sex", "age", "address",
    "Medu", "Fedu",
    "studytime", "failures", "absences",
    "schoolsup", "famsup",
    "higher", "internet",
    "Dalc", "Walc", "health"
]
FULL_FEATURES = EARLY_FEATURES + ["G1", "G2"]

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

@api.get("/schema")
def schema():
    """
    Return the required input feature schema for each prediction mode.

    This endpoint helps frontend clients understand which input fields
    are required for 'early' and 'full' prediction modes.
    """
    return jsonify({
        "early": EARLY_FEATURES,
        "full": FULL_FEATURES
    })

@api.post("/predict")
def predict():
    """
    Predict student academic performance using a trained ML model.

    This endpoint accepts JSON input where keys correspond to the model's
    expected feature names. An optional "mode" selects the prediction model:
      
    "early": does NOT require G1/G2 (default)
    "full": requires G1 and G2

    Returns:
        flask.Response: JSON response containing predicted final grade (G3)
        and the associated academic risk band.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}

    mode = data.pop("mode", "early")
    if mode not in ("full", "early"):
        return jsonify({"error": {"message": "mode must be 'full' or 'early'"}}), 400

    model = full_model if mode == "full" else early_model
    expected_features = FULL_FEATURES if mode == "full" else EARLY_FEATURES

    try:
        missing_features = [f for f in expected_features if f not in data]
        if missing_features:
            return jsonify({
                "error": {
                    "message": "Missing required features",
                    "missing": missing_features,
                    "mode": mode
                }
            }), 400

        X = pd.DataFrame([{f: data[f] for f in expected_features}])
        pred = float(model.predict(X)[0])

        return jsonify({
            "status": "success",
            "mode": mode,
            "predicted_g3": round(pred, 2),
            "risk_band": risk_band(pred),
        })

    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 400