"""
FastAPI REST Service for EV Charging Demand Prediction.

Provides RESTful endpoints for real-time inference, system health monitoring,
and project architectural documentation.

Author: Academic Machine Learning Mentorship & Research Team
Project: EV Charging Demand Prediction Using Ensemble Machine Learning
"""

import sys
import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

# Ensure root and src directory are on sys.path for joblib unpickling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import schemas and preprocessing transformer
from api.schemas import (
    ChargingSessionInput,
    PredictionResponse,
    HealthResponse,
    ProjectInfoResponse,
)
import __main__
from preprocessing import FeatureEngineer  # Required for joblib deserialization
setattr(__main__, 'FeatureEngineer', FeatureEngineer)

# Global model container and path
MODEL_PIPELINE = None
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ev_charging_demand_pipeline.joblib")


def load_model():
    """Load model pipeline from disk."""
    global MODEL_PIPELINE
    if os.path.exists(MODEL_PATH):
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            print(f"[LOAD] Successfully loaded ML pipeline from: {MODEL_PATH}")
        except Exception as e:
            print(f"[LOAD ERROR] Failed to load model pipeline: {str(e)}")
            MODEL_PIPELINE = None
    else:
        print(f"[LOAD WARNING] Model pipeline file not found at: {MODEL_PATH}")
        MODEL_PIPELINE = None


# Trigger load on module initialization
load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ensure the serialized Scikit-learn Pipeline is loaded on application startup.
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        load_model()
    yield


app = FastAPI(
    title="EV Charging Demand Prediction API",
    description=(
        "Production-ready REST API for predicting EV charging session energy consumption (in kWh) "
        "using an ensemble machine learning architecture (Stacking Regressor). "
        "Developed for charging station capacity planning and load forecasting."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


@app.get(
    "/",
    response_model=ProjectInfoResponse,
    tags=["Project Information"],
    summary="Get project overview and endpoint directory"
)
def get_project_info() -> ProjectInfoResponse:
    """
    Returns high-level project metadata, architectural summary, and endpoint routing.
    """
    return ProjectInfoResponse(
        title="EV Charging Demand Prediction Using Ensemble Machine Learning",
        description=(
            "An academic machine learning project evaluating Bagging, Boosting, Voting, "
            "and Stacking for predicting electric vehicle charging energy consumption (kWh)."
        ),
        authors="Academic Machine Learning Mentorship & Research Team",
        version="1.0.0",
        architecture="Stacking Ensemble (Base Learners: Linear Regression, Decision Tree, KNN; Meta-Learner: Ridge)",
        target_variable="Energy Consumed (kWh)",
        best_model_name="Stacking Regressor",
        best_model_metrics={
            "MAE_kwh": 18.406,
            "MSE": 454.897,
            "RMSE_kwh": 21.328,
            "R2": -0.0047,
            "CV_RMSE_Mean_kwh": 22.661,
            "CV_RMSE_Std": 1.138
        },
        endpoints={
            "GET /": "Project metadata and documentation",
            "GET /health": "Health and model status check",
            "POST /predict": "Predict energy demand (kWh) for a charging session",
            "GET /docs": "Interactive Swagger UI documentation",
            "GET /redoc": "ReDoc documentation"
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Check API service and model pipeline health"
)
def check_health() -> HealthResponse:
    """
    Returns system status and verifies that the pre-trained ML pipeline is loaded in memory.
    """
    is_loaded = MODEL_PIPELINE is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict expected energy demand (kWh) for an EV charging session"
)
def predict_energy_demand(payload: ChargingSessionInput) -> PredictionResponse:
    """
    Accepts pre-session arrival features of an EV charging event, passes them
    through the feature engineering and preprocessing ColumnTransformer pipeline,
    and returns the predicted energy demand in kWh.
    """
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model pipeline is not loaded on the server."
        )

    try:
        # Convert Pydantic model into a single-row DataFrame
        input_data = pd.DataFrame([payload.model_dump()])

        # Run inference through end-to-end Scikit-learn Pipeline
        raw_prediction = float(MODEL_PIPELINE.predict(input_data)[0])

        # Physical sanity check: Energy consumed cannot be negative
        sanitized_prediction = max(0.0, raw_prediction)

        return PredictionResponse(
            predicted_energy_kwh=round(sanitized_prediction, 2),
            unit="kWh",
            model_name="Stacking Regressor (LinearRegression + DecisionTree + KNN -> Ridge)",
            status="success"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
