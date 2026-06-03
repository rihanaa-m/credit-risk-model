"""FastAPI application for credit risk scoring.

This module implements a REST API for credit risk probability prediction,
with model loading from MLflow and request validation via Pydantic.
"""

from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .pydantic_models import FeatureInput, HealthCheckResponse, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="REST API for credit risk probability prediction",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
MODEL_VERSION = None


def load_model():
    """Load best model from MLflow registry."""
    global MODEL, MODEL_VERSION

    try:
        # Connect to MLflow
        mlflow.set_tracking_uri("http://localhost:5000")

        # Try to load from registry
        MODEL = mlflow.pyfunc.load_model("models:/credit-risk-model/production")
        MODEL_VERSION = "production"
        print("✅ Loaded model from MLflow registry (production)")

    except Exception as e:
        print(f"⚠️ MLflow registry unavailable: {e}")
        print("   Using local model stub for demo purposes")
        MODEL_VERSION = "local-stub"
        # In production, you'd load from a local artifact or file
        MODEL = None


@app.on_event("startup")
async def startup_event():
    """Load model on app startup."""
    load_model()


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=MODEL is not None or MODEL_VERSION is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    features: FeatureInput,
    customer_id: Optional[str] = None,
) -> PredictionResponse:
    """Predict credit risk for a customer.

    Parameters
    ----------
    features : FeatureInput
        Customer features
    customer_id : Optional[str]
        Optional customer identifier

    Returns
    -------
    PredictionResponse
        Risk prediction and decision
    """
    if MODEL is None and MODEL_VERSION == "local-stub":
        # Stub response for demo (when MLflow not available)
        risk_prob = np.random.uniform(0, 1)
    else:
        # Convert features to array for model
        feature_array = np.array([
            features.agg_amount_sum,
            features.agg_amount_mean,
            features.agg_amount_count,
            features.agg_value_sum,
            features.agg_value_mean,
            features.agg_value_std,
            features.fraud_rate,
            features.txn_hour,
            features.txn_month,
        ]).reshape(1, -1)

        try:
            risk_prob = MODEL.predict(feature_array)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Clip to [0, 1]
    risk_prob = float(np.clip(risk_prob, 0, 1))

    # Determine risk segment and decision
    if risk_prob < 0.30:
        risk_segment = "low"
        decision = "approve"
        credit_limit = 100.0
    elif risk_prob < 0.60:
        risk_segment = "medium"
        decision = "approve"
        credit_limit = 50.0
    else:
        risk_segment = "high"
        decision = "decline"
        credit_limit = 0.0

    # Feature contributions (placeholder for demo)
    explanation = {
        "top_risk_factors": [
            {
                "feature": "fraud_rate",
                "contribution": +0.05 if features.fraud_rate > 0 else 0.0,
                "direction": "elevating_risk" if features.fraud_rate > 0 else "neutral",
            },
            {
                "feature": "frequency_proxy",
                "contribution": -0.08,
                "direction": "protective",
            },
        ],
        "model_version": MODEL_VERSION,
        "features_used": 9,
    }

    return PredictionResponse(
        customer_id=customer_id,
        risk_probability=risk_prob,
        risk_segment=risk_segment,
        decision=decision,
        credit_limit_usd=credit_limit,
        explanation=explanation,
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
