"""Pydantic models for FastAPI request/response validation."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureInput(BaseModel):
    """Input features for risk prediction."""

    agg_amount_sum: float = Field(..., description="Sum of transaction amounts")
    agg_amount_mean: float = Field(..., description="Mean transaction amount")
    agg_amount_count: float = Field(..., description="Number of transactions")
    agg_value_sum: float = Field(..., description="Sum of transaction values")
    agg_value_mean: float = Field(..., description="Mean transaction value")
    agg_value_std: float = Field(..., description="Std dev of transaction values")
    fraud_rate: float = Field(..., description="Historical fraud rate")
    txn_hour: int = Field(..., description="Hour of transaction")
    txn_month: int = Field(..., description="Month of transaction")
    product_category: str = Field(..., description="Product category")
    channel_id: str = Field(..., description="Transaction channel")

    class Config:
        schema_extra = {
            "example": {
                "agg_amount_sum": 50000,
                "agg_amount_mean": 5000,
                "agg_amount_count": 10,
                "agg_value_sum": 50000,
                "agg_value_mean": 5000,
                "agg_value_std": 2000,
                "fraud_rate": 0.0,
                "txn_hour": 14,
                "txn_month": 6,
                "product_category": "airtime",
                "channel_id": "web",
            }
        }


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""

    customer_id: Optional[str] = Field(None, description="Customer identifier")
    risk_probability: float = Field(..., description="Predicted risk probability (0-1)")
    risk_segment: str = Field(..., description="Risk segment: low, medium, high")
    decision: str = Field(..., description="Approval decision: approve or decline")
    credit_limit_usd: float = Field(..., description="Recommended credit limit in USD")
    explanation: Dict = Field(..., description="Feature contributions and model details")
    model_version: str = Field(..., description="Model version identifier")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is available")
    model_version: Optional[str] = Field(None, description="Loaded model version")
