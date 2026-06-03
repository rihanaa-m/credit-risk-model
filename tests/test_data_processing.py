"""Unit tests for data processing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_processing import (
    TransactionAggregator,
    TemporalFeatureExtractor,
    CustomerRiskFeatures,
    build_feature_engineering_pipeline,
)


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        "CustomerId": ["C001", "C001", "C002", "C002", "C002"],
        "TransactionStartTime": pd.to_datetime([
            "2019-02-01 10:00:00",
            "2019-02-05 14:30:00",
            "2019-02-03 09:15:00",
            "2019-02-07 11:45:00",
            "2019-02-10 16:20:00",
        ]),
        "Amount": [1000, 2000, 500, 1500, 3000],
        "Value": [1000, 2000, 500, 1500, 3000],
        "FraudResult": [0, 0, 0, 1, 0],
        "ProductCategory": ["airtime", "financial_services", "airtime", "utility_bill", "airtime"],
        "ChannelId": ["web", "web", "mobile", "web", "mobile"],
    })


class TestTransactionAggregator:
    """Test TransactionAggregator transformer."""

    def test_aggregator_output_shape(self, sample_data):
        """Test that aggregator reduces data to customer level."""
        agg = TransactionAggregator()
        agg.fit(sample_data)
        result = agg.transform(sample_data)

        # Should have one row per customer
        assert len(result) == 2
        assert "CustomerId" in result.columns
        assert "agg_Amount_sum" in result.columns

    def test_aggregator_calculations(self, sample_data):
        """Test that aggregation calculations are correct."""
        agg = TransactionAggregator()
        agg.fit(sample_data)
        result = agg.transform(sample_data)

        # Check C001: 2 transactions, sum=3000, mean=1500
        c001 = result[result["CustomerId"] == "C001"].iloc[0]
        assert c001["agg_Amount_sum"] == 3000
        assert c001["agg_Amount_mean"] == 1500

    def test_aggregator_handles_missing_values(self):
        """Test aggregator with missing values."""
        data = pd.DataFrame({
            "CustomerId": ["C001", "C001"],
            "Amount": [1000, np.nan],
            "Value": [1000, 2000],
        })

        agg = TransactionAggregator()
        agg.fit(data)
        result = agg.transform(data)

        assert len(result) == 1
        assert not result["agg_Amount_sum"].isna().any()


class TestTemporalFeatureExtractor:
    """Test TemporalFeatureExtractor transformer."""

    def test_temporal_features_extracted(self, sample_data):
        """Test that temporal features are extracted correctly."""
        extractor = TemporalFeatureExtractor()
        extractor.fit(sample_data)
        result = extractor.transform(sample_data)

        # Check that new columns exist
        assert "txn_hour" in result.columns
        assert "txn_day" in result.columns
        assert "txn_month" in result.columns
        assert "txn_year" in result.columns

    def test_temporal_features_values(self, sample_data):
        """Test temporal feature values are correct."""
        extractor = TemporalFeatureExtractor()
        extractor.fit(sample_data)
        result = extractor.transform(sample_data)

        # First row: 2019-02-01 10:00:00
        assert result.iloc[0]["txn_hour"] == 10
        assert result.iloc[0]["txn_day"] == 1
        assert result.iloc[0]["txn_month"] == 2
        assert result.iloc[0]["txn_year"] == 2019


class TestCustomerRiskFeatures:
    """Test CustomerRiskFeatures transformer."""

    def test_fraud_rate_calculation(self, sample_data):
        """Test fraud rate feature calculation."""
        risk_features = CustomerRiskFeatures()
        risk_features.fit(sample_data)
        result = risk_features.transform(sample_data)

        assert "fraud_rate" in result.columns
        assert "has_negative_amount" in result.columns

    def test_negative_amount_flag(self):
        """Test negative amount flag detection."""
        data = pd.DataFrame({
            "CustomerId": ["C001", "C001"],
            "Amount": [1000, -500],
            "FraudResult": [0, 0],
            "ProductCategory": ["airtime", "airtime"],
            "ChannelId": ["web", "web"],
        })

        risk_features = CustomerRiskFeatures()
        risk_features.fit(data)
        result = risk_features.transform(data)

        assert result.iloc[0]["has_negative_amount"] == 0
        assert result.iloc[1]["has_negative_amount"] == 1


class TestPipeline:
    """Test complete feature engineering pipeline."""

    def test_pipeline_output_structure(self, sample_data):
        """Test that pipeline produces expected output structure."""
        pipeline = build_feature_engineering_pipeline()

        # Pipeline should be a sklearn Pipeline
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_pipeline_has_required_steps(self, sample_data):
        """Test that pipeline includes all required steps."""
        pipeline = build_feature_engineering_pipeline()

        step_names = [name for name, _ in pipeline.steps]
        assert "temporal" in step_names or "risk_features" in step_names
        assert "preprocessor" in step_names


class TestDataQuality:
    """Test data quality checks."""

    def test_no_missing_critical_columns(self, sample_data):
        """Test that sample data has required columns."""
        required_cols = ["CustomerId", "Amount", "Value", "TransactionStartTime"]
        for col in required_cols:
            assert col in sample_data.columns

    def test_transaction_timestamp_valid(self, sample_data):
        """Test that timestamps are valid datetime objects."""
        assert pd.api.types.is_datetime64_any_dtype(sample_data["TransactionStartTime"])
