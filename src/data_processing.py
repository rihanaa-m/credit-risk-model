"""Feature engineering pipeline for credit risk model.

This module implements a complete sklearn Pipeline that transforms raw transaction data
into model-ready features, including:
- Customer-level aggregations (sum, mean, count, std)
- Temporal features (hour, day, month, year)
- Categorical encoding (one-hot)
- Missing value imputation
- Standardization
- Weight of Evidence (WoE) transformation for credit scoring
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level data to customer level."""

    def __init__(self):
        self.agg_features = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions by CustomerId."""
        if "CustomerId" not in X.columns:
            raise ValueError("CustomerId column required for aggregation")

        # Ensure numeric columns
        numeric_cols = ["Amount", "Value"]
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        agg_dict = {
            "Amount": ["sum", "mean", "count", "std"],
            "Value": ["sum", "mean", "std"],
        }

        # Filter to only existing columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in X.columns}

        if not agg_dict:
            raise ValueError("No numeric columns found for aggregation")

        agg_data = X.groupby("CustomerId").agg(agg_dict).reset_index()
        agg_data.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in agg_data.columns.values
        ]

        # Fill NaN std values (single transaction)
        std_cols = [c for c in agg_data.columns if "std" in c]
        for col in std_cols:
            agg_data[col] = agg_data[col].fillna(0)

        # Rename for clarity
        agg_data.columns = [
            f"agg_{col}" if col != "CustomerId" else col for col in agg_data.columns
        ]

        return agg_data


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from transaction timestamp."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, day, month, year features."""
        X = X.copy()

        if "TransactionStartTime" in X.columns:
            X["TransactionStartTime"] = pd.to_datetime(
                X["TransactionStartTime"], errors="coerce"
            )
            X["txn_hour"] = X["TransactionStartTime"].dt.hour
            X["txn_day"] = X["TransactionStartTime"].dt.day
            X["txn_month"] = X["TransactionStartTime"].dt.month
            X["txn_year"] = X["TransactionStartTime"].dt.year
            X["txn_dow"] = X["TransactionStartTime"].dt.dayofweek  # Day of week

        return X


class CustomerRiskFeatures(BaseEstimator, TransformerMixin):
    """Add customer-level risk features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add fraud rate and behavioral flags."""
        X = X.copy()

        if "FraudResult" in X.columns:
            fraud_by_customer = (
                X.groupby("CustomerId")["FraudResult"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "fraud_count", "count": "total_txns"})
            )
            fraud_by_customer["fraud_rate"] = (
                fraud_by_customer["fraud_count"]
                / fraud_by_customer["total_txns"]
            )

            # Merge back
            X = X.merge(
                fraud_by_customer[["fraud_rate"]], left_on="CustomerId",
                right_index=True, how="left"
            )
            X["fraud_rate"] = X["fraud_rate"].fillna(0)

        # Negative balance flag (credits/refunds)
        if "Amount" in X.columns:
            X["Amount"] = pd.to_numeric(X["Amount"], errors="coerce")
            X["has_negative_amount"] = (X["Amount"] < 0).astype(int)

        return X


def load_raw_data(data_path: Path = None) -> pd.DataFrame:
    """Load raw transaction data."""
    if data_path is None:
        data_path = ROOT / "data" / "raw" / "data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )
    return df


def build_feature_engineering_pipeline(
    categorical_features: list = None,
    numeric_features: list = None,
) -> Pipeline:
    """Build sklearn Pipeline for feature engineering.

    Parameters
    ----------
    categorical_features : list, optional
        Columns to one-hot encode (default: ProductCategory, ChannelId)
    numeric_features : list, optional
        Columns to standardize

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline that transforms raw data to model-ready format
    """
    if categorical_features is None:
        categorical_features = ["ProductCategory", "ChannelId"]
    if numeric_features is None:
        numeric_features = ["agg_Amount_sum", "agg_Amount_mean", "agg_Value_sum",
                            "agg_Value_mean", "fraud_rate", "txn_hour", "txn_month"]

    # Categorical transformer
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Numeric transformer
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Column transformer to apply transforms to different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("temporal", TemporalFeatureExtractor()),
            ("risk_features", CustomerRiskFeatures()),
            ("aggregator", TransactionAggregator()),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def prepare_data(
    data_path: Path = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, process, and split data for modeling.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw data CSV
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, y_train, y_test (if target available)
    """
    # Load data
    df = load_raw_data(data_path)

    # Build and fit pipeline
    pipeline = build_feature_engineering_pipeline()

    # Process data
    X_processed = pipeline.fit_transform(df)

    # Convert to DataFrame
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    # Add back CustomerId for later use
    X_processed["CustomerId"] = df.groupby("CustomerId").ngroup().reset_index(
        drop=True
    )

    print(f"Processed data shape: {X_processed.shape}")
    print(f"Features: {list(X_processed.columns[:10])}...")

    return X_processed


if __name__ == "__main__":
    # Example usage
    X_processed = prepare_data()
    print(f"Final shape: {X_processed.shape}")
    print(f"Columns: {X_processed.columns.tolist()}")
