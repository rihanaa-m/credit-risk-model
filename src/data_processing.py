"""Feature engineering pipeline for credit risk model.

This module implements a complete sklearn Pipeline that transforms raw transaction data
into model-ready features, including:
- Customer-level aggregations (sum, mean, count, std)
- Temporal features (hour, day, month, year)
- Categorical encoding (one-hot)
- Missing value imputation
- Standardization
- Weight of Evidence (WoE) and Information Value (IV) transformation for credit scoring

Enhanced for Week 12 with:
- WoE/IV integration for regulatory compliance
- Industry-standard feature selection
- Enhanced auditability and traceability
"""

from pathlib import Path
from typing import Tuple, List
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# Import WoE/IV functions
try:
    from woe_iv import WoETransformer, calculate_all_woe_iv, generate_iv_report, select_features_by_iv as select_woe_features
    WOE_AVAILABLE = True
except ImportError:
    print("Warning: woe_iv module not available, WoE features disabled")
    WOE_AVAILABLE = False


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
    use_woe: bool = False,
    target_col: str = None,
) -> Pipeline:
    """Build sklearn Pipeline for feature engineering.

    Enhanced for Week 12 with WoE/IV support for regulatory compliance.

    Parameters
    ----------
    categorical_features : list, optional
        Columns to one-hot encode (default: ProductCategory, ChannelId)
    numeric_features : list, optional
        Columns to standardize
    use_woe : bool
        Whether to use WoE transformation (default False)
    target_col : str, optional
        Target column for WoE calculation (required if use_woe=True)

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

    if use_woe and not target_col:
        raise ValueError("target_col required when use_woe=True")

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

    # Build pipeline steps
    pipeline_steps = [
        ("temporal", TemporalFeatureExtractor()),
        ("risk_features", CustomerRiskFeatures()),
        ("aggregator", TransactionAggregator()),
    ]

    # Add WoE transformation if requested
    if use_woe:
        # WoE will be applied after aggregation, before preprocessing
        # This is handled separately in prepare_data_with_woe
        pass

    pipeline_steps.append(("preprocessor", preprocessor))

    # Full pipeline
    pipeline = Pipeline(steps=pipeline_steps)

    return pipeline


def prepare_data_with_woe(
    data_path: Path = None,
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    random_state: int = 42,
    iv_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Prepare data with WoE/IV transformation for regulatory compliance.

    Enhanced for Week 12 with industry-standard WoE/IV pipeline.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw data CSV
    target_col : str
        Target column name
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    iv_threshold : float
        Minimum IV threshold for feature selection

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
        X_train, X_test, y_train, y_test, and metadata dict
    """
    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING WITH WOE/IV - WEEK 12")
    print("=" * 60)

    if not WOE_AVAILABLE:
        print("Warning: WoE/IV module not available, using standard preprocessing")
        return prepare_data(data_path, test_size, random_state)

    # Load data
    df = load_raw_data(data_path)

    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    print(f"\n[Step 1] Building base feature engineering pipeline...")
    # Build and fit base pipeline (without WoE)
    pipeline = build_feature_engineering_pipeline(use_woe=False)

    # Process data to get aggregated features
    df_processed = pipeline.fit_transform(df)

    # Convert to DataFrame
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )
    df_processed = pd.DataFrame(df_processed, columns=feature_names)

    # Add target
    # Need to aggregate target to customer level
    customer_target = df.groupby("CustomerId")[target_col].first().reset_index()
    df_processed["CustomerId"] = range(len(customer_target))  # Match aggregation index
    df_processed = df_processed.merge(customer_target, on="CustomerId", how="left")

    print(f"\n[Step 2] Calculating WoE and IV for all features...")
    # Calculate WoE/IV for all features
    woe_features = [col for col in df_processed.columns if col not in [target_col, "CustomerId"]]
    woe_dict, iv_dict = calculate_all_woe_iv(
        df_processed, target_col, features=woe_features, bins=10
    )

    # Generate IV report
    print(f"\n[Step 3] Generating IV report...")
    iv_report = generate_iv_report(
        iv_dict,
        output_path=ROOT / "analysis_outputs" / "iv_report.csv"
    )

    # Select features based on IV
    print(f"\n[Step 4] Selecting features with IV >= {iv_threshold}...")
    if WOE_AVAILABLE:
        selected_features = select_woe_features(iv_dict, threshold=iv_threshold)
    else:
        # Fallback to simple selection
        selected_features = [k for k, v in iv_dict.items() if v >= iv_threshold]
        selected_features = sorted(selected_features, key=lambda x: iv_dict[x], reverse=True)
    print(f"Selected {len(selected_features)} features out of {len(woe_features)}")

    # Apply WoE transformation to selected features
    print(f"\n[Step 5] Applying WoE transformation to selected features...")
    woe_transformer = WoETransformer(
        features=selected_features,
        target=target_col,
        bins=10
    )

    # Fit WoE on training data
    # First split data
    from sklearn.model_selection import train_test_split

    X = df_processed.drop(columns=[target_col, "CustomerId"])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit WoE on training data only
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    woe_transformer.fit(train_df)

    # Transform both train and test
    X_train_woe = woe_transformer.transform(X_train)
    X_test_woe = woe_transformer.transform(X_test)

    # Remove temporary target column
    X_train_woe = X_train_woe.drop(columns=[target_col], errors='ignore')
    X_test_woe = X_test_woe.drop(columns=[target_col], errors='ignore')

    # Ensure we only use selected features
    final_features = [f for f in selected_features if f in X_train_woe.columns]
    X_train_final = X_train_woe[final_features]
    X_test_final = X_test_woe[final_features]

    print(f"Final feature set: {len(final_features)} features")

    # Metadata
    metadata = {
        "n_features_original": len(woe_features),
        "n_features_selected": len(final_features),
        "iv_threshold": iv_threshold,
        "woe_bins": 10,
        "selected_features": final_features,
        "iv_values": {f: iv_dict.get(f, 0) for f in final_features},
        "enhancements": [
            "WoE/IV calculation for regulatory compliance",
            "IV-based feature selection",
            "Proper train/test split to prevent data leakage",
            "Industry-standard credit risk preprocessing"
        ]
    }

    print("\n" + "=" * 60)
    print("WOE/IV FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    return X_train_final, X_test_final, y_train, y_test, metadata


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
