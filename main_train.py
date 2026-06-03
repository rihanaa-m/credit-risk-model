"""End-to-end training pipeline orchestration.

This script runs the complete workflow:
1. Load raw data
2. Engineer features (Task 3)
3. Create proxy target via RFM (Task 4)
4. Train and track models with MLflow (Task 5)
"""

from pathlib import Path
import pandas as pd

from src.data_processing import prepare_data
from src.proxy_target import engineer_proxy_target
from src.train import (
    split_data,
    train_and_track_models,
    select_best_model,
)

ROOT = Path(__file__).resolve().parents[0]


def main():
    """Run complete training pipeline."""
    print("\n" + "="*70)
    print("CREDIT RISK MODEL — END-TO-END TRAINING PIPELINE")
    print("="*70)

    # Step 1: Feature Engineering (Task 3)
    print("\n[1] FEATURE ENGINEERING (Task 3)")
    print("-" * 70)
    try:
        X_processed = prepare_data(
            data_path=ROOT / "data" / "raw" / "data.csv"
        )
        print(f"✅ Features engineered: {X_processed.shape}")
    except Exception as e:
        print(f"⚠️ Feature engineering error (data may not exist locally): {e}")
        print("   Continuing with demonstration setup...")
        # For demo, create synthetic processed data
        X_processed = pd.DataFrame({
            "agg_Amount_sum": [50000, 10000, 75000],
            "agg_Amount_mean": [5000, 1000, 7500],
            "agg_Value_sum": [50000, 10000, 75000],
            "agg_Value_mean": [5000, 1000, 7500],
            "fraud_rate": [0.0, 0.1, 0.0],
            "txn_hour": [10, 14, 9],
            "txn_month": [2, 3, 2],
            "CustomerId": ["C001", "C002", "C003"],
        })

    # Step 2: Proxy Target via RFM (Task 4)
    print("\n[2] PROXY TARGET ENGINEERING (Task 4 — RFM Clustering)")
    print("-" * 70)
    try:
        df_target, rfm, metadata = engineer_proxy_target(
            data_path=ROOT / "data" / "raw" / "data.csv",
            n_clusters=3,
            random_state=42,
        )
        print(f"✅ Proxy target created: {df_target.shape}")
        print(f"   Metadata: {metadata}")

        # For training, use the RFM-based target
        y = df_target["is_high_risk"]

    except Exception as e:
        print(f"⚠️ RFM clustering error: {e}")
        print("   Using synthetic target for demonstration...")
        # For demo, create synthetic target
        y = pd.Series([0, 1, 0], name="is_high_risk")

    # Align X and y
    X_processed = X_processed.head(len(y)).reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Step 3: Train/Test Split
    print("\n[3] DATA SPLITTING")
    print("-" * 70)
    X_train, X_test, y_train, y_test = split_data(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Step 4: Model Training & MLflow Tracking (Task 5)
    print("\n[4] MODEL TRAINING & MLflow TRACKING (Task 5)")
    print("-" * 70)
    models_results = train_and_track_models(
        X_train,
        y_train,
        X_test,
        y_test,
        experiment_name="credit_risk_final_submission",
        random_state=42,
    )

    # Step 5: Select Best Model
    print("\n[5] MODEL SELECTION")
    print("-" * 70)
    best_model_name, best_model = select_best_model(
        models_results,
        metric="pr_auc",
    )

    print(f"\n{'='*70}")
    print(f"✅ TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Best model: {best_model_name}")
    print(f"Experiments tracked in: mlruns/")
    print(f"\nNext steps:")
    print(f"  1. Review MLflow UI: mlflow ui")
    print(f"  2. Deploy API: python -m uvicorn src.api.main:app")
    print(f"  3. Docker: docker-compose up")
    print(f"  4. Review: reports/final_report.pdf")

    return models_results, best_model_name, best_model


if __name__ == "__main__":
    main()
