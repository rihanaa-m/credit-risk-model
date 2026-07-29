"""Enhanced end-to-end training script for Week 12 improvements.

This script integrates all Week 12 enhancements:
1. Enhanced RFM proxy target engineering with explicit scoring
2. WoE/IV feature transformation for regulatory compliance
3. LightGBM integration for improved performance
4. Enhanced MLflow tracking with comprehensive audit trail
5. Industry-standard credit risk preprocessing pipeline

Usage:
    python main_train_enhanced.py
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

import pandas as pd
import mlflow
import numpy as np

# Import enhanced modules
from src.proxy_target import engineer_proxy_target
from src.data_processing import prepare_data_with_woe
from src.train import train_and_track_models, select_best_model

ROOT = Path(__file__).resolve().parents[0]


def main():
    """Main training pipeline with Week 12 enhancements."""
    print("=" * 80)
    print("WEEK 12 ENHANCED CREDIT RISK MODEL TRAINING")
    print("=" * 80)

    # Set up MLflow
    mlflow.set_tracking_uri(ROOT / "mlruns")
    experiment_name = "credit_risk_week12_enhanced"
    mlflow.set_experiment(experiment_name)

    print(f"\nMLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Step 1: Enhanced RFM Proxy Target Engineering
    print("\n" + "=" * 80)
    print("STEP 1: ENHANCED RFM PROXY TARGET ENGINEERING")
    print("=" * 80)

    data_path = ROOT / "data" / "raw" / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    # Engineer proxy target with enhanced RFM framework
    df_with_target, rfm_metrics, rfm_metadata = engineer_proxy_target(
        data_path=data_path,
        n_clusters=3,
        random_state=42
    )

    # Save processed data with target
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)
    output_path = processed_dir / "data_with_enhanced_target.csv"
    df_with_target.to_csv(output_path, index=False)
    print(f"Saved data with enhanced target to {output_path}")

    # Log RFM metadata to MLflow
    with mlflow.start_run(run_name="rfm_target_engineering"):
        mlflow.log_params(rfm_metadata)
        mlflow.log_metric("n_customers", rfm_metadata["n_customers"])
        mlflow.log_metric("n_high_risk", rfm_metadata["n_high_risk"])
        mlflow.log_metric("high_risk_proportion", rfm_metadata["high_risk_proportion"])
        print("RFM metadata logged to MLflow")

    # Step 2: WoE/IV Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 2: WOE/IV FEATURE ENGINEERING FOR REGULATORY COMPLIANCE")
    print("=" * 80)

    try:
        X_train, X_test, y_train, y_test, fe_metadata = prepare_data_with_woe(
            data_path=processed_dir / "data_with_enhanced_target.csv",
            target_col="is_high_risk",
            test_size=0.2,
            random_state=42,
            iv_threshold=0.02
        )

        # Log feature engineering metadata
        with mlflow.start_run(run_name="woe_iv_feature_engineering"):
            mlflow.log_params(fe_metadata)
            mlflow.log_metric("n_features_original", fe_metadata["n_features_original"])
            mlflow.log_metric("n_features_selected", fe_metadata["n_features_selected"])
            print("WoE/IV metadata logged to MLflow")

    except Exception as e:
        print(f"Error in WoE/IV feature engineering: {e}")
        print("Falling back to standard preprocessing...")
        # Fallback to standard preprocessing
        from src.data_processing import prepare_data
        X_train, X_test, y_train, y_test = prepare_data(
            data_path=processed_dir / "data_with_enhanced_target.csv",
            test_size=0.2,
            random_state=42
        )
        fe_metadata = {"fallback": True, "method": "standard_preprocessing"}

    # Step 3: Enhanced Model Training
    print("\n" + "=" * 80)
    print("STEP 3: ENHANCED MODEL TRAINING WITH LIGHTGBM")
    print("=" * 80)

    models_results = train_and_track_models(
        X_train, y_train, X_test, y_test,
        experiment_name=experiment_name,
        random_state=42
    )

    # Step 4: Model Selection and Evaluation
    print("\n" + "=" * 80)
    print("STEP 4: MODEL SELECTION AND EVALUATION")
    print("=" * 80)

    best_model_name, best_model = select_best_model(models_results, metric="pr_auc")

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Metrics:")
    for metric, value in models_results[best_model_name]["metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nAll Model Comparison:")
    for model_name, result in models_results.items():
        print(f"\n{model_name}:")
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:.4f}")

    # Log final results to MLflow
    with mlflow.start_run(run_name="final_selection"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_params(models_results[best_model_name]["metrics"])
        mlflow.log_param("week12_enhancements", True)
        mlflow.log_param("enhanced_rfm", True)
        mlflow.log_param("woe_iv_features", not fe_metadata.get("fallback", False))
        mlflow.log_param("lightgbm_enabled", True)

    print("\n" + "=" * 80)
    print("WEEK 12 ENHANCED TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nAll results tracked in MLflow experiment: {experiment_name}")
    print(f"Best model: {best_model_name}")
    print(f"Processed data saved to: {output_path}")
    print(f"IV report saved to: {ROOT / 'analysis_outputs' / 'iv_report.csv'}")

    return models_results, best_model_name


if __name__ == "__main__":
    try:
        results, best_model = main()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)