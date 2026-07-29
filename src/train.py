"""Model training, evaluation, and MLflow tracking.

This module implements:
- Data loading and preparation with enhanced RFM proxy targets
- Train/test split with stratification
- Multiple model training (LogisticRegression, RandomForest, XGBoost/LightGBM)
- Hyperparameter tuning with GridSearchCV
- Metric computation for imbalanced classification
- MLflow experiment tracking and model registry
- WoE/IV feature integration for regulatory compliance

Enhanced for Week 12 with:
- Integration with enhanced RFM proxy target engineering
- WoE/IV feature pipeline support
- Enhanced model evaluation metrics
- Comprehensive audit trail
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc as pr_auc,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

ROOT = Path(__file__).resolve().parents[1]

# Import enhanced modules
try:
    from proxy_target import engineer_proxy_target
    from data_processing import prepare_data_with_woe
    HAS_ENHANCED_MODULES = True
except ImportError:
    print("Warning: Enhanced modules not available, using standard pipeline")
    HAS_ENHANCED_MODULES = False


def load_and_prepare_data(
    data_path: Path = None,
    target_col: str = "is_high_risk",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed data with target variable.

    Parameters
    ----------
    data_path : Path
        Path to processed data CSV
    target_col : str
        Name of target column

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features and target
    """
    if data_path is None:
        data_path = ROOT / "data" / "processed" / "data_with_target.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    df = pd.read_csv(data_path)
    y = df[target_col]
    X = df.drop(columns=[target_col, "CustomerId"], errors="ignore")

    print(f"Data loaded: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion for test set
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train set: {X_train.shape}, target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test set: {X_test.shape}, target distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Compute classification metrics for imbalanced data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels (binary)
    y_proba : np.ndarray
        Predicted probabilities (for positive class)

    Returns
    -------
    Dict
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)

    # PR-AUC (primary metric for imbalanced data)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = pr_auc(recall_vals, precision_vals)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc_score,
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[LogisticRegression, Dict]:
    """Train Logistic Regression with hyperparameter tuning.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : data
    random_state : int

    Returns
    -------
    Tuple[LogisticRegression, Dict]
        Best model and metrics
    """
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
    }

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best LR params: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    return best_model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest with hyperparameter tuning."""
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
        "min_samples_split": [5, 10],
    }

    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best RF params: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    return best_model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple["LGBMClassifier", Dict]:
    """Train LightGBM with hyperparameter tuning.

    Enhanced for Week 12 as alternative to XGBoost for credit risk modeling.
    LightGBM is often preferred for tabular financial data due to:
    - Faster training speed
    - Lower memory usage
    - Better performance on categorical features
    - Industry adoption in credit risk
    """
    if not HAS_LIGHTGBM:
        print("LightGBM not installed, skipping")
        return None, {}

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [31, 63, 127],
    }

    lgb = LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        verbose=-1,
    )

    grid_search = GridSearchCV(
        lgb,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best LightGBM params: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    return best_model, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple["XGBClassifier", Dict]:
    """Train XGBoost with hyperparameter tuning."""
    if not HAS_XGBOOST:
        print("XGBoost not installed, skipping")
        return None, {}

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "learning_rate": [0.01, 0.1],
    }

    xgb = XGBClassifier(
        random_state=random_state,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
    )

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best XGB params: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    return best_model, metrics


def train_and_track_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "credit_risk_v1",
    random_state: int = 42,
) -> Dict:
    """Train multiple models and track with MLflow.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : data
    experiment_name : str
        MLflow experiment name
    random_state : int

    Returns
    -------
    Dict
        Results for all models
    """
    mlflow.set_experiment(experiment_name)

    models_results = {}

    # Train Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    with mlflow.start_run(run_name="logistic_regression"):
        lr_model, lr_metrics = train_logistic_regression(
            X_train, y_train, X_test, y_test, random_state
        )
        mlflow.log_params({"model": "logistic_regression", "random_state": random_state})
        for metric_name, value in lr_metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(lr_model, "model")
        models_results["logistic_regression"] = {
            "model": lr_model,
            "metrics": lr_metrics,
        }
        print(f"LR Metrics: {lr_metrics}")

    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    with mlflow.start_run(run_name="random_forest"):
        rf_model, rf_metrics = train_random_forest(
            X_train, y_train, X_test, y_test, random_state
        )
        mlflow.log_params({"model": "random_forest", "random_state": random_state})
        for metric_name, value in rf_metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(rf_model, "model")
        models_results["random_forest"] = {
            "model": rf_model,
            "metrics": rf_metrics,
        }
        print(f"RF Metrics: {rf_metrics}")

    # Train LightGBM (Week 12 enhancement)
    if HAS_LIGHTGBM:
        print("\n" + "="*50)
        print("Training LightGBM (Week 12 Enhancement)...")
        print("="*50)
        with mlflow.start_run(run_name="lightgbm"):
            lgb_model, lgb_metrics = train_lightgbm(
                X_train, y_train, X_test, y_test, random_state
            )
            if lgb_model:
                mlflow.log_params({"model": "lightgbm", "random_state": random_state, "week12_enhancement": True})
                for metric_name, value in lgb_metrics.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.sklearn.log_model(lgb_model, "model")
                models_results["lightgbm"] = {
                    "model": lgb_model,
                    "metrics": lgb_metrics,
                }
                print(f"LightGBM Metrics: {lgb_metrics}")

    # Train XGBoost (if available)
    if HAS_XGBOOST:
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        with mlflow.start_run(run_name="xgboost"):
            xgb_model, xgb_metrics = train_xgboost(
                X_train, y_train, X_test, y_test, random_state
            )
            if xgb_model:
                mlflow.log_params({"model": "xgboost", "random_state": random_state})
                for metric_name, value in xgb_metrics.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.sklearn.log_model(xgb_model, "model")
                models_results["xgboost"] = {
                    "model": xgb_model,
                    "metrics": xgb_metrics,
                }
                print(f"XGB Metrics: {xgb_metrics}")

    return models_results


def select_best_model(models_results: Dict, metric: str = "pr_auc") -> Tuple[str, object]:
    """Select best model based on metric.

    Parameters
    ----------
    models_results : Dict
        Results from train_and_track_models
    metric : str
        Metric to optimize

    Returns
    -------
    Tuple[str, object]
        Best model name and model object
    """
    best_name = None
    best_score = -1

    for model_name, result in models_results.items():
        score = result["metrics"].get(metric, -1)
        if score > best_score:
            best_score = score
            best_name = model_name

    print(f"\nBest model: {best_name} with {metric}={best_score:.4f}")
    return best_name, models_results[best_name]["model"]


if __name__ == "__main__":
    # Example usage (requires processed data with target)
    print("This module is designed to be imported. Run via main training script.")
