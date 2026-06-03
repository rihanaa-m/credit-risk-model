"""RFM (Recency, Frequency, Monetary) proxy target variable engineering.

This module implements the proxy target variable using RFM-based customer segmentation
and K-Means clustering to identify high-risk customer segments.

Key steps:
1. Calculate RFM metrics for each customer
2. Scale RFM features
3. Apply K-Means clustering (k=3)
4. Identify and label high-risk cluster
5. Merge back to main dataset
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def calculate_rfm_metrics(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """Calculate RFM metrics per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level data with CustomerId, TransactionStartTime, Amount, Value
    snapshot_date : pd.Timestamp, optional
        Date to calculate recency from (default: max date in data)

    Returns
    -------
    pd.DataFrame
        Customer-level RFM metrics
    """
    # Parse timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(df["TransactionStartTime"]):
        df = df.copy()
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], utc=True, errors="coerce"
        )

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()

    print(f"RFM snapshot date: {snapshot_date}")

    # Recency: days since last transaction
    last_txn = df.groupby("CustomerId")["TransactionStartTime"].max()
    recency = (snapshot_date - last_txn).dt.days

    # Frequency: number of transactions
    frequency = df.groupby("CustomerId").size()

    # Monetary: sum of transaction values (use absolute Value)
    monetary = df.groupby("CustomerId")["Value"].sum()

    # Combine into RFM DataFrame
    rfm = pd.DataFrame(
        {
            "recency_days": recency,
            "frequency": frequency,
            "monetary_value": monetary,
        }
    ).reset_index()

    print(f"RFM stats:")
    print(rfm[["recency_days", "frequency", "monetary_value"]].describe())

    return rfm


def cluster_customers(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """Cluster customers using K-Means on RFM metrics.

    Parameters
    ----------
    rfm : pd.DataFrame
        RFM metrics per customer
    n_clusters : int
        Number of clusters (default 3)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[pd.DataFrame, KMeans]
        RFM data with cluster labels, and fitted KMeans object
    """
    rfm = rfm.copy()

    # Log-transform and handle skew
    rfm["recency_log"] = np.log1p(rfm["recency_days"])
    rfm["frequency_log"] = np.log1p(rfm["frequency"])
    rfm["monetary_log"] = np.log1p(rfm["monetary_value"])

    # Prepare features for clustering
    X_rfm = rfm[["recency_log", "frequency_log", "monetary_log"]].values

    # Standardize
    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(X_rfm)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_rfm_scaled)

    print(f"Cluster distribution:")
    print(rfm["cluster"].value_counts().sort_index())

    print(f"Cluster centers (scaled RFM):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"  Cluster {i}: recency={center[0]:.2f}, freq={center[1]:.2f}, "
              f"monetary={center[2]:.2f}")

    return rfm, kmeans, scaler


def identify_high_risk_cluster(rfm: pd.DataFrame) -> int:
    """Identify which cluster represents high-risk customers.

    High-risk = low engagement = high recency, low frequency, low monetary.

    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with cluster labels

    Returns
    -------
    int
        Cluster label for high-risk segment
    """
    # Analyze cluster characteristics
    cluster_stats = rfm.groupby("cluster")[
        ["recency_days", "frequency", "monetary_value"]
    ].mean()

    print(f"Mean metrics by cluster:")
    print(cluster_stats)

    # High-risk: high recency (inactive), low frequency, low monetary
    # Calculate a risk score for each cluster
    cluster_stats["risk_score"] = (
        (cluster_stats["recency_days"] / cluster_stats["recency_days"].max())
        - (cluster_stats["frequency"] / cluster_stats["frequency"].max())
        - (cluster_stats["monetary_value"] / cluster_stats["monetary_value"].max())
    )

    print(f"\nRisk scores by cluster:")
    print(cluster_stats[["risk_score"]])

    high_risk_cluster = cluster_stats["risk_score"].idxmax()
    print(f"\nHigh-risk cluster identified: {high_risk_cluster}")
    print(f"  Customers in high-risk cluster: {(rfm['cluster'] == high_risk_cluster).sum()}")

    return int(high_risk_cluster)


def create_proxy_target(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    high_risk_cluster: int,
) -> pd.DataFrame:
    """Create is_high_risk binary target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Original transaction-level data
    rfm : pd.DataFrame
        Customer-level RFM with cluster labels
    high_risk_cluster : int
        Cluster label for high-risk segment

    Returns
    -------
    pd.DataFrame
        Original data with is_high_risk column added
    """
    df = df.copy()

    # Create binary high_risk indicator
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    # Merge back to transaction level
    df = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

    # Verify no missing values
    assert df["is_high_risk"].isna().sum() == 0, "NaN values in is_high_risk column"

    # Report label distribution
    print(f"\nTarget variable distribution:")
    print(df["is_high_risk"].value_counts())
    print(f"High-risk proportion: {df['is_high_risk'].mean():.4%}")

    return df


def engineer_proxy_target(
    data_path: Path = None,
    snapshot_date: pd.Timestamp = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """End-to-end RFM proxy target engineering.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw transaction data
    snapshot_date : pd.Timestamp, optional
        Snapshot date for RFM calculation
    n_clusters : int
        Number of clusters for K-Means
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        Data with proxy target, RFM metrics, metadata dict
    """
    # Load data
    if data_path is None:
        data_path = ROOT / "data" / "raw" / "data.csv"

    df = pd.read_csv(data_path)
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )

    # Step 1: Calculate RFM
    rfm = calculate_rfm_metrics(df, snapshot_date=snapshot_date)

    # Step 2: Cluster
    rfm_clustered, kmeans, scaler = cluster_customers(
        rfm, n_clusters=n_clusters, random_state=random_state
    )

    # Step 3: Identify high-risk cluster
    high_risk_cluster = identify_high_risk_cluster(rfm_clustered)

    # Step 4: Create target variable
    df_with_target = create_proxy_target(df, rfm_clustered, high_risk_cluster)

    # Metadata
    metadata = {
        "snapshot_date": snapshot_date or df["TransactionStartTime"].max(),
        "n_clusters": n_clusters,
        "high_risk_cluster": high_risk_cluster,
        "random_state": random_state,
        "n_customers": len(rfm_clustered),
        "n_high_risk": (rfm_clustered["cluster"] == high_risk_cluster).sum(),
    }

    return df_with_target, rfm_clustered, metadata


if __name__ == "__main__":
    # Example usage
    df_target, rfm, metadata = engineer_proxy_target()
    print(f"\nMetadata: {metadata}")
    print(f"Data with proxy target shape: {df_target.shape}")
    print(f"is_high_risk column added: {'is_high_risk' in df_target.columns}")
