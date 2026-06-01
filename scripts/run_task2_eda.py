"""Run Task 2 EDA and save figures to analysis_outputs/task2/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "data.csv"
OUTPUT_DIR = ROOT / "analysis_outputs" / "task2"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )
    return df


def save_value_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = df["Value"].clip(upper=df["Value"].quantile(0.99))
    ax.hist(vals, bins=60, color="#2E86AB", edgecolor="white")
    ax.set_title("Transaction Value Distribution (99th pct cap)")
    ax.set_xlabel("Value (UGX)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = OUTPUT_DIR / "value_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_fraud_by_category(df: pd.DataFrame) -> Path:
    rates = (
        df.groupby("ProductCategory")["FraudResult"]
        .agg(rate="mean", n="count")
        .query("n >= 100")
        .sort_values("rate", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=rates.reset_index(),
        x="ProductCategory",
        y="rate",
        hue="ProductCategory",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Fraud Rate by Product Category (n >= 100)")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = OUTPUT_DIR / "fraud_rate_by_category.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_temporal_volume(df: pd.DataFrame) -> Path:
    daily = (
        df.set_index("TransactionStartTime")
        .resample("D")
        .size()
        .rename("transactions")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, color="#E94F37")
    ax.set_title("Daily Transaction Volume")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    path = OUTPUT_DIR / "daily_transaction_volume.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric = df[["Amount", "Value", "CountryCode", "PricingStrategy", "FraudResult"]]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation — Numeric Features")
    fig.tight_layout()
    path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_outlier_boxplots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=df, y="Value", ax=axes[0], color="#2E86AB")
    axes[0].set_title("Value — Outlier Check")
    sns.boxplot(data=df, y="Amount", ax=axes[1], color="#E94F37")
    axes[1].set_title("Amount — Outlier Check")
    fig.tight_layout()
    path = OUTPUT_DIR / "outlier_boxplots.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    paths = [
        save_value_distribution(df),
        save_fraud_by_category(df),
        save_temporal_volume(df),
        save_correlation_heatmap(df),
        save_outlier_boxplots(df),
    ]
    print(f"Loaded {len(df):,} rows, {df['CustomerId'].nunique():,} customers")
    print(f"Fraud rate: {df['FraudResult'].mean():.4%}")
    print("Saved figures:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
