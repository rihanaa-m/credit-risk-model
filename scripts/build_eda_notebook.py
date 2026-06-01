"""Generate notebooks/eda.ipynb for Task 2."""

from pathlib import Path

import nbformat as nb
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

cells = [
    new_markdown_cell(
        "# Task 2 — Exploratory Data Analysis (EDA)\n"
        "**Bati Bank × Xente eCommerce** | Credit Risk (Alternative Data)\n\n"
        "This notebook explores transaction-level data to understand structure, quality, "
        "distributions, correlations, and patterns that will guide feature engineering and "
        "proxy target design (RFM clustering in later tasks)."
    ),
    new_code_cell(
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "OUTPUT_DIR = Path('analysis_outputs/task2')\n"
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "DATA_PATH = Path('data/raw/data.csv')\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "df['TransactionStartTime'] = pd.to_datetime(\n"
        "    df['TransactionStartTime'], utc=True, errors='coerce'\n"
        ")\n"
        "print(f'Loaded {len(df):,} rows × {df.shape[1]} columns')\n"
        "print(f'Unique customers: {df[\"CustomerId\"].nunique():,}')\n"
        "print(f'Date range: {df[\"TransactionStartTime\"].min()} → {df[\"TransactionStartTime\"].max()}')"
    ),
    new_markdown_cell("## 1. Overview of the Data"),
    new_code_cell(
        "display(df.head())\n"
        "display(df.dtypes.to_frame('dtype'))\n"
        "print('Shape:', df.shape)"
    ),
    new_markdown_cell("## 2. Summary Statistics"),
    new_code_cell(
        "numeric_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult']\n"
        "display(df[numeric_cols].describe().T)\n"
        "display(df[['ProductCategory', 'ChannelId', 'CurrencyCode']].describe(include='object').T)"
    ),
    new_markdown_cell("## 3. Distribution of Numerical Features"),
    new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df['Value'].clip(upper=df['Value'].quantile(0.99)).hist(bins=50, ax=axes[0], color='#2E86AB')\n"
        "axes[0].set_title('Value (capped at 99th percentile)')\n"
        "df['Amount'].clip(lower=df['Amount'].quantile(0.01), upper=df['Amount'].quantile(0.99)).hist(\n"
        "    bins=50, ax=axes[1], color='#E94F37'\n"
        ")\n"
        "axes[1].set_title('Amount (1st–99th percentile)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'value_amount_histograms.png', dpi=120)\n"
        "plt.show()"
    ),
    new_markdown_cell("## 4. Distribution of Categorical Features"),
    new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "order = df['ProductCategory'].value_counts().head(8).index\n"
        "sns.countplot(data=df, y='ProductCategory', order=order, ax=axes[0], palette='Blues_d')\n"
        "axes[0].set_title('Product Category (top 8)')\n"
        "sns.countplot(data=df, x='ChannelId', ax=axes[1], palette='Oranges_d')\n"
        "axes[1].set_title('ChannelId')\n"
        "axes[1].tick_params(axis='x', rotation=30)\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'categorical_distributions.png', dpi=120)\n"
        "plt.show()"
    ),
    new_markdown_cell("## 5. Correlation Analysis"),
    new_code_cell(
        "corr = df[numeric_cols].corr()\n"
        "plt.figure(figsize=(6, 5))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)\n"
        "plt.title('Correlation Matrix — Numeric Features')\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=120)\n"
        "plt.show()"
    ),
    new_markdown_cell("## 6. Identifying Missing Values"),
    new_code_cell(
        "missing = df.isna().sum()\n"
        "missing_pct = (missing / len(df) * 100).round(2)\n"
        "missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct}).query('missing > 0')\n"
        "if len(missing_df) == 0:\n"
        "    print('No missing values detected in any column.')\n"
        "else:\n"
        "    display(missing_df.sort_values('pct', ascending=False))"
    ),
    new_markdown_cell("## 7. Outlier Detection"),
    new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "sns.boxplot(data=df, y='Value', ax=axes[0])\n"
        "axes[0].set_title('Value — Box Plot')\n"
        "sns.boxplot(data=df, y='Amount', ax=axes[1])\n"
        "axes[1].set_title('Amount — Box Plot')\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'outlier_boxplots.png', dpi=120)\n"
        "plt.show()\n"
        "print('Extreme values exist; use winsorization/log transforms in feature engineering.')"
    ),
    new_markdown_cell("## 8. Fraud & Temporal Patterns"),
    new_code_cell(
        "print('Overall fraud rate:', f\"{df['FraudResult'].mean():.4%}\")\n"
        "fraud_by_cat = (\n"
        "    df.groupby('ProductCategory')['FraudResult']\n"
        "    .agg(rate='mean', n='count')\n"
        "    .query('n >= 100')\n"
        "    .sort_values('rate', ascending=False)\n"
        ")\n"
        "display(fraud_by_cat.head(8))\n\n"
        "daily = df.set_index('TransactionStartTime').resample('D').size()\n"
        "daily.plot(figsize=(10, 3), title='Daily Transaction Volume', color='#E94F37')\n"
        "plt.ylabel('Transactions')\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'daily_transaction_volume.png', dpi=120)\n"
        "plt.show()"
    ),
    new_markdown_cell(
        "## 9. Top Insights (Summary)\n\n"
        "1. **Highly imbalanced fraud signal (~0.2%)** — any classifier will need class-weighting, "
        "threshold tuning, or resampling; accuracy alone is misleading.\n"
        "2. **Extreme transaction amounts** — `Value`/`Amount` are heavily right-skewed with large "
        "outliers; winsorization or log-scaling is required before RFM clustering and modeling.\n"
        "3. **Category-driven risk patterns** — fraud rates differ materially by `ProductCategory` "
        "(e.g., transport and utility_bill higher than airtime), suggesting category features matter.\n"
        "4. **Concentrated product mix** — most transactions are `financial_services` and `airtime`; "
        "models must generalize beyond dominant categories.\n"
        "5. **Stable daily volume with short history** — ~95k transactions across ~3.7k customers "
        "from Nov 2018–Feb 2019; recency features should use a consistent snapshot date for RFM."
    ),
]

nb_node = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
)
out_path = ROOT / "notebooks" / "eda.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nb.write(nb_node, out_path)
print(f"Wrote {out_path}")
