"""Weight of Evidence (WoE) and Information Value (IV) calculation for credit risk.

This module implements industry-standard WoE and IV calculations essential for:
- Regulatory compliance in credit risk modeling
- Feature selection and importance ranking
- Monotonic relationship transformation
- Model interpretability and auditability

Key concepts:
- WoE: Measures the strength of relationship between features and target
- IV: Quantifies predictive power of features (industry standard thresholds)
- Binning: Continuous features are binned for WoE calculation

Industry standard IV thresholds:
- < 0.02: Not useful
- 0.02 - 0.1: Weak predictor
- 0.1 - 0.3: Medium predictor
- 0.3 - 0.5: Strong predictor
- > 0.5: Suspicious (potential overfitting)
"""

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    min_bin_size: int = 30,
) -> Tuple[pd.DataFrame, float]:
    """Calculate WoE and IV for a single feature.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing feature and target
    feature : str
        Feature column name
    target : str
        Target column name (binary: 0/1)
    bins : int
        Number of bins for continuous features
    min_bin_size : int
        Minimum samples per bin for stability

    Returns
    -------
    Tuple[pd.DataFrame, float]
        WoE DataFrame with bins, and IV value
    """
    df = df.copy()

    # Handle continuous vs categorical features
    if df[feature].dtype == 'object' or df[feature].nunique() <= 10:
        # Categorical feature - use existing categories
        df['bin'] = df[feature].astype(str)
    else:
        # Continuous feature - create equal-sized bins
        df['bin'] = pd.qcut(
            df[feature],
            q=bins,
            duplicates='drop',
            labels=False
        ).astype(str)

    # Calculate event and non-event counts per bin
    bin_stats = df.groupby('bin').agg({
        target: ['sum', 'count']
    }).reset_index()

    bin_stats.columns = ['bin', 'events', 'total']
    bin_stats['non_events'] = bin_stats['total'] - bin_stats['events']

    # Remove bins with too few samples
    bin_stats = bin_stats[bin_stats['total'] >= min_bin_size]

    # Calculate WoE
    total_events = bin_stats['events'].sum()
    total_non_events = bin_stats['non_events'].sum()

    # Add small constant to avoid division by zero
    epsilon = 0.0001

    bin_stats['event_rate'] = (bin_stats['events'] + epsilon) / (total_events + epsilon)
    bin_stats['non_event_rate'] = (bin_stats['non_events'] + epsilon) / (total_non_events + epsilon)

    bin_stats['woe'] = np.log(bin_stats['non_event_rate'] / bin_stats['event_rate'])

    # Calculate IV contribution
    bin_stats['iv_contrib'] = (bin_stats['event_rate'] - bin_stats['non_event_rate']) * bin_stats['woe']

    # Total IV
    iv = bin_stats['iv_contrib'].sum()

    return bin_stats[['bin', 'woe', 'iv_contrib']], iv


def calculate_all_woe_iv(
    df: pd.DataFrame,
    target: str,
    features: List[str] = None,
    bins: int = 10,
    min_bin_size: int = 30,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """Calculate WoE and IV for all features.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    target : str
        Target column name
    features : List[str], optional
        List of features to process (default: all except target)
    bins : int
        Number of bins for continuous features
    min_bin_size : int
        Minimum samples per bin

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, float]]
        Dictionary of WoE DataFrames per feature, and IV values
    """
    if features is None:
        features = [col for col in df.columns if col != target]

    woe_dict = {}
    iv_dict = {}

    print(f"Calculating WoE and IV for {len(features)} features...")

    for feature in features:
        try:
            woe_df, iv = calculate_woe_iv(
                df, feature, target, bins=bins, min_bin_size=min_bin_size
            )
            woe_dict[feature] = woe_df
            iv_dict[feature] = iv

            # Interpret IV strength
            if iv < 0.02:
                strength = "Not useful"
            elif iv < 0.1:
                strength = "Weak"
            elif iv < 0.3:
                strength = "Medium"
            elif iv < 0.5:
                strength = "Strong"
            else:
                strength = "Suspicious"

            print(f"  {feature}: IV = {iv:.4f} ({strength})")

        except Exception as e:
            print(f"  {feature}: Error - {str(e)}")
            woe_dict[feature] = None
            iv_dict[feature] = 0.0

    return woe_dict, iv_dict


class WoETransformer(BaseEstimator, TransformerMixin):
    """Transform features using Weight of Evidence encoding.

    This transformer learns WoE mappings from training data and applies
    them consistently to new data, ensuring no data leakage.
    """

    def __init__(self, features: List[str], target: str, bins: int = 10):
        """Initialize WoE transformer.

        Parameters
        ----------
        features : List[str]
            Features to transform with WoE
        target : str
            Target variable for WoE calculation
        bins : int
            Number of bins for continuous features
        """
        self.features = features
        self.target = target
        self.bins = bins
        self.woe_mappings = {}
        self.feature_bins = {}
        self.iv_values = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit WoE mappings from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Training target (if not in X)

        Returns
        -------
        self
        """
        df = X.copy()

        # Add target if provided separately
        if y is not None:
            df[self.target] = y.values

        print("Fitting WoE transformer...")

        for feature in self.features:
            if feature not in df.columns:
                print(f"  Warning: {feature} not in data, skipping")
                continue

            try:
                # Calculate WoE for this feature
                woe_df, iv = calculate_woe_iv(
                    df, feature, self.target, bins=self.bins
                )

                # Store bin boundaries for continuous features
                if df[feature].dtype != 'object' and df[feature].nunique() > 10:
                    # Create bin boundaries from quantiles
                    _, bin_edges = pd.qcut(
                        df[feature],
                        q=self.bins,
                        duplicates='drop',
                        retbins=True
                    )
                    self.feature_bins[feature] = bin_edges

                # Store WoE mapping (bin -> woe)
                woe_mapping = dict(zip(woe_df['bin'], woe_df['woe']))
                self.woe_mappings[feature] = woe_mapping
                self.iv_values[feature] = iv

                print(f"  {feature}: IV = {iv:.4f}")

            except Exception as e:
                print(f"  Error fitting {feature}: {str(e)}")
                self.woe_mappings[feature] = {}
                self.iv_values[feature] = 0.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted WoE mappings.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Data with WoE-transformed features
        """
        X = X.copy()

        for feature in self.features:
            if feature not in self.woe_mappings or not self.woe_mappings[feature]:
                continue

            if feature not in X.columns:
                print(f"  Warning: {feature} not in data, skipping")
                continue

            try:
                # Apply binning for continuous features
                if feature in self.feature_bins:
                    X[f'{feature}_bin'] = pd.cut(
                        X[feature],
                        bins=self.feature_bins[feature],
                        labels=False,
                        include_lowest=True
                    ).astype(str)
                    bin_col = f'{feature}_bin'
                else:
                    # Use original values for categorical
                    bin_col = feature
                    X[bin_col] = X[bin_col].astype(str)

                # Apply WoE mapping
                woe_col = f'{feature}_woe'
                X[woe_col] = X[bin_col].map(self.woe_mappings[feature])

                # Handle unseen bins with default WoE (0)
                X[woe_col] = X[woe_col].fillna(0)

                # Replace original feature with WoE
                X[feature] = X[woe_col]

                # Clean up temporary columns
                if f'{feature}_bin' in X.columns:
                    X = X.drop(columns=[f'{feature}_bin'])

            except Exception as e:
                print(f"  Error transforming {feature}: {str(e)}")

        return X


def select_features_by_iv(
    iv_dict: Dict[str, float],
    threshold: float = 0.02,
    max_features: int = None,
) -> List[str]:
    """Select features based on Information Value.

    Parameters
    ----------
    iv_dict : Dict[str, float]
        Dictionary of feature -> IV values
    threshold : float
        Minimum IV threshold (default 0.02 for weak predictors)
    max_features : int, optional
        Maximum number of features to select

    Returns
    -------
    List[str]
        Selected feature names, sorted by IV
    """
    # Filter by threshold
    selected = {k: v for k, v in iv_dict.items() if v >= threshold}

    # Sort by IV (descending)
    sorted_features = sorted(selected.items(), key=lambda x: x[1], reverse=True)

    # Extract feature names
    feature_names = [f[0] for f in sorted_features]

    # Limit to max_features if specified
    if max_features and len(feature_names) > max_features:
        feature_names = feature_names[:max_features]

    print(f"Selected {len(feature_names)} features with IV >= {threshold}")
    if max_features:
        print(f"Limited to top {max_features} features")

    return feature_names


def generate_iv_report(
    iv_dict: Dict[str, float],
    output_path: Path = None,
) -> pd.DataFrame:
    """Generate IV report for feature selection and audit.

    Parameters
    ----------
    iv_dict : Dict[str, float]
        Dictionary of feature -> IV values
    output_path : Path, optional
        Path to save CSV report

    Returns
    -------
    pd.DataFrame
        IV report with interpretation
    """
    # Create DataFrame
    iv_df = pd.DataFrame([
        {'feature': k, 'iv': v}
        for k, v in iv_dict.items()
    ]).sort_values('iv', ascending=False)

    # Add interpretation
    def interpret_iv(iv):
        if iv < 0.02:
            return "Not useful"
        elif iv < 0.1:
            return "Weak predictor"
        elif iv < 0.3:
            return "Medium predictor"
        elif iv < 0.5:
            return "Strong predictor"
        else:
            return "Suspicious (potential overfitting)"

    iv_df['interpretation'] = iv_df['iv'].apply(interpret_iv)

    # Save if path provided
    if output_path:
        iv_df.to_csv(output_path, index=False)
        print(f"IV report saved to {output_path}")

    return iv_df


if __name__ == "__main__":
    # Example usage
    print("WoE and IV module loaded successfully")
    print("This module provides:")
    print("  - calculate_woe_iv(): Calculate WoE/IV for single feature")
    print("  - calculate_all_woe_iv(): Calculate WoE/IV for all features")
    print("  - WoETransformer: sklearn-compatible WoE transformer")
    print("  - select_features_by_iv(): Feature selection based on IV")
    print("  - generate_iv_report(): Generate IV report for audit")