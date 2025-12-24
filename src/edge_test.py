"""
Edge Test: Does this feature space contain predictable alpha?

This script tests if there's ANY edge in our features using simple classifiers.
If XGBoost can't find edge → problem is feature space, not RL.

Target: "Will price increase by 2%+ in next 24-72 hours?"
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from indicators import (
    load_raw_data,
    resample_to_timeframe,
    add_base_indicators,
    add_multi_timeframe_features,
    add_market_regime_flags,
    normalize_features
)


def create_target(df: pd.DataFrame, horizon: int = 24, threshold: float = 0.02) -> pd.Series:
    """
    Create binary target: Will price increase by threshold% in next horizon bars?

    Args:
        df: DataFrame with 'close' column
        horizon: Look-ahead period (hours for 1H data)
        threshold: Minimum return to be considered "good trade"

    Returns:
        Binary series: 1 if good long opportunity, 0 otherwise
    """
    # Future max price in horizon
    future_max = df['close'].rolling(window=horizon, min_periods=1).max().shift(-horizon)

    # Potential return
    potential_return = (future_max - df['close']) / df['close']

    # Binary target
    target = (potential_return >= threshold).astype(int)

    return target


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare feature matrix from raw data."""

    # Add indicators
    df = add_base_indicators(df)
    df = add_multi_timeframe_features(df)
    df = add_market_regime_flags(df)

    # Get feature columns (exclude OHLCV and target-like columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return df, feature_cols


def walk_forward_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target',
    train_years: int = 2,
    test_years: int = 1
) -> List[Dict]:
    """
    Walk-forward validation for edge test.

    Same fold structure as RL walk-forward.
    """
    results = []

    # Get year range
    df['year'] = df.index.year
    years = sorted(df['year'].unique())

    # Create folds
    folds = []
    for i in range(len(years) - train_years):
        train_start = years[i]
        train_end = years[i + train_years - 1]
        test_year = years[i + train_years]

        if test_year <= years[-1]:
            folds.append({
                'name': f'Fold_{i+1}',
                'train_years': list(range(train_start, train_end + 1)),
                'test_year': test_year
            })

    print(f"\nRunning {len(folds)} folds...")
    print("=" * 60)

    for fold in folds:
        print(f"\n{fold['name']}: Train {fold['train_years']} → Test {fold['test_year']}")

        # Split data
        train_mask = df['year'].isin(fold['train_years'])
        test_mask = df['year'] == fold['test_year']

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # Drop rows with NaN
        train_df = train_df.dropna(subset=feature_cols + [target_col])
        test_df = test_df.dropna(subset=feature_cols + [target_col])

        if len(train_df) < 100 or len(test_df) < 100:
            print(f"  Skipping: insufficient data")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle inf/nan
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)

        # Class balance
        pos_rate_train = y_train.mean()
        pos_rate_test = y_test.mean()

        print(f"  Train: {len(train_df):,} samples, {pos_rate_train:.1%} positive")
        print(f"  Test:  {len(test_df):,} samples, {pos_rate_test:.1%} positive")

        # ============ MODELS ============
        fold_result = {
            'name': fold['name'],
            'test_year': fold['test_year'],
            'n_train': len(train_df),
            'n_test': len(test_df),
            'baseline_acc': max(pos_rate_test, 1 - pos_rate_test),
        }

        # 1. Logistic Regression
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)
            y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

            fold_result['lr_acc'] = accuracy_score(y_test, y_pred_lr)
            fold_result['lr_auc'] = roc_auc_score(y_test, y_prob_lr)
        except Exception as e:
            fold_result['lr_acc'] = 0.5
            fold_result['lr_auc'] = 0.5

        # 2. Gradient Boosting (XGBoost-like)
        try:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            y_pred_gb = gb.predict(X_test_scaled)
            y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]

            fold_result['gb_acc'] = accuracy_score(y_test, y_pred_gb)
            fold_result['gb_auc'] = roc_auc_score(y_test, y_prob_gb)
        except Exception as e:
            fold_result['gb_acc'] = 0.5
            fold_result['gb_auc'] = 0.5

        # Print results
        print(f"\n  Results:")
        print(f"    Baseline (majority class): {fold_result['baseline_acc']:.1%}")
        print(f"    Logistic:  Acc={fold_result['lr_acc']:.1%}, AUC={fold_result['lr_auc']:.3f}")
        print(f"    GBoosting: Acc={fold_result['gb_acc']:.1%}, AUC={fold_result['gb_auc']:.3f}")

        # Edge detection
        gb_lift = fold_result['gb_auc'] - 0.5
        if gb_lift > 0.05:
            print(f"    → POTENTIAL EDGE DETECTED (+{gb_lift:.1%} over random)")
        elif gb_lift > 0.02:
            print(f"    → Weak signal (+{gb_lift:.1%})")
        else:
            print(f"    → NO EDGE (AUC ≈ random)")

        results.append(fold_result)

    return results


def main():
    """Run edge test."""
    print("=" * 60)
    print("  EDGE TEST: Does this feature space contain alpha?")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    raw_df = load_raw_data("data/btcusd_1-min_data.csv")
    print(f"Raw data: {len(raw_df):,} rows")

    # Resample to 1H
    print("Resampling to 1H...")
    df = resample_to_timeframe(raw_df, '1h')
    print(f"1H data: {len(df):,} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Prepare features
    print("\nPreparing features...")
    df, feature_cols = prepare_features(df)
    print(f"Features: {len(feature_cols)} columns")

    # Create target
    print("\nCreating target (2% gain in 24h)...")
    df['target'] = create_target(df, horizon=24, threshold=0.02)

    # Drop rows with NaN target
    df = df.dropna(subset=['target'])
    print(f"Samples with target: {len(df):,}")
    print(f"Positive rate: {df['target'].mean():.1%}")

    # Run walk-forward
    results = walk_forward_test(df, feature_cols)

    # ============ AGGREGATE RESULTS ============
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    print("\nPer-Fold Summary:")
    print(results_df[['name', 'test_year', 'baseline_acc', 'lr_acc', 'lr_auc', 'gb_acc', 'gb_auc']].to_string())

    print("\nAggregate Metrics:")
    print(f"  Mean Baseline Accuracy: {results_df['baseline_acc'].mean():.1%}")
    print(f"  Mean Logistic Accuracy: {results_df['lr_acc'].mean():.1%}")
    print(f"  Mean Logistic AUC:      {results_df['lr_auc'].mean():.3f}")
    print(f"  Mean GBoost Accuracy:   {results_df['gb_acc'].mean():.1%}")
    print(f"  Mean GBoost AUC:        {results_df['gb_auc'].mean():.3f}")

    # Verdict
    mean_gb_auc = results_df['gb_auc'].mean()
    auc_lift = mean_gb_auc - 0.5

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    if auc_lift > 0.05:
        print(f"  ✓ EDGE EXISTS in this feature space")
        print(f"    GBoost AUC = {mean_gb_auc:.3f} (+{auc_lift:.1%} over random)")
        print(f"    → Problem was RL, not features")
        print(f"    → Consider: simpler model, or better RL tuning")
    elif auc_lift > 0.02:
        print(f"  ~ WEAK EDGE detected")
        print(f"    GBoost AUC = {mean_gb_auc:.3f} (+{auc_lift:.1%} over random)")
        print(f"    → Edge exists but very thin")
        print(f"    → May not survive transaction costs")
    else:
        print(f"  ✗ NO EDGE in this feature space")
        print(f"    GBoost AUC = {mean_gb_auc:.3f} (≈ random)")
        print(f"    → Problem is FEATURES, not model")
        print(f"    → Need different indicators / approach")

    # Save results
    output_dir = Path("logs/edge_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "edge_test_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/")

    return results_df


if __name__ == "__main__":
    results = main()
    input("\nPress Enter to close...")
