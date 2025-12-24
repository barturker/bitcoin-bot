"""
Walk-Forward Validation for Bitcoin Trading Bot

This script performs proper out-of-sample validation using a walk-forward approach.
Each fold trains on historical data and validates on the next unseen period.

Key Features:
- No data leakage: Scaler fit only on training data per fold
- Regime-aware: Reports performance across different market conditions
- Aggregated metrics: Combines results from all folds
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from indicators import load_raw_data, resample_to_timeframe, add_technical_indicators
from indicators import add_multi_timeframe_features, prepare_features, normalize_features
from trading_env import BitcoinTradingEnv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_folds(
    start_year: int,
    end_year: int,
    train_years: int = 2,
    val_years: int = 1
) -> List[Tuple[str, str, str, str]]:
    """
    Create walk-forward folds.

    Args:
        start_year: First year of data
        end_year: Last year of data
        train_years: Years of training data per fold
        val_years: Years of validation data per fold

    Returns:
        List of (train_start, train_end, val_start, val_end) tuples
    """
    folds = []
    current_year = start_year

    while current_year + train_years + val_years <= end_year + 1:
        train_start = f"{current_year}-01-01"
        train_end = f"{current_year + train_years - 1}-12-31"
        val_start = f"{current_year + train_years}-01-01"
        val_end = f"{current_year + train_years + val_years - 1}-12-31"

        folds.append((train_start, train_end, val_start, val_end))
        current_year += 1  # Slide by 1 year

    return folds


def prepare_fold_data(
    raw_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    timeframe: str = "1h"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Prepare data for a single fold with proper scaler handling.
    Scaler is fit ONLY on training data.
    """
    # Resample
    df = resample_to_timeframe(raw_df, timeframe)

    # Split BEFORE indicators to avoid any look-ahead
    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    val_df = df[(df.index >= val_start) & (df.index <= val_end)].copy()

    # Add indicators to each split separately
    train_df = add_technical_indicators(train_df)
    train_df = add_multi_timeframe_features(train_df)
    train_df = train_df.dropna()

    val_df = add_technical_indicators(val_df)
    val_df = add_multi_timeframe_features(val_df)
    val_df = val_df.dropna()

    # Prepare features
    train_features = prepare_features(train_df)
    val_features = prepare_features(val_df)

    # Normalize - fit scaler ONLY on training data
    scaler = StandardScaler()
    train_features_norm = pd.DataFrame(
        scaler.fit_transform(train_features.values),
        index=train_features.index,
        columns=train_features.columns
    )

    # Transform validation with training scaler (no fit!)
    val_features_norm = pd.DataFrame(
        scaler.transform(val_features.values),
        index=val_features.index,
        columns=val_features.columns
    )

    return train_df, train_features_norm, val_df, val_features_norm, scaler


def create_env(df: pd.DataFrame, features: pd.DataFrame, config: dict) -> BitcoinTradingEnv:
    """Create trading environment."""
    env = BitcoinTradingEnv(
        df=df,
        feature_df=features,
        window_size=config['environment']['window_size'],
        initial_capital=config['environment']['initial_capital'],
        max_position_duration=config['environment']['max_position_duration'],
        max_position_pct=config['environment'].get('max_position_pct', 0.20),
        sl_pct=config['trading'].get('sl_pct', 0.02),
        tp_pct=config['trading'].get('tp_pct', 0.04),
        spread_pct=config['trading']['spread_pct'],
        commission_pct=config['trading']['commission_pct'],
        max_trades_per_day=config['trading'].get('max_trades_per_day', 3)
    )
    return env


def train_fold(
    train_env: BitcoinTradingEnv,
    config: dict,
    fold_name: str
) -> PPO:
    """Train model on a single fold."""
    vec_env = DummyVecEnv([lambda: Monitor(train_env)])

    model = PPO(
        policy=config['model']['policy'],
        env=vec_env,
        learning_rate=config['model']['learning_rate'],
        n_steps=config['model']['n_steps'],
        batch_size=config['model']['batch_size'],
        n_epochs=config['model']['n_epochs'],
        gamma=config['model']['gamma'],
        ent_coef=config['model']['ent_coef'],
        clip_range=config['model']['clip_range'],
        verbose=0
    )

    # Reduced timesteps for walk-forward (faster iteration)
    timesteps = config['model'].get('wf_timesteps', 100000)

    print(f"  Training {fold_name} for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    return model


def evaluate_fold(
    model: PPO,
    env: BitcoinTradingEnv,
    name: str
) -> Dict:
    """Evaluate model on validation environment."""
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated

    trades = env.get_trade_history()
    equity_curve = env.get_equity_curve()
    decision_stats = env.get_decision_stats()

    # Calculate metrics
    initial = env.initial_capital
    final = info['equity']
    total_return = (final / initial - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_dd = drawdown.max()

    # Trade metrics
    n_trades = len(trades)
    if n_trades > 0:
        wins = trades[trades['pnl'] > 0]
        win_rate = len(wins) / n_trades * 100
        avg_pnl = trades['pnl'].mean()

        # Lucky wins
        lucky_wins = trades[trades['is_lucky_win'] == True] if 'is_lucky_win' in trades.columns else pd.DataFrame()
        lucky_rate = len(lucky_wins) / n_trades * 100 if n_trades > 0 else 0
    else:
        win_rate = 0
        avg_pnl = 0
        lucky_rate = 0

    return {
        'name': name,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': n_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'lucky_win_rate': lucky_rate,
        'correct_rate': decision_stats.get('correct_rate', 0) * 100,
        'total_reward': total_reward,
        'equity_curve': equity_curve
    }


def run_walk_forward(config_path: str = "config.yaml"):
    """Run complete walk-forward validation."""
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Load config
    config = load_config(config_path)

    # Load raw data once
    print("\nLoading raw data...")
    raw_df = load_raw_data(config['data']['train_path'])
    print(f"Raw data: {raw_df.shape[0]:,} rows")
    print(f"Date range: {raw_df.index.min()} to {raw_df.index.max()}")

    # Create folds
    folds = create_folds(
        start_year=2015,  # Start from 2015 (enough data for indicators)
        end_year=2023,
        train_years=2,
        val_years=1
    )

    print(f"\nCreated {len(folds)} walk-forward folds:")
    for i, (ts, te, vs, ve) in enumerate(folds):
        print(f"  Fold {i+1}: Train [{ts} to {te}] → Val [{vs} to {ve}]")

    # Results storage
    all_results = []
    all_equity_curves = []

    # Create output directory
    os.makedirs("logs/walk_forward", exist_ok=True)

    # Run each fold
    for i, (train_start, train_end, val_start, val_end) in enumerate(folds):
        fold_name = f"Fold_{i+1}"
        print(f"\n{'='*70}")
        print(f"  {fold_name}: Train [{train_start} to {train_end}]")
        print(f"           Val   [{val_start} to {val_end}]")
        print("=" * 70)

        # Prepare data
        print("  Preparing data...")
        train_df, train_features, val_df, val_features, scaler = prepare_fold_data(
            raw_df, train_start, train_end, val_start, val_end,
            timeframe=config['data']['timeframe']
        )
        print(f"  Train samples: {len(train_df):,}")
        print(f"  Val samples: {len(val_df):,}")

        # Create environments
        train_env = create_env(train_df, train_features, config)
        val_env = create_env(val_df, val_features, config)

        # Train
        model = train_fold(train_env, config, fold_name)

        # Evaluate on validation
        print(f"  Evaluating on validation...")
        result = evaluate_fold(model, val_env, fold_name)
        all_results.append(result)
        all_equity_curves.append(result['equity_curve'])

        # Print fold results
        print(f"\n  {fold_name} Validation Results:")
        print(f"    Return: {result['total_return']:.2f}%")
        print(f"    Max DD: {result['max_drawdown']:.2f}%")
        print(f"    Trades: {result['total_trades']}")
        print(f"    Win Rate: {result['win_rate']:.1f}%")
        print(f"    Lucky Win Rate: {result['lucky_win_rate']:.1f}%")
        print(f"    Correct Decision Rate: {result['correct_rate']:.1f}%")

        # Save model
        model.save(f"logs/walk_forward/{fold_name}_model.zip")

    # ============ AGGREGATE RESULTS ============
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)

    print("\nPer-Fold Summary:")
    print(results_df[['name', 'total_return', 'max_drawdown', 'total_trades', 'win_rate', 'lucky_win_rate']].to_string())

    print("\nAggregate Metrics:")
    print(f"  Mean Return: {results_df['total_return'].mean():.2f}%")
    print(f"  Std Return: {results_df['total_return'].std():.2f}%")
    print(f"  Mean Max DD: {results_df['max_drawdown'].mean():.2f}%")
    print(f"  Mean Win Rate: {results_df['win_rate'].mean():.1f}%")
    print(f"  Mean Lucky Win Rate: {results_df['lucky_win_rate'].mean():.1f}%")
    print(f"  Mean Correct Decision Rate: {results_df['correct_rate'].mean():.1f}%")

    # Profitable folds
    profitable = results_df[results_df['total_return'] > 0]
    print(f"\n  Profitable Folds: {len(profitable)}/{len(folds)} ({len(profitable)/len(folds)*100:.0f}%)")

    # ============ VERDICT ============
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Criteria for "production-ready"
    mean_return = results_df['total_return'].mean()
    profitable_rate = len(profitable) / len(folds)
    mean_dd = results_df['max_drawdown'].mean()
    mean_lucky = results_df['lucky_win_rate'].mean()

    issues = []
    if mean_return < 5:
        issues.append(f"Low mean return ({mean_return:.1f}%)")
    if profitable_rate < 0.6:
        issues.append(f"Only {profitable_rate*100:.0f}% profitable folds")
    if mean_dd > 30:
        issues.append(f"High mean drawdown ({mean_dd:.1f}%)")
    if mean_lucky > 20:
        issues.append(f"Too many lucky wins ({mean_lucky:.1f}%)")

    if len(issues) == 0:
        print("  ✓ Model PASSED walk-forward validation")
        print("  → Ready for paper trading")
    else:
        print("  ✗ Model FAILED walk-forward validation")
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
        print("  → Needs more work before paper trading")

    # ============ SAVE RESULTS ============
    results_df.to_csv("logs/walk_forward/results.csv", index=False)
    print(f"\nResults saved to logs/walk_forward/")

    # Plot combined equity curves
    plt.figure(figsize=(14, 6))
    for i, ec in enumerate(all_equity_curves):
        normalized = ec / ec[0] * 100  # Normalize to 100
        plt.plot(normalized, label=f'Fold {i+1}', alpha=0.7)
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.title('Walk-Forward Validation - Equity Curves (Normalized)')
    plt.xlabel('Step')
    plt.ylabel('Equity (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logs/walk_forward/equity_curves.png', dpi=150)
    plt.close()

    print("\nEquity curves plot saved to logs/walk_forward/equity_curves.png")

    return results_df


if __name__ == "__main__":
    try:
        run_walk_forward()
    except Exception as e:
        import traceback
        print("\n" + "=" * 70)
        print("  ERROR OCCURRED!")
        print("=" * 70)
        traceback.print_exc()
    finally:
        print("\nPress Enter to close...")
        input()
