"""
Simple Trader: No RL, just Logistic Regression + Risk Management

Based on edge_test.py results:
- Logistic Regression AUC = 0.791 (strong edge!)
- Features contain predictable alpha
- Problem was RL, not features

Strategy:
1. Train Logistic Regression on "Will price gain 2%+ in 24h?"
2. Trade when confidence > threshold
3. Apply hard veto rules (bear/shock/conflict/chop)
4. Fixed SL/TP (2%/4%)
5. Walk-forward validation
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from indicators import (
    load_raw_data,
    resample_to_timeframe,
    add_technical_indicators,
    add_multi_timeframe_features,
    add_market_regime_flags,
)


@dataclass
class Trade:
    """Single trade record."""
    entry_idx: int
    entry_price: float
    direction: str  # 'long' or 'short'
    confidence: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class SimpleTradingSystem:
    """
    Simple trading system using Logistic Regression.

    No RL, no complex rewards - just:
    1. Predict opportunity
    2. Check veto conditions
    3. Enter with fixed SL/TP
    4. Exit mechanically
    """

    # Hard veto flags - same as V2.3
    VETO_FLAGS = [
        'regime_bear_trend',
        'regime_shock',
        'regime_trend_conflict',
        'regime_chop',
        'regime_no_trade_zone',
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
        max_duration: int = 72,
        min_cooldown: int = 24,
        spread_pct: float = 0.001,
        commission_pct: float = 0.001,
    ):
        self.confidence_threshold = confidence_threshold
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.max_duration = max_duration
        self.min_cooldown = min_cooldown
        self.spread_pct = spread_pct
        self.commission_pct = commission_pct

        self.model = None
        self.scaler = None
        self.feature_cols = None

    def _is_vetoed(self, row: pd.Series) -> Tuple[bool, str]:
        """Check if current conditions are vetoed."""
        for flag in self.VETO_FLAGS:
            if flag in row.index and row[flag] > 0.5:
                return True, flag.replace('regime_', '')
        return False, ''

    def _get_htf_bias(self, row: pd.Series) -> int:
        """Get HTF bias: 1 (long), -1 (short), 0 (none)."""
        mtf_align = row.get('mtf_trend_alignment', 0)
        daily_trend = row.get('htf_1d_trend', 0)
        h4_trend = row.get('htf_4h_trend', 0)
        conflict = row.get('trend_conflict', 0)

        if conflict > 0.5:
            return 0

        if mtf_align > 0.3 and daily_trend > 0 and h4_trend > 0:
            return 1  # Long
        elif mtf_align < -0.3 and daily_trend < 0 and h4_trend < 0:
            return -1  # Short

        return 0

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = 'target'):
        """Train the model."""
        # Store feature columns
        self.feature_cols = feature_cols

        # Prepare data
        train_df = df.dropna(subset=feature_cols + [target_col])
        X = train_df[feature_cols].values
        y = train_df[target_col].values

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

        # Train Logistic Regression
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of good opportunity."""
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.20,
    ) -> Dict:
        """
        Run backtest on data.

        Returns dict with performance metrics.
        """
        # Predict probabilities
        probs = self.predict_proba(df)
        df = df.copy()
        df['prob'] = probs

        # State
        capital = initial_capital
        equity_curve = [initial_capital]
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        last_trade_idx = -999
        vetoes = 0

        # Simulate
        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']

            # Update equity
            if current_trade is not None:
                if current_trade.direction == 'long':
                    unrealized = (price - current_trade.entry_price) / current_trade.entry_price
                else:
                    unrealized = (current_trade.entry_price - price) / current_trade.entry_price
                equity = capital + (capital * position_size_pct * unrealized)
            else:
                equity = capital
            equity_curve.append(equity)

            # Check existing position
            if current_trade is not None:
                duration = i - current_trade.entry_idx

                # Check SL/TP
                if current_trade.direction == 'long':
                    sl_price = current_trade.entry_price * (1 - self.sl_pct)
                    tp_price = current_trade.entry_price * (1 + self.tp_pct)
                    if row['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                    elif duration >= self.max_duration:
                        exit_price = price
                        exit_reason = 'max_duration'
                    else:
                        continue
                else:  # short
                    sl_price = current_trade.entry_price * (1 + self.sl_pct)
                    tp_price = current_trade.entry_price * (1 - self.tp_pct)
                    if row['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                    elif duration >= self.max_duration:
                        exit_price = price
                        exit_reason = 'max_duration'
                    else:
                        continue

                # Close trade
                if current_trade.direction == 'long':
                    pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
                else:
                    pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price

                # Apply costs
                pnl_pct -= self.spread_pct + self.commission_pct * 2

                position_value = capital * position_size_pct
                pnl = position_value * pnl_pct
                capital += pnl

                current_trade.exit_idx = i
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl_pct * 100
                trades.append(current_trade)
                current_trade = None
                last_trade_idx = i

            else:
                # Check for new entry
                if i - last_trade_idx < self.min_cooldown:
                    continue

                # Check veto
                is_vetoed, veto_reason = self._is_vetoed(row)
                if is_vetoed:
                    vetoes += 1
                    continue

                # Check confidence
                prob = row['prob']
                if prob < self.confidence_threshold:
                    continue

                # Check HTF bias
                bias = self._get_htf_bias(row)
                if bias == 0:
                    continue

                # Enter trade
                direction = 'long' if bias > 0 else 'short'
                entry_price = price * (1 + self.spread_pct / 2) if direction == 'long' else price * (1 - self.spread_pct / 2)

                current_trade = Trade(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction=direction,
                    confidence=prob,
                )

        # Close any open trade at end
        if current_trade is not None:
            price = df.iloc[-1]['close']
            if current_trade.direction == 'long':
                pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price
            else:
                pnl_pct = (current_trade.entry_price - price) / current_trade.entry_price
            pnl_pct -= self.spread_pct + self.commission_pct * 2
            position_value = capital * position_size_pct
            pnl = position_value * pnl_pct
            capital += pnl
            current_trade.exit_idx = len(df) - 1
            current_trade.exit_price = price
            current_trade.exit_reason = 'end'
            current_trade.pnl = pnl
            current_trade.pnl_pct = pnl_pct * 100
            trades.append(current_trade)

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        total_return = (capital / initial_capital - 1) * 100

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_dd = drawdown.max()

        # Trade stats
        n_trades = len(trades)
        if n_trades > 0:
            wins = [t for t in trades if t.pnl > 0]
            win_rate = len(wins) / n_trades * 100
            avg_pnl = np.mean([t.pnl for t in trades])
            avg_pnl_pct = np.mean([t.pnl_pct for t in trades])
        else:
            win_rate = 0
            avg_pnl = 0
            avg_pnl_pct = 0

        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'total_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'vetoes': vetoes,
            'final_capital': capital,
            'equity_curve': equity_curve,
            'trades': trades,
        }


def create_target(df: pd.DataFrame, horizon: int = 24, threshold: float = 0.02) -> pd.Series:
    """Create binary target for training."""
    future_max = df['close'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
    potential_return = (future_max - df['close']) / df['close']
    return (potential_return >= threshold).astype(int)


def prepare_data(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare all data."""
    print("Loading data...")
    raw_df = load_raw_data(csv_path)

    print("Resampling to 1H...")
    df = resample_to_timeframe(raw_df, '1h')

    print("Adding indicators...")
    df = add_technical_indicators(df)
    df = add_multi_timeframe_features(df)
    df = add_market_regime_flags(df)

    # Feature columns
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'year', 'prob']
    feature_cols = [c for c in df.columns if c not in exclude]

    # Add target
    df['target'] = create_target(df, horizon=24, threshold=0.02)
    df['year'] = df.index.year

    return df, feature_cols


def run_walk_forward(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """Run walk-forward validation."""
    results = []

    years = sorted(df['year'].unique())
    train_years = 2

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
    print("=" * 70)

    all_equity_curves = []

    for fold in folds:
        print(f"\n{fold['name']}: Train {fold['train_years']} → Test {fold['test_year']}")

        # Split
        train_mask = df['year'].isin(fold['train_years'])
        test_mask = df['year'] == fold['test_year']

        train_df = df[train_mask].dropna(subset=feature_cols + ['target'])
        test_df = df[test_mask].dropna(subset=feature_cols + ['target'])

        if len(train_df) < 500 or len(test_df) < 100:
            print(f"  Skipping: insufficient data")
            continue

        print(f"  Train: {len(train_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")

        # Train
        system = SimpleTradingSystem(
            confidence_threshold=0.60,
            sl_pct=0.02,
            tp_pct=0.04,
        )
        system.fit(train_df, feature_cols)

        # Backtest
        result = system.backtest(test_df)
        result['name'] = fold['name']
        result['test_year'] = fold['test_year']

        print(f"\n  Results:")
        print(f"    Return: {result['total_return']:.2f}%")
        print(f"    Max DD: {result['max_drawdown']:.2f}%")
        print(f"    Trades: {result['total_trades']} | Vetoes: {result['vetoes']}")
        print(f"    Win Rate: {result['win_rate']:.1f}%")

        results.append(result)
        all_equity_curves.append(result['equity_curve'])

    return results


def main():
    """Run simple trader walk-forward validation."""
    print("=" * 70)
    print("  SIMPLE TRADER - Walk-Forward Validation")
    print("  (No RL, just Logistic Regression + Risk Management)")
    print("=" * 70)

    # Prepare data
    df, feature_cols = prepare_data("data/btcusd_1-min_data.csv")
    print(f"\nData: {len(df):,} samples, {len(feature_cols)} features")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")

    # Run walk-forward
    results = run_walk_forward(df, feature_cols)

    # Aggregate
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame([{
        'name': r['name'],
        'test_year': r['test_year'],
        'total_return': r['total_return'],
        'max_drawdown': r['max_drawdown'],
        'total_trades': r['total_trades'],
        'vetoes': r['vetoes'],
        'win_rate': r['win_rate'],
    } for r in results])

    print("\nPer-Fold Summary:")
    print(results_df.to_string())

    print("\nAggregate Metrics:")
    print(f"  Mean Return: {results_df['total_return'].mean():.2f}%")
    print(f"  Std Return: {results_df['total_return'].std():.2f}%")
    print(f"  Mean Max DD: {results_df['max_drawdown'].mean():.2f}%")
    print(f"  Mean Win Rate: {results_df['win_rate'].mean():.1f}%")
    print(f"  Total Trades: {results_df['total_trades'].sum()}")
    print(f"  Total Vetoes: {results_df['vetoes'].sum()}")

    # Profitable folds
    profitable = results_df[results_df['total_return'] > 0]
    profitable_pct = len(profitable) / len(results_df) * 100
    print(f"\n  Profitable Folds: {len(profitable)}/{len(results_df)} ({profitable_pct:.0f}%)")

    # Verdict
    mean_return = results_df['total_return'].mean()

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    if mean_return > 5 and profitable_pct >= 60:
        print("  ✓ PASS - Simple trader shows consistent edge")
        print("  → Ready for paper trading")
    elif mean_return > 0 and profitable_pct >= 50:
        print("  ~ MARGINAL - Edge exists but thin")
        print("  → Need optimization or higher threshold")
    else:
        print("  ✗ FAIL - No consistent edge")
        print(f"    Mean return: {mean_return:.2f}%")
        print(f"    Profitable: {profitable_pct:.0f}%")

    # Save
    output_dir = Path("logs/simple_trader")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "walk_forward_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/")

    return results_df


if __name__ == "__main__":
    results = main()
    input("\nPress Enter to close...")
