"""
Backtesting script for Bitcoin RL trading bot.
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import joblib

from indicators import load_and_preprocess_data
from trading_env import BitcoinTradingEnv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_metrics(trades_df: pd.DataFrame, equity_curve: np.ndarray,
                      initial_capital: float = 10000) -> dict:
    """Calculate comprehensive performance metrics."""
    metrics = {}

    # Basic metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    metrics['final_equity'] = final_equity
    metrics['total_return_pct'] = total_return
    metrics['total_trades'] = len(trades_df)

    if len(trades_df) == 0:
        return metrics

    # Win/Loss metrics
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    metrics['win_count'] = len(wins)
    metrics['loss_count'] = len(losses)
    metrics['win_rate'] = len(wins) / len(trades_df) * 100

    metrics['total_profit'] = wins['pnl'].sum() if len(wins) > 0 else 0
    metrics['total_loss'] = abs(losses['pnl'].sum()) if len(losses) > 0 else 0

    # Profit factor
    if metrics['total_loss'] > 0:
        metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss']
    else:
        metrics['profit_factor'] = float('inf') if metrics['total_profit'] > 0 else 0

    # Average metrics
    metrics['avg_win'] = wins['pnl'].mean() if len(wins) > 0 else 0
    metrics['avg_loss'] = losses['pnl'].mean() if len(losses) > 0 else 0
    metrics['avg_trade'] = trades_df['pnl'].mean()

    # Risk/Reward ratio
    if metrics['avg_loss'] != 0:
        metrics['avg_rr_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['avg_rr_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0

    # Drawdown metrics
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    metrics['max_drawdown_pct'] = drawdown.max() * 100

    # Find drawdown periods
    in_drawdown = drawdown > 0
    drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
    drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]

    if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
        if drawdown_ends[0] < drawdown_starts[0]:
            drawdown_ends = drawdown_ends[1:]
        if len(drawdown_starts) > len(drawdown_ends):
            drawdown_starts = drawdown_starts[:len(drawdown_ends)]

        if len(drawdown_starts) > 0:
            drawdown_durations = drawdown_ends - drawdown_starts
            metrics['max_drawdown_duration'] = drawdown_durations.max()
            metrics['avg_drawdown_duration'] = drawdown_durations.mean()
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['avg_drawdown_duration'] = 0
    else:
        metrics['max_drawdown_duration'] = 0
        metrics['avg_drawdown_duration'] = 0

    # Returns for Sharpe/Sortino
    returns = np.diff(equity_curve) / equity_curve[:-1]

    # Sharpe Ratio (assuming hourly data, annualized)
    if len(returns) > 1 and returns.std() > 0:
        # Assuming hourly data: 24 hours * 365 days = 8760 periods per year
        periods_per_year = 8760
        metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    else:
        metrics['sharpe_ratio'] = 0

    # Sortino Ratio (only downside volatility)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 1 and negative_returns.std() > 0:
        metrics['sortino_ratio'] = (returns.mean() / negative_returns.std()) * np.sqrt(8760)
    else:
        metrics['sortino_ratio'] = 0

    # Calmar Ratio
    if metrics['max_drawdown_pct'] > 0:
        # Annualized return / max drawdown
        n_periods = len(equity_curve)
        annualized_return = ((final_equity / initial_capital) ** (8760 / n_periods) - 1) * 100
        metrics['calmar_ratio'] = annualized_return / metrics['max_drawdown_pct']
    else:
        metrics['calmar_ratio'] = float('inf') if total_return > 0 else 0

    # Trade duration
    metrics['avg_trade_duration'] = trades_df['duration'].mean()
    metrics['max_trade_duration'] = trades_df['duration'].max()

    # Trade reasons
    metrics['sl_hits'] = len(trades_df[trades_df['reason'] == 'stop_loss'])
    metrics['tp_hits'] = len(trades_df[trades_df['reason'] == 'take_profit'])
    metrics['manual_closes'] = len(trades_df[trades_df['reason'] == 'manual'])

    return metrics


def print_metrics(metrics: dict):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\n--- Performance ---")
    print(f"Final Equity:      ${metrics['final_equity']:,.2f}")
    print(f"Total Return:      {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):.2f}")
    print(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):.2f}")

    print("\n--- Risk ---")
    print(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Max DD Duration:   {metrics.get('max_drawdown_duration', 0):.0f} periods")

    print("\n--- Trading ---")
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Win Rate:          {metrics.get('win_rate', 0):.1f}%")
    print(f"Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
    print(f"Avg R:R Ratio:     {metrics.get('avg_rr_ratio', 0):.2f}")

    print("\n--- Trade Details ---")
    print(f"Avg Win:           ${metrics.get('avg_win', 0):.2f}")
    print(f"Avg Loss:          ${metrics.get('avg_loss', 0):.2f}")
    print(f"Avg Trade:         ${metrics.get('avg_trade', 0):.2f}")
    print(f"Avg Duration:      {metrics.get('avg_trade_duration', 0):.1f} periods")

    print("\n--- Exit Reasons ---")
    print(f"Stop Loss:         {metrics.get('sl_hits', 0)}")
    print(f"Take Profit:       {metrics.get('tp_hits', 0)}")
    print(f"Manual Close:      {metrics.get('manual_closes', 0)}")

    print("=" * 60)


def plot_results(equity_curve: np.ndarray, trades_df: pd.DataFrame,
                 save_path: str = "logs/backtest_results.png"):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Equity curve
    ax1 = axes[0, 0]
    ax1.plot(equity_curve, linewidth=1)
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[0, 1]
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.7, color='red')
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # Trade PnL distribution
    ax3 = axes[1, 0]
    if len(trades_df) > 0:
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
        ax3.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Trade PnL')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('PnL ($)')
    ax3.grid(True, alpha=0.3)

    # Cumulative PnL
    ax4 = axes[1, 1]
    if len(trades_df) > 0:
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax4.plot(cumulative_pnl, marker='o', markersize=3)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Cumulative Trade PnL')
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('Cumulative PnL ($)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nResults saved to: {save_path}")


def backtest(model_path: str, config_path: str = "config.yaml"):
    """Run backtest on test data."""
    # Load config
    config = load_config(config_path)

    # Load scaler
    scaler = None
    if os.path.exists("cache/scaler.pkl"):
        scaler = joblib.load("cache/scaler.pkl")
        print("Loaded scaler from cache.")

    # Load test data
    print("\nLoading test data...")
    test_df, test_features, _ = load_and_preprocess_data(
        csv_path=config['data']['train_path'],
        timeframe=config['data']['timeframe'],
        start_date=config['data']['test_start'],
        end_date=config['data']['test_end'],
        normalize=config['features']['normalize'],
        scaler=scaler
    )
    print(f"Test data: {test_df.shape[0]} samples")
    print(f"Date range: {test_df.index.min()} to {test_df.index.max()}")

    # Create test environment
    print("\nCreating test environment...")
    test_env = BitcoinTradingEnv(
        df=test_df,
        feature_df=test_features,
        window_size=config['environment']['window_size'],
        initial_capital=config['environment']['initial_capital'],
        max_position_duration=config['environment']['max_position_duration'],
        sl_options=config['trading']['sl_options'],
        tp_options=config['trading']['tp_options'],
        spread_pct=config['trading']['spread_pct'],
        commission_pct=config['trading']['commission_pct']
    )

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)

    # Run backtest
    print("\nRunning backtest...")
    obs, info = test_env.reset()
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        done = done or truncated
        step += 1

        if step % 1000 == 0:
            print(f"  Step {step}, Equity: ${info['equity']:.2f}")

    # Get results
    trades_df = test_env.get_trade_history()
    equity_curve = test_env.get_equity_curve()

    # Calculate metrics
    metrics = calculate_metrics(
        trades_df,
        equity_curve,
        config['environment']['initial_capital']
    )

    # Print results
    print_metrics(metrics)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trades
    if len(trades_df) > 0:
        trades_path = f"logs/trades/backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"\nTrades saved to: {trades_path}")

    # Plot results
    plot_results(equity_curve, trades_df, f"logs/backtest_results_{timestamp}.png")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f"logs/backtest_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    return metrics, trades_df, equity_curve


def compare_with_benchmark(test_df: pd.DataFrame, equity_curve: np.ndarray,
                           initial_capital: float = 10000):
    """Compare with buy-and-hold benchmark."""
    # Buy and hold
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return = (last_price / first_price - 1) * 100

    # Model return
    model_return = (equity_curve[-1] / initial_capital - 1) * 100

    print("\n--- Benchmark Comparison ---")
    print(f"Buy & Hold Return:  {bh_return:.2f}%")
    print(f"Model Return:       {model_return:.2f}%")
    print(f"Outperformance:     {model_return - bh_return:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Bitcoin RL Trading Bot")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    backtest(args.model, args.config)
