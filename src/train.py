"""
Training script for Bitcoin RL trading bot.
"""

import os
import yaml
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from indicators import load_and_preprocess_data
from trading_env import BitcoinTradingEnv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(df, features, config):
    """Create trading environment from config."""
    env = BitcoinTradingEnv(
        df=df,
        feature_df=features,
        window_size=config['environment']['window_size'],
        initial_capital=config['environment']['initial_capital'],
        max_position_duration=config['environment']['max_position_duration'],
        sl_options=config['trading']['sl_options'],
        tp_options=config['trading']['tp_options'],
        spread_pct=config['trading']['spread_pct'],
        commission_pct=config['trading']['commission_pct']
    )
    return env


def train(config_path: str = "config.yaml"):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print("Configuration loaded.")

    # Create directories
    os.makedirs(config['logging']['tensorboard_log'], exist_ok=True)
    os.makedirs(config['logging']['model_save_path'], exist_ok=True)
    os.makedirs(config['logging']['trade_log_path'], exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # Load and preprocess data
    print("\nLoading training data...")
    train_df, train_features, scaler = load_and_preprocess_data(
        csv_path=config['data']['train_path'],
        timeframe=config['data']['timeframe'],
        start_date=config['data']['train_start'],
        end_date=config['data']['train_end'],
        normalize=config['features']['normalize'],
        scaler_path="cache/scaler.pkl"
    )
    print(f"Training data: {train_df.shape[0]} samples")
    print(f"Date range: {train_df.index.min()} to {train_df.index.max()}")

    # Create training environment
    print("\nCreating training environment...")
    train_env = create_env(train_df, train_features, config)
    train_env = Monitor(train_env)
    vec_env = DummyVecEnv([lambda: train_env])

    # Create validation environment (last 20% of training data)
    val_split = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_split:].reset_index(drop=True)
    val_features = train_features.iloc[val_split:].reset_index(drop=True)

    val_env = create_env(val_df, val_features, config)
    val_env = Monitor(val_env)
    val_vec_env = DummyVecEnv([lambda: val_env])

    # Create model
    print("\nCreating PPO model...")
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
        verbose=1,
        tensorboard_log=config['logging']['tensorboard_log']
    )

    print(f"Model policy: {config['model']['policy']}")
    print(f"Total timesteps: {config['model']['total_timesteps']}")

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_callback = EvalCallback(
        val_vec_env,
        best_model_save_path=config['logging']['model_save_path'],
        log_path=config['logging']['trade_log_path'],
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config['logging']['model_save_path'],
        name_prefix=f"ppo_btc_{timestamp}"
    )

    # Train
    print("\nStarting training...")
    print("=" * 50)

    model.learn(
        total_timesteps=config['model']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    print("=" * 50)
    print("Training completed!")

    # Save final model
    final_model_path = os.path.join(
        config['logging']['model_save_path'],
        f"ppo_btc_final_{timestamp}.zip"
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Evaluate on training data
    print("\nEvaluating on training data...")
    evaluate_model(model, train_env, "Training")

    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    evaluate_model(model, val_env, "Validation")

    return model, train_env


def evaluate_model(model, env, name: str = ""):
    """Evaluate model on environment."""
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated

    # Get results
    trades = env.get_trade_history()
    equity_curve = env.get_equity_curve()

    print(f"\n{name} Results:")
    print(f"  Final Equity: ${info['equity']:.2f}")
    print(f"  Total Return: {((info['equity'] / 10000) - 1) * 100:.2f}%")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Total Reward: {total_reward:.2f}")

    if len(trades) > 0:
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100

        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Win: ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "  Avg Win: N/A")
        print(f"  Avg Loss: ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "  Avg Loss: N/A")

        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_dd = drawdown.max()
        print(f"  Max Drawdown: {max_dd:.2f}%")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f'{name} Equity Curve')
    plt.xlabel('Step')
    plt.ylabel('Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'logs/{name.lower()}_equity_curve.png', dpi=150)
    plt.close()
    print(f"  Equity curve saved to: logs/{name.lower()}_equity_curve.png")


if __name__ == "__main__":
    train()
