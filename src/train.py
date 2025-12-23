"""
Training script for Bitcoin RL trading bot.
"""

import os
import sys
import yaml
import numpy as np
from datetime import datetime
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from indicators import load_and_preprocess_data
from trading_env import BitcoinTradingEnv


class RealTimeProgressCallback(BaseCallback):
    """
    Real-time progress bar with live metrics.
    Writes progress to JSON file for dashboard integration.
    """
    def __init__(self, total_timesteps: int, update_freq: int = 100,
                 progress_file: str = "logs/training_progress.json", verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.progress_file = progress_file
        self.start_time = None
        self.last_update = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_training_start(self):
        self.start_time = time.time()
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        self._save_progress(status="running")
        self._print_header()

    def _print_header(self):
        print("\n" + "=" * 70)
        print("  TRAINING PROGRESS")
        print("=" * 70)

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours:.0f}h {mins:.0f}m"

    def _save_progress(self, status="running"):
        """Save progress to JSON file for dashboard."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = self.num_timesteps / self.total_timesteps if self.total_timesteps > 0 else 0

        if progress > 0:
            eta = elapsed / progress - elapsed
        else:
            eta = 0

        avg_reward = float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0
        avg_length = float(np.mean(self.episode_lengths[-10:])) if self.episode_lengths else 0

        data = {
            "status": status,
            "current_step": self.num_timesteps,
            "total_steps": self.total_timesteps,
            "progress": progress,
            "percent": progress * 100,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": self._format_time(elapsed),
            "eta_seconds": eta,
            "eta_formatted": self._format_time(eta),
            "episodes": len(self.episode_rewards),
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "steps_per_second": self.num_timesteps / elapsed if elapsed > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            import json
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Don't fail training if file write fails

    def _on_step(self) -> bool:
        # Track episode stats
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1

        # Check for episode end
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Update display and file
        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps
            self._update_progress()
            self._save_progress(status="running")

        return True

    def _update_progress(self):
        # Calculate progress
        progress = self.num_timesteps / self.total_timesteps
        percent = progress * 100

        # Time calculations
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
        else:
            eta = 0

        # Progress bar (40 chars wide)
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Get recent metrics
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        episodes = len(self.episode_rewards)

        # Steps per second
        sps = self.num_timesteps / elapsed if elapsed > 0 else 0

        # Clear line and print progress
        sys.stdout.write("\r" + " " * 80 + "\r")
        progress_line = f"  [{bar}] {percent:5.1f}%"
        sys.stdout.write(progress_line)
        sys.stdout.flush()

        # Print detailed stats every 1000 steps
        if self.num_timesteps % 1000 < self.update_freq:
            print()
            print(f"  Step: {self.num_timesteps:,} / {self.total_timesteps:,} | "
                  f"Episodes: {episodes} | "
                  f"Speed: {sps:.0f} steps/s")
            print(f"  Elapsed: {self._format_time(elapsed)} | "
                  f"ETA: {self._format_time(eta)} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Avg Length: {avg_length:.0f}")
            print("-" * 70)

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        self._save_progress(status="completed")

        print("\n" + "=" * 70)
        print(f"  TRAINING COMPLETE!")
        print(f"  Total time: {self._format_time(elapsed)}")
        print(f"  Total episodes: {len(self.episode_rewards)}")
        if self.episode_rewards:
            print(f"  Final avg reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
        print("=" * 70 + "\n")


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
        max_position_pct=config['environment'].get('max_position_pct', 0.20),
        simple_actions=config['environment'].get('simple_actions', True),
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

    # Real-time progress callback
    progress_callback = RealTimeProgressCallback(
        total_timesteps=config['model']['total_timesteps'],
        update_freq=50  # Update every 50 steps for smooth progress
    )

    # Train
    print("\nStarting training...")

    model.learn(
        total_timesteps=config['model']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=False  # Use our custom progress bar instead
    )

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
