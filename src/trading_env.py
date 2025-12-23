"""
Bitcoin Trading Environment for Reinforcement Learning.
Based on OpenAI Gymnasium interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class BitcoinTradingEnv(gym.Env):
    """
    A Bitcoin trading environment for reinforcement learning.

    Action Space:
        0: Hold (do nothing)
        1-9: Open Long with SL/TP combinations
        10-18: Open Short with SL/TP combinations
        19: Close position

    Observation Space:
        Window of historical features (window_size x num_features)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        window_size: int = 48,
        initial_capital: float = 10000.0,
        max_position_duration: int = 72,
        sl_options: list = None,
        tp_options: list = None,
        spread_pct: float = 0.001,
        commission_pct: float = 0.001,
        max_position_pct: float = 0.20,  # Max %20 risk per trade
        simple_actions: bool = True,  # Use simplified 5-action space
        render_mode: str = None
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_df = feature_df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_position_duration = max_position_duration
        self.spread_pct = spread_pct
        self.commission_pct = commission_pct
        self.max_position_pct = max_position_pct  # Max position size as % of capital
        self.simple_actions = simple_actions
        self.render_mode = render_mode

        # SL/TP options as percentages
        self.sl_options = sl_options or [0.01, 0.02, 0.03]
        self.tp_options = tp_options or [0.02, 0.04, 0.06]

        # Default SL/TP for simple mode (middle values)
        self.default_sl = self.sl_options[len(self.sl_options) // 2]
        self.default_tp = self.tp_options[len(self.tp_options) // 2]

        if self.simple_actions:
            # Simple mode: 5 actions
            # 0: Hold, 1: Long, 2: Short, 3: Close, 4: Hold (duplicate for symmetry)
            self.n_actions = 5
        else:
            # Complex mode: Full SL/TP combinations
            n_sl = len(self.sl_options)
            n_tp = len(self.tp_options)
            self.n_long_actions = n_sl * n_tp
            self.n_short_actions = n_sl * n_tp
            self.n_actions = 1 + self.n_long_actions + self.n_short_actions + 1

        # Action space
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: window of features + position info
        n_features = feature_df.shape[1]
        # Additional features: position_type, unrealized_pnl, position_duration
        self.n_position_features = 3
        total_features = n_features + self.n_position_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, total_features),
            dtype=np.float32
        )

        # Build action map
        self._build_action_map()

        # Initialize state
        self.reset()

    def _build_action_map(self):
        """Build mapping from action index to (direction, sl, tp)."""
        if self.simple_actions:
            # Simple 5-action mode
            self.action_map = {
                0: ('hold', None, None),
                1: ('long', self.default_sl, self.default_tp),
                2: ('short', self.default_sl, self.default_tp),
                3: ('close', None, None),
                4: ('hold', None, None),  # Extra hold for balanced action space
            }
        else:
            # Complex mode with all SL/TP combinations
            self.action_map = {0: ('hold', None, None)}

            action_idx = 1
            # Long positions
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map[action_idx] = ('long', sl, tp)
                    action_idx += 1

            # Short positions
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map[action_idx] = ('short', sl, tp)
                    action_idx += 1

            # Close position
            self.action_map[action_idx] = ('close', None, None)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.equity = self.initial_capital

        # Position state
        self.position = None  # 'long', 'short', or None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_duration = 0

        # Tracking
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.total_pnl = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step

        # Get feature window
        feature_window = self.feature_df.iloc[start_idx:end_idx].values

        # Pad if necessary
        if len(feature_window) < self.window_size:
            padding = np.tile(feature_window[0], (self.window_size - len(feature_window), 1))
            feature_window = np.vstack([padding, feature_window])

        # Add position features
        position_features = np.zeros((self.window_size, self.n_position_features))

        # Position type: -1 (short), 0 (none), 1 (long)
        if self.position == 'long':
            position_type = 1.0
        elif self.position == 'short':
            position_type = -1.0
        else:
            position_type = 0.0

        # Unrealized PnL (normalized)
        unrealized_pnl = self._calculate_unrealized_pnl() / self.initial_capital

        # Position duration (normalized)
        norm_duration = self.position_duration / self.max_position_duration

        position_features[-1] = [position_type, unrealized_pnl, norm_duration]

        # Combine features
        obs = np.hstack([feature_window, position_features]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        """Get additional info."""
        return {
            'capital': self.capital,
            'equity': self.equity,
            'position': self.position,
            'entry_price': self.entry_price,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'total_trades': len(self.trades),
            'current_step': self.current_step
        }

    def _get_current_price(self) -> float:
        """Get current close price."""
        return self.df.iloc[self.current_step]['close']

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL for open position."""
        if self.position is None:
            return 0.0

        current_price = self._get_current_price()

        if self.position == 'long':
            pnl = (current_price - self.entry_price) / self.entry_price * self.position_size
        else:  # short
            pnl = (self.entry_price - current_price) / self.entry_price * self.position_size

        return pnl

    def _open_position(self, direction: str, sl_pct: float, tp_pct: float):
        """Open a new position."""
        current_price = self._get_current_price()

        # Apply spread
        if direction == 'long':
            entry_price = current_price * (1 + self.spread_pct / 2)
            self.stop_loss = entry_price * (1 - sl_pct)
            self.take_profit = entry_price * (1 + tp_pct)
        else:  # short
            entry_price = current_price * (1 - self.spread_pct / 2)
            self.stop_loss = entry_price * (1 + sl_pct)
            self.take_profit = entry_price * (1 - tp_pct)

        # Position size - use max_position_pct of capital (default 20%)
        self.position_size = self.capital * self.max_position_pct
        commission = self.position_size * self.commission_pct

        # Reserve position size from capital (money is now in the trade)
        self.capital -= self.position_size
        self.capital -= commission

        self.position = direction
        self.entry_price = entry_price
        self.position_duration = 0

    def _close_position(self, reason: str = 'manual') -> float:
        """Close current position and return PnL."""
        if self.position is None:
            return 0.0

        current_price = self._get_current_price()

        # Apply spread on exit
        if self.position == 'long':
            exit_price = current_price * (1 - self.spread_pct / 2)
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            exit_price = current_price * (1 + self.spread_pct / 2)
            pnl_pct = (self.entry_price - exit_price) / self.entry_price

        # Calculate PnL
        gross_pnl = self.position_size * pnl_pct
        commission = abs(self.position_size * (1 + pnl_pct)) * self.commission_pct
        net_pnl = gross_pnl - commission

        # Update capital - add back position size plus/minus PnL
        self.capital += self.position_size + net_pnl

        # Record trade
        self.trades.append({
            'entry_step': self.current_step - self.position_duration,
            'exit_step': self.current_step,
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'duration': self.position_duration,
            'reason': reason
        })

        self.total_pnl += net_pnl

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_duration = 0

        return net_pnl

    def _check_sl_tp(self) -> Optional[str]:
        """Check if stop loss or take profit is hit."""
        if self.position is None:
            return None

        current_row = self.df.iloc[self.current_step]
        high = current_row['high']
        low = current_row['low']

        if self.position == 'long':
            if low <= self.stop_loss:
                return 'stop_loss'
            if high >= self.take_profit:
                return 'take_profit'
        else:  # short
            if high >= self.stop_loss:
                return 'stop_loss'
            if low <= self.take_profit:
                return 'take_profit'

        return None

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        reward = 0.0
        done = False
        truncated = False

        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        # Get action details
        action_type, sl, tp = self.action_map[action]

        # Check SL/TP first (before any action)
        sl_tp_hit = self._check_sl_tp()
        if sl_tp_hit:
            pnl = self._close_position(reason=sl_tp_hit)
            reward = self._calculate_reward(pnl, reason=sl_tp_hit)

        # Execute action
        if self.position is None:
            # No position - can open new one
            if action_type in ['long', 'short']:
                self._open_position(action_type, sl, tp)
        else:
            # Have position
            if action_type == 'close':
                pnl = self._close_position(reason='manual')
                reward = self._calculate_reward(pnl, reason='manual')
            elif self.position_duration >= self.max_position_duration:
                # Force close if max duration reached
                pnl = self._close_position(reason='max_duration')
                reward = self._calculate_reward(pnl, reason='max_duration')

        # Update position duration
        if self.position is not None:
            self.position_duration += 1

        # Add step-based reward (intermediate feedback)
        reward += self._calculate_step_reward()

        # Update equity
        self.equity = self.capital + self._calculate_unrealized_pnl()
        self.equity_curve.append(self.equity)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            # Close any open position at end
            if self.position is not None:
                pnl = self._close_position(reason='episode_end')
                reward += self._calculate_reward(pnl, reason='episode_end')
            done = True

        # Check for bankruptcy
        if self.equity <= 0:
            reward = -10.0  # Large penalty for bankruptcy
            done = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, done, truncated, info

    def _calculate_reward(self, pnl: float, reason: str = 'manual') -> float:
        """
        Calculate reward for a trade.
        Risk-adjusted with bonuses/penalties.
        """
        # Normalize PnL by initial capital
        normalized_pnl = pnl / self.initial_capital

        # Base reward is the normalized PnL (scaled)
        reward = normalized_pnl * 100

        # Bonus for take profit hits (good exit)
        if reason == 'take_profit':
            reward *= 1.5  # 50% bonus for TP

        # Smaller penalty for stop loss (it's risk management, not bad)
        elif reason == 'stop_loss':
            reward *= 0.8  # Reduce penalty - SL is protective

        # Penalty for max duration timeout (bad exit strategy)
        elif reason == 'max_duration':
            reward *= 0.5  # 50% penalty - should have exited earlier

        # Bonus for profitable manual exits
        elif reason == 'manual' and pnl > 0:
            reward *= 1.2  # 20% bonus for good manual exit

        # Episode end - neutral, no bonus/penalty
        elif reason == 'episode_end':
            pass  # Keep base reward

        return reward

    def _calculate_step_reward(self) -> float:
        """
        Calculate step-based reward for holding positions.
        Gives intermediate feedback during trades.
        """
        reward = 0.0

        if self.position is not None:
            # Unrealized PnL feedback (small, to guide learning)
            unrealized = self._calculate_unrealized_pnl()
            normalized_unrealized = unrealized / self.initial_capital

            # Small reward/penalty based on unrealized PnL
            reward += normalized_unrealized * 2  # Scaled down for step reward

            # Penalty for holding too long (encourages timely exits)
            duration_ratio = self.position_duration / self.max_position_duration
            if duration_ratio > 0.5:  # After 50% of max duration
                reward -= 0.01 * duration_ratio  # Increasing penalty

            # Risk penalty - if position is in significant drawdown
            if normalized_unrealized < -0.02:  # More than 2% loss
                reward -= 0.02  # Extra penalty for risky positions

        else:
            # Small penalty for being idle too long (encourages action)
            # But not too much - waiting for good entry is valid
            pass

        return reward

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Capital: ${self.capital:.2f}")
            print(f"Equity: ${self.equity:.2f}")
            print(f"Position: {self.position}")
            print(f"Total Trades: {len(self.trades)}")
            print("-" * 40)


if __name__ == "__main__":
    # Test the environment
    from indicators import load_and_preprocess_data

    print("Loading data...")
    df, features, scaler = load_and_preprocess_data(
        "data/btcusd_1-min_data.csv",
        timeframe='1h',
        start_date='2023-01-01',
        end_date='2023-06-30'
    )

    print(f"Data shape: {df.shape}")
    print(f"Features shape: {features.shape}")

    print("\nCreating environment...")
    env = BitcoinTradingEnv(df, features, window_size=48)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action map: {env.action_map}")

    print("\nRunning random episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\nEpisode finished after {i+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final equity: ${info['equity']:.2f}")
    print(f"Total trades: {info['total_trades']}")

    trades = env.get_trade_history()
    if not trades.empty:
        print(f"\nTrade history:\n{trades}")
