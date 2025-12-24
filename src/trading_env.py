"""
Bitcoin Trading Environment for Reinforcement Learning - V2 (Gatekeeper)

Key Design Principles:
- RL is a GATEKEEPER, not a trader
- RL only decides: TRADE or NO_TRADE
- Direction is rule-based (HTF bias)
- Exit is mechanical (SL/TP/max_duration)
- Reward is quality-based, not PnL-centric
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from collections import deque


class BitcoinTradingEnv(gym.Env):
    """
    Bitcoin Trading Environment - Gatekeeper Model

    RL Agent Role:
        - Decides WHETHER to trade, not HOW to trade
        - Direction comes from HTF bias (rule-based)
        - RL learns: "Is this a good setup to enter?"

    Action Space:
        0: NO_TRADE (wait)
        1: TRADE (execute based on HTF bias)

    Exit Mechanism:
        - Stop Loss (fixed %)
        - Take Profit (fixed %)
        - Max Duration timeout
        - NO manual close by RL
    """

    metadata = {'render_modes': ['human']}

    # Bias thresholds
    BIAS_THRESHOLD = 0.3  # Minimum MTF alignment for valid bias
    QUALITY_THRESHOLD = 0.0  # Minimum quality to consider trading

    def __init__(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        window_size: int = 48,
        initial_capital: float = 10000.0,
        max_position_duration: int = 72,
        sl_pct: float = 0.02,  # Fixed 2% SL
        tp_pct: float = 0.04,  # Fixed 4% TP (R:R = 1:2)
        spread_pct: float = 0.001,
        commission_pct: float = 0.001,
        max_position_pct: float = 0.20,
        max_trades_per_day: int = 3,  # Trade frequency limit
        render_mode: str = None
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_df = feature_df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_position_duration = max_position_duration
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.spread_pct = spread_pct
        self.commission_pct = commission_pct
        self.max_position_pct = max_position_pct
        self.max_trades_per_day = max_trades_per_day
        self.render_mode = render_mode

        # Action space: Binary (TRADE / NO_TRADE)
        self.action_space = spaces.Discrete(2)

        # Observation space
        n_features = feature_df.shape[1]
        # Position features: position_type, unrealized_pnl, position_duration, htf_bias
        self.n_position_features = 4
        total_features = n_features + self.n_position_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, total_features),
            dtype=np.float32
        )

        # Reward weights (tunable)
        self.reward_weights = {
            'decision': 0.4,      # Weight for decision reward
            'outcome': 0.3,       # Weight for trade outcome
            'quality': 0.2,       # Weight for entry quality
            'frequency': 0.1,     # Weight for frequency penalty
        }

        # Initialize state
        self.reset()

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

        # Quality tracking
        self.entry_market_quality = 0.0
        self.entry_htf_bias = 0.0

        # Trade tracking
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.total_pnl = 0.0

        # Trade frequency tracking (rolling 24-hour window)
        self.recent_trades = deque(maxlen=24)  # Track trades per hour for last 24h
        self.daily_trade_count = 0
        self.last_trade_step = -999  # Cooldown tracking

        # Metrics for analysis
        self.decisions = {
            'correct_no_trade': 0,  # Didn't trade in bad conditions
            'correct_trade': 0,      # Traded in good conditions + profit
            'wrong_no_trade': 0,     # Missed good opportunity
            'wrong_trade': 0,        # Traded in bad conditions
            'lucky_wins': 0,         # Profit despite bad entry
        }

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_htf_bias(self) -> float:
        """
        Calculate HTF (Higher Timeframe) bias for trade direction.
        Returns: 1.0 (long), -1.0 (short), 0.0 (no trade)

        Rule-based logic:
        - Daily trend + 4H confirmation required
        - Conflict = no bias
        """
        current_features = self.feature_df.iloc[self.current_step]

        # Get MTF trend alignment
        mtf_alignment = current_features.get('mtf_trend_alignment', 0.0)

        # Get individual timeframe trends
        daily_trend = current_features.get('htf_1d_trend', 0.0)
        h4_trend = current_features.get('htf_4h_trend', 0.0)
        h1_regime = current_features.get('market_regime', 0.0)

        # Check for conflict
        trend_conflict = current_features.get('trend_conflict', 0.0)
        if trend_conflict > 0.5:
            return 0.0  # No trade when timeframes conflict

        # Strong confluence required
        if mtf_alignment > self.BIAS_THRESHOLD:
            # All timeframes bullish
            if daily_trend > 0 and h4_trend > 0:
                return 1.0  # LONG bias
        elif mtf_alignment < -self.BIAS_THRESHOLD:
            # All timeframes bearish
            if daily_trend < 0 and h4_trend < 0:
                return -1.0  # SHORT bias

        return 0.0  # No clear bias

    def _get_market_quality(self) -> float:
        """
        Get market quality score from pre-computed regime flags.

        Returns value between -1 (bad) and 1 (good).
        Uses decomposed regime flags for interpretable quality assessment.

        The quality score is pre-computed in indicators.py using:
        - Trend regime (ADX-based)
        - Volatility regime (ATR percentile)
        - HTF alignment (multi-timeframe confluence)
        - Risk flags (conflict, chop, dangerous conditions)
        """
        current_features = self.feature_df.iloc[self.current_step]

        # Use pre-computed quality score if available (new system)
        if 'regime_quality_score' in current_features.index:
            return float(current_features['regime_quality_score'])

        # Fallback to old calculation for backward compatibility
        quality = 0.0

        if 'adx_14' in current_features.index:
            adx = current_features['adx_14']
            if adx > 0.5:
                quality += 0.25
            elif adx < -0.5:
                quality -= 0.3

        if 'mtf_trend_alignment' in current_features.index:
            mtf_align = current_features['mtf_trend_alignment']
            quality += mtf_align * 0.4

        if 'mtf_strong_bull' in current_features.index:
            if current_features['mtf_strong_bull'] > 0.5:
                quality += 0.2
            elif current_features.get('mtf_strong_bear', 0) > 0.5:
                quality += 0.2

        if 'trend_conflict' in current_features.index:
            if current_features['trend_conflict'] > 0.5:
                quality -= 0.3

        if 'htf_4h_adx' in current_features.index:
            htf_adx = current_features['htf_4h_adx']
            if htf_adx > 0.5:
                quality += 0.15
            elif htf_adx < -0.5:
                quality -= 0.15

        return max(-1.0, min(1.0, quality))

    def _get_quality_components(self) -> dict:
        """
        Get decomposed quality components for analysis and debugging.

        Returns dict with individual regime flags and their current values.
        This allows understanding WHY the quality score is what it is.

        Categories returned:
            - trend: trending, strong_trend, ranging, weak_trend, trend_clarity
            - volatility: volatility_pct, low/high/normal, bb_squeeze
            - momentum: overbought, oversold, macd_bullish, macd_increasing
            - htf: htf_bullish, htf_bearish, htf_neutral, full_confluence
            - risk: trend_conflict, ideal_setup, chop, dangerous, caution
            - score: quality_score (composite)
        """
        current = self.feature_df.iloc[self.current_step]
        components = {}

        # Extract all regime flags
        regime_cols = [col for col in current.index if col.startswith('regime_')]

        for col in regime_cols:
            # Remove 'regime_' prefix for cleaner keys
            key = col.replace('regime_', '')
            components[key] = float(current[col])

        return components

    def _can_trade(self) -> bool:
        """Check if trading is allowed (frequency limits, cooldown, etc.)."""
        # Already in position
        if self.position is not None:
            return False

        # Trade frequency limit
        if self.daily_trade_count >= self.max_trades_per_day:
            return False

        # Minimum cooldown between trades (2 hours)
        if self.current_step - self.last_trade_step < 2:
            return False

        return True

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

        # HTF bias (important for RL to know current direction)
        htf_bias = self._get_htf_bias()

        position_features[-1] = [position_type, unrealized_pnl, norm_duration, htf_bias]

        # Combine features
        obs = np.hstack([feature_window, position_features]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        """Get additional info including decomposed quality components."""
        return {
            'capital': self.capital,
            'equity': self.equity,
            'position': self.position,
            'entry_price': self.entry_price,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'total_trades': len(self.trades),
            'current_step': self.current_step,
            'market_quality': self._get_market_quality(),
            'htf_bias': self._get_htf_bias(),
            'daily_trade_count': self.daily_trade_count,
            'decisions': self.decisions.copy(),
            # Decomposed quality components for analysis
            'quality_components': self._get_quality_components()
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
        else:
            pnl = (self.entry_price - current_price) / self.entry_price * self.position_size

        return pnl

    def _open_position(self, direction: str):
        """Open a new position with fixed SL/TP."""
        current_price = self._get_current_price()

        # Apply spread
        if direction == 'long':
            entry_price = current_price * (1 + self.spread_pct / 2)
            self.stop_loss = entry_price * (1 - self.sl_pct)
            self.take_profit = entry_price * (1 + self.tp_pct)
        else:
            entry_price = current_price * (1 - self.spread_pct / 2)
            self.stop_loss = entry_price * (1 + self.sl_pct)
            self.take_profit = entry_price * (1 - self.tp_pct)

        # Position sizing
        self.position_size = self.capital * self.max_position_pct
        commission = self.position_size * self.commission_pct

        self.capital -= self.position_size
        self.capital -= commission

        self.position = direction
        self.entry_price = entry_price
        self.position_duration = 0

        # Track entry conditions
        self.entry_market_quality = self._get_market_quality()
        self.entry_htf_bias = self._get_htf_bias()

        # Update trade tracking
        self.daily_trade_count += 1
        self.last_trade_step = self.current_step
        self.recent_trades.append(self.current_step)

    def _close_position(self, reason: str = 'sl_tp') -> float:
        """Close current position and return PnL."""
        if self.position is None:
            return 0.0

        current_price = self._get_current_price()

        # Apply spread on exit
        if self.position == 'long':
            exit_price = current_price * (1 - self.spread_pct / 2)
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            exit_price = current_price * (1 + self.spread_pct / 2)
            pnl_pct = (self.entry_price - exit_price) / self.entry_price

        # Calculate PnL
        gross_pnl = self.position_size * pnl_pct
        commission = abs(self.position_size * (1 + pnl_pct)) * self.commission_pct
        net_pnl = gross_pnl - commission

        self.capital += self.position_size + net_pnl

        # Determine if this was a lucky win
        is_lucky_win = (net_pnl > 0 and self.entry_market_quality < 0)
        if is_lucky_win:
            self.decisions['lucky_wins'] += 1

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
            'reason': reason,
            'entry_market_quality': self.entry_market_quality,
            'entry_htf_bias': self.entry_htf_bias,
            'is_lucky_win': is_lucky_win
        })

        self.total_pnl += net_pnl

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_duration = 0
        self.entry_market_quality = 0.0
        self.entry_htf_bias = 0.0

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
        else:
            if high >= self.stop_loss:
                return 'stop_loss'
            if low <= self.take_profit:
                return 'take_profit'

        return None

    def _calculate_decision_reward(self, action: int, market_quality: float, htf_bias: float) -> float:
        """
        Calculate immediate reward for the decision itself.
        This teaches the model WHEN to trade.
        """
        reward = 0.0

        # Action: NO_TRADE (0)
        if action == 0:
            if market_quality < self.QUALITY_THRESHOLD or htf_bias == 0:
                # Correct decision: Didn't trade in bad conditions
                reward = 0.1 * (1 - market_quality)  # More reward for avoiding worse conditions
                self.decisions['correct_no_trade'] += 1
            elif market_quality > 0.3 and htf_bias != 0:
                # Missed opportunity (but small penalty - being conservative is OK)
                reward = -0.05
                self.decisions['wrong_no_trade'] += 1

        # Action: TRADE (1)
        elif action == 1:
            if not self._can_trade():
                # Tried to trade when not allowed - no immediate penalty
                # (the lack of position is punishment enough)
                pass
            elif htf_bias == 0:
                # Tried to trade with no bias - BAD
                reward = -0.2
                self.decisions['wrong_trade'] += 1
            elif market_quality < self.QUALITY_THRESHOLD:
                # Trading in bad conditions - BAD
                reward = -0.15 * abs(market_quality)
                self.decisions['wrong_trade'] += 1
            else:
                # Trading in good conditions with bias - potentially good
                # Final verdict comes from outcome reward
                reward = 0.05 * market_quality
                self.decisions['correct_trade'] += 1

        return reward

    def _calculate_outcome_reward(self, pnl: float, reason: str) -> float:
        """
        Calculate reward when trade closes.
        Quality-adjusted, with lucky win penalty.
        """
        # Normalize PnL
        normalized_pnl = pnl / self.initial_capital
        entry_quality = self.entry_market_quality

        # Base outcome (small weight on raw PnL)
        pnl_component = normalized_pnl * 50  # Reduced from 100

        # Quality adjustment
        quality_component = 0.0

        if pnl > 0:
            # Profitable trade
            if entry_quality > 0:
                # Good setup + profit = well done
                quality_component = entry_quality * 2.0
            else:
                # LUCKY WIN - bad setup + profit
                # Clamp the reward, don't reinforce this behavior
                pnl_component = min(pnl_component, 0.5)  # Cap the profit reward
                quality_component = -0.5  # Penalty for lucky win
        else:
            # Losing trade
            if entry_quality < 0:
                # Bad setup + loss = extra penalty (should have known)
                quality_component = entry_quality * 1.5  # Amplify penalty
            else:
                # Good setup + loss = acceptable (SL did its job)
                quality_component = 0.0  # Neutral

        # Exit reason adjustment
        reason_modifier = 1.0
        if reason == 'take_profit':
            reason_modifier = 1.1
        elif reason == 'max_duration':
            reason_modifier = 0.8
        # stop_loss is neutral (1.0)

        total = (pnl_component + quality_component) * reason_modifier

        return total

    def _calculate_frequency_penalty(self) -> float:
        """
        Penalize overtrading.
        """
        penalty = 0.0

        # Penalty for approaching daily limit
        if self.daily_trade_count > self.max_trades_per_day * 0.7:
            penalty -= 0.05 * (self.daily_trade_count / self.max_trades_per_day)

        return penalty

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        reward = 0.0
        done = False
        truncated = False

        # Convert action
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        # Get current market state
        market_quality = self._get_market_quality()
        htf_bias = self._get_htf_bias()

        # ============ HANDLE EXISTING POSITION ============
        if self.position is not None:
            # Check SL/TP
            sl_tp_hit = self._check_sl_tp()
            if sl_tp_hit:
                pnl = self._close_position(reason=sl_tp_hit)
                reward += self._calculate_outcome_reward(pnl, reason=sl_tp_hit)

            # Check max duration
            elif self.position_duration >= self.max_position_duration:
                pnl = self._close_position(reason='max_duration')
                reward += self._calculate_outcome_reward(pnl, reason='max_duration')

            # Still in position - update duration
            if self.position is not None:
                self.position_duration += 1

        # ============ HANDLE NEW DECISION ============
        else:
            # No position - RL makes decision
            decision_reward = self._calculate_decision_reward(action, market_quality, htf_bias)
            reward += decision_reward * self.reward_weights['decision']

            # Execute trade if action is TRADE and conditions allow
            if action == 1 and self._can_trade() and htf_bias != 0:
                direction = 'long' if htf_bias > 0 else 'short'
                self._open_position(direction)

        # ============ FREQUENCY PENALTY ============
        freq_penalty = self._calculate_frequency_penalty()
        reward += freq_penalty * self.reward_weights['frequency']

        # ============ UPDATE STATE ============
        self.equity = self.capital + self._calculate_unrealized_pnl()
        self.equity_curve.append(self.equity)
        self.current_step += 1

        # Reset daily counter (simplified - every 24 steps)
        if self.current_step % 24 == 0:
            self.daily_trade_count = 0

        # ============ CHECK TERMINATION ============
        if self.current_step >= len(self.df) - 1:
            if self.position is not None:
                pnl = self._close_position(reason='episode_end')
                reward += self._calculate_outcome_reward(pnl, reason='episode_end')
            done = True

        if self.equity <= 0:
            reward = -10.0
            done = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, done, truncated, info

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)

    def get_decision_stats(self) -> dict:
        """Get decision quality statistics."""
        total = sum(self.decisions.values())
        if total == 0:
            return self.decisions

        stats = self.decisions.copy()
        stats['total_decisions'] = total
        stats['correct_rate'] = (
            stats['correct_no_trade'] + stats['correct_trade']
        ) / max(1, total)
        stats['lucky_win_rate'] = stats['lucky_wins'] / max(1, len(self.trades))

        return stats

    def render(self):
        """Render the environment with regime flags."""
        if self.render_mode == 'human':
            htf_bias = self._get_htf_bias()
            bias_str = "LONG" if htf_bias > 0 else "SHORT" if htf_bias < 0 else "NONE"
            components = self._get_quality_components()

            print(f"Step: {self.current_step}")
            print(f"Capital: ${self.capital:.2f} | Equity: ${self.equity:.2f}")
            print(f"Position: {self.position} | HTF Bias: {bias_str}")
            print(f"Market Quality: {self._get_market_quality():.2f}")

            # Show key regime flags
            regime_flags = []
            if components.get('trending', 0) > 0.5:
                regime_flags.append("TREND")
            if components.get('strong_trend', 0) > 0.5:
                regime_flags.append("STRONG")
            if components.get('ranging', 0) > 0.5:
                regime_flags.append("RANGE")
            if components.get('htf_bullish', 0) > 0.5:
                regime_flags.append("HTF_BULL")
            if components.get('htf_bearish', 0) > 0.5:
                regime_flags.append("HTF_BEAR")
            if components.get('trend_conflict', 0) > 0.5:
                regime_flags.append("CONFLICT!")
            if components.get('chop', 0) > 0.5:
                regime_flags.append("CHOP!")
            if components.get('ideal_setup', 0) > 0.5:
                regime_flags.append("IDEAL*")
            print(f"Regime: [{' | '.join(regime_flags) if regime_flags else 'NEUTRAL'}]")

            print(f"Trades Today: {self.daily_trade_count}/{self.max_trades_per_day}")
            print(f"Total Trades: {len(self.trades)}")
            print("-" * 50)


if __name__ == "__main__":
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

    print("\nCreating environment (V2 - Gatekeeper)...")
    env = BitcoinTradingEnv(df, features, window_size=48)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    print("\nRunning random episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\nEpisode finished after {i+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final equity: ${info['equity']:.2f}")
    print(f"Total trades: {info['total_trades']}")

    print("\nDecision Stats:")
    for k, v in env.get_decision_stats().items():
        print(f"  {k}: {v}")

    trades = env.get_trade_history()
    if not trades.empty:
        print(f"\nTrade history:\n{trades[['direction', 'pnl', 'entry_market_quality', 'is_lucky_win']]}")
