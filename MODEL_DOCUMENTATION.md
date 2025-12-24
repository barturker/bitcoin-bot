# Bitcoin Trading Bot - Model Documentation (V2.1 - Gatekeeper)

> **Version 2.1** - Added decomposed market regime flags for interpretable quality assessment

## Executive Summary

| Aspect | V1 (Old) | V2.0 | V2.1 (Current) |
|--------|----------|------|----------------|
| RL Role | Full trader | Gatekeeper | Gatekeeper |
| Action Space | 5 actions | 2 actions | 2 actions |
| Direction | RL chooses | Rule-based (HTF) | Rule-based (HTF) |
| SL/TP | RL chooses | Fixed (2%/4%) | Fixed (2%/4%) |
| Market Quality | N/A | Single scalar [-1,1] | **25 decomposed flags** |
| RL Sees | Raw features | Raw + quality | **Explicit regime flags** |
| Lucky Wins | Rewarded | Penalized | Penalized |

### V2.1 Key Improvement

The single `market_quality` scalar has been **decomposed into 25 explicit regime flags**:
- RL now sees WHY conditions are good/bad, not just a score
- Flags remain unnormalized (0/1 values) for interpretability
- Pre-computed quality score available for reward calculation

---

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Architecture](#architecture)
3. [Action Space](#action-space)
4. [HTF Bias (Rule-Based Direction)](#htf-bias-rule-based-direction)
5. [Reward System](#reward-system)
6. [Market Quality Assessment](#market-quality-assessment)
7. [Market Regime Flags (V2.1)](#market-regime-flags-v21)
8. [Trade Frequency Control](#trade-frequency-control)
9. [Walk-Forward Validation](#walk-forward-validation)
10. [Feature Engineering](#feature-engineering)
11. [Configuration](#configuration)
12. [Key Design Decisions](#key-design-decisions)
13. [Metrics to Monitor](#metrics-to-monitor)

---

## Core Philosophy

### The Problem with V1

In V1, the RL agent had too much power:
- Chose direction (long/short)
- Chose when to enter
- Chose SL/TP levels
- Could manually close positions

This led to:
- **Overfitting**: Model learned dataset-specific patterns
- **Lucky wins**: Profits in bad conditions reinforced bad behavior
- **Overtrading**: No penalty for excessive trading
- **Regime fragility**: Worked in one market, failed in another

### The V2 Solution: Gatekeeper Model

> **"RL should decide WHETHER to trade, not HOW to trade"**

The model now only answers ONE question:

> **"Is this a good setup to enter?"**

Everything else is rule-based:
- Direction from HTF bias
- Fixed SL/TP
- Mechanical exits
- Frequency limits

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   V2.1 GATEKEEPER ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │  Market Data │                                                   │
│  │  (1H Candles)│                                                   │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Indicators  │───►│   Features   │───►│   REGIME FLAGS       │   │
│  │  (1H + MTF)  │    │ (Normalized) │    │   (NOT Normalized)   │   │
│  └──────────────┘    └──────────────┘    │                      │   │
│                                          │  • Trend (5 flags)   │   │
│                                          │  • Volatility (5)    │   │
│                                          │  • Momentum (6)      │   │
│                                          │  • HTF Align (5)     │   │
│                                          │  • Risk (4)          │   │
│                                          │  • Quality Score     │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│         ┌───────────────────┬───────────────────────┤               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ HTF Bias     │    │   Quality    │    │     RL       │          │
│  │ (Rule-Based) │    │   Score      │    │   Agent      │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         │  LONG/SHORT/      │  regime_quality   │  TRADE/          │
│         │     NONE          │    _score         │  NO_TRADE        │
│         │                   │                   │                   │
│         └─────────┬─────────┴─────────┬─────────┘                   │
│                   │                   │                              │
│                   ▼                   ▼                              │
│            ┌──────────────────────────────┐                          │
│            │      TRADE EXECUTION         │                          │
│            │  (Only if bias != NONE       │                          │
│            │   AND action == TRADE)       │                          │
│            └──────────────────────────────┘                          │
│                          │                                           │
│                          ▼                                           │
│            ┌──────────────────────────────┐                          │
│            │     POSITION MANAGEMENT      │                          │
│            │  - Fixed SL: 2%              │                          │
│            │  - Fixed TP: 4%              │                          │
│            │  - Max Duration: 72h         │                          │
│            │  - NO manual close by RL     │                          │
│            └──────────────────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/
├── trading_env.py      # V2 Gatekeeper Environment
├── trading_env_v1_backup.py  # Old V1 backup
├── train.py            # Training script
├── walk_forward.py     # Walk-forward validation
├── indicators.py       # Data loading, indicators, MTF features
├── dashboard.py        # Visualization
└── test.py             # Testing utilities

config.yaml             # V2 Configuration
MODEL_DOCUMENTATION.md  # This file
```

---

## Action Space

### V2 Action Space (Binary)

| Action | Meaning | When Valid |
|--------|---------|------------|
| 0 | NO_TRADE | Always |
| 1 | TRADE | Only if HTF bias exists AND can_trade() |

### What Happens on TRADE Action

```python
if action == 1 and self._can_trade() and htf_bias != 0:
    direction = 'long' if htf_bias > 0 else 'short'
    self._open_position(direction)
```

The agent doesn't choose direction - it comes from HTF bias.

### What Happens on NO_TRADE Action

```python
if action == 0:
    if market_quality < 0 or htf_bias == 0:
        reward += 0.1 * (1 - market_quality)  # Good decision!
    elif market_quality > 0.3 and htf_bias != 0:
        reward -= 0.05  # Missed opportunity (small penalty)
```

---

## HTF Bias (Rule-Based Direction)

### Logic

```python
def _get_htf_bias(self) -> float:
    """
    Returns: 1.0 (LONG), -1.0 (SHORT), 0.0 (NO TRADE)
    """
    # Check for conflict first
    if trend_conflict > 0.5:
        return 0.0  # NEVER trade in conflict

    # Strong bullish confluence
    if mtf_alignment > 0.3 and daily_trend > 0 and h4_trend > 0:
        return 1.0  # LONG

    # Strong bearish confluence
    if mtf_alignment < -0.3 and daily_trend < 0 and h4_trend < 0:
        return -1.0  # SHORT

    return 0.0  # No clear bias
```

### Requirements for Valid Bias

| Condition | Required |
|-----------|----------|
| Daily trend direction | Must match |
| 4H trend direction | Must match |
| MTF alignment | > 0.3 or < -0.3 |
| Trend conflict (1H vs Daily) | Must be 0 |

### Key Insight

> **If bias is 0, trade is IMPOSSIBLE**
>
> The agent cannot force a trade when market structure doesn't support it.

---

## Reward System

### Two-Part Reward Structure

#### Part 1: Decision Reward (Immediate)

Given at the moment of decision:

| Scenario | Reward | Description |
|----------|--------|-------------|
| NO_TRADE when bad conditions | +0.1 × (1 - quality) | Correct patience |
| NO_TRADE when good opportunity | -0.05 | Missed opportunity |
| TRADE with no bias | -0.2 | Wrong decision |
| TRADE in bad quality | -0.15 × |quality| | Wrong decision |
| TRADE in good conditions | +0.05 × quality | Potentially good |

#### Part 2: Outcome Reward (Trade Close)

Given when trade closes:

```python
def _calculate_outcome_reward(pnl, reason):
    normalized_pnl = pnl / initial_capital
    pnl_component = normalized_pnl * 50  # Reduced weight

    if pnl > 0:
        if entry_quality > 0:
            # Good setup + profit = bonus
            quality_component = entry_quality * 2.0
        else:
            # LUCKY WIN = penalized
            pnl_component = min(pnl_component, 0.5)  # Cap profit
            quality_component = -0.5  # Penalty
    else:
        if entry_quality < 0:
            # Bad setup + loss = extra penalty
            quality_component = entry_quality * 1.5
        else:
            # Good setup + loss = neutral (SL worked)
            quality_component = 0.0

    return (pnl_component + quality_component) * reason_modifier
```

### Lucky Win Penalty

This is **critical**. In V1, a lucky win (profit in bad conditions) was rewarded like any profit. This trained the model to take bad trades.

In V2:
- Lucky win profit is **capped** (max 0.5 reward)
- Lucky win gets **-0.5 penalty**
- Net effect: Lucky win ≈ neutral or negative

### Reward Weights

```python
reward_weights = {
    'decision': 0.4,   # 40% for decision quality
    'outcome': 0.3,    # 30% for trade result
    'quality': 0.2,    # 20% for entry quality
    'frequency': 0.1,  # 10% for frequency penalty
}
```

---

## Market Quality Assessment

### V2.1: Pre-Computed Quality Score

In V2.1, the quality score is **pre-computed** in `indicators.py` using decomposed regime flags:

```python
def _get_market_quality(self) -> float:
    """Returns pre-computed regime_quality_score from features."""
    current_features = self.feature_df.iloc[self.current_step]

    # Use pre-computed score (V2.1)
    if 'regime_quality_score' in current_features.index:
        return float(current_features['regime_quality_score'])

    # Fallback to old calculation for backward compatibility
    # ... (legacy code)
```

### Quality Score Calculation

The `regime_quality_score` is computed from individual flags:

```python
quality = 0.0

# Positive contributions
quality += regime_trending * 0.20        # Trending market
quality += regime_strong_trend * 0.10    # Strong trend bonus
quality += regime_full_confluence * 0.25 # All TFs agree
quality += regime_normal_volatility * 0.10
quality += regime_strong_momentum * 0.05
quality += regime_trend_clarity * 0.10

# Negative contributions
quality -= regime_ranging * 0.25         # No trend
quality -= regime_trend_conflict * 0.40  # TFs disagree (critical!)
quality -= regime_chop * 0.20            # Ranging + squeeze
quality -= regime_dangerous * 0.15       # High risk conditions
quality -= regime_bb_squeeze * 0.05      # Consolidation

return clip(quality, -1, 1)
```

### Quality Interpretation

| Quality | Meaning | RL Should | Example Flags |
|---------|---------|-----------|---------------|
| > 0.5 | Excellent | Trade | `ideal_setup=1`, `full_confluence=1` |
| 0 to 0.5 | Acceptable | Consider | `trending=1`, `htf_bullish=1` |
| -0.5 to 0 | Poor | Wait | `ranging=1`, `htf_neutral=1` |
| < -0.5 | Dangerous | Avoid | `trend_conflict=1`, `chop=1` |

### Accessing Quality Components

```python
# In trading_env.py
components = env._get_quality_components()
print(components)
# {'trending': 1.0, 'ranging': 0.0, 'trend_conflict': 0.0,
#  'ideal_setup': 1.0, 'quality_score': 0.65, ...}

# In step info
obs, reward, done, truncated, info = env.step(action)
print(info['quality_components']['chop'])  # 0.0 or 1.0
```

---

## Market Regime Flags (V2.1)

### The Problem with Single Scalar Quality

In V2.0, market quality was a single number [-1, +1]. This had issues:
- Model couldn't distinguish WHY conditions were bad
- Different failure modes (chop vs conflict) looked the same
- No interpretability for debugging

### The Solution: Decomposed Regime Flags

V2.1 introduces **25 explicit regime flags** that:
- Use RAW indicator values with meaningful TA thresholds
- Remain as 0/1 values (NOT normalized)
- Give RL explicit signals it can learn to interpret

### Flag Categories

#### 1. Trend Regime (5 flags)

| Flag | Threshold | Meaning |
|------|-----------|---------|
| `regime_trending` | ADX > 25 | Market is trending |
| `regime_strong_trend` | ADX > 40 | Strong trend |
| `regime_ranging` | ADX < 20 | No trend (choppy) |
| `regime_weak_trend` | ADX 20-25 | Weak/forming trend |
| `regime_trend_clarity` | \|DI+ - DI-\| / sum | How clear is direction (0-1) |

#### 2. Volatility Regime (5 flags)

| Flag | Threshold | Meaning |
|------|-----------|---------|
| `regime_volatility_pct` | ATR percentile | Current vol vs history (0-1) |
| `regime_low_volatility` | < 25th pct | Low volatility |
| `regime_high_volatility` | > 75th pct | High volatility |
| `regime_normal_volatility` | 25-75th pct | Normal volatility |
| `regime_bb_squeeze` | BB width < 20th pct | Consolidation (breakout pending) |

#### 3. Momentum Regime (6 flags)

| Flag | Threshold | Meaning |
|------|-----------|---------|
| `regime_overbought` | RSI > 70 | Overbought |
| `regime_oversold` | RSI < 30 | Oversold |
| `regime_rsi_neutral` | RSI 40-60 | Neutral zone |
| `regime_macd_bullish` | MACD hist > 0 | Bullish momentum |
| `regime_macd_increasing` | MACD hist rising | Strengthening |
| `regime_strong_momentum` | MACD > 70th pct | Strong momentum |

#### 4. HTF Alignment (5 flags)

| Flag | Threshold | Meaning |
|------|-----------|---------|
| `regime_htf_bullish` | MTF align > 0.5 | All TFs bullish |
| `regime_htf_bearish` | MTF align < -0.5 | All TFs bearish |
| `regime_htf_neutral` | MTF align in [-0.5, 0.5] | Mixed signals |
| `regime_full_confluence` | All 3 TFs aligned | Strong confluence |
| `regime_trend_conflict` | 1H vs Daily disagree | **CRITICAL WARNING** |

#### 5. Composite Risk Flags (4 flags)

| Flag | Condition | Meaning |
|------|-----------|---------|
| `regime_ideal_setup` | trending + no conflict + HTF aligned + normal vol | **BEST CONDITIONS** |
| `regime_chop` | ranging + BB squeeze | **AVOID** |
| `regime_dangerous` | conflict OR (high vol + ranging) | **HIGH RISK** |
| `regime_caution` | weak trend OR extreme RSI + neutral HTF | Be careful |

### Normalization Behavior

```python
# In normalize_features():
skip_cols = [col for col in df.columns if col.startswith('regime_')]
# These columns are NOT normalized - they stay as 0/1 values
```

### Example: Reading Market State

```python
components = env._get_quality_components()

if components['ideal_setup'] == 1.0:
    print("Perfect setup - all conditions favorable")
elif components['chop'] == 1.0:
    print("Chop zone - DO NOT TRADE")
elif components['trend_conflict'] == 1.0:
    print("Timeframes conflict - wait for alignment")
elif components['trending'] == 1.0 and components['htf_bullish'] == 1.0:
    print("Good conditions for long")
```

---

## Trade Frequency Control

### Limits

```python
max_trades_per_day = 3  # Maximum 3 trades per 24 hours
min_cooldown = 2        # Minimum 2 hours between trades
```

### `_can_trade()` Check

```python
def _can_trade(self) -> bool:
    # Already in position
    if self.position is not None:
        return False

    # Daily limit reached
    if self.daily_trade_count >= self.max_trades_per_day:
        return False

    # Cooldown not passed
    if self.current_step - self.last_trade_step < 2:
        return False

    return True
```

### Frequency Penalty

```python
def _calculate_frequency_penalty(self) -> float:
    if daily_trade_count > max_trades_per_day * 0.7:
        return -0.05 * (daily_trade_count / max_trades_per_day)
    return 0.0
```

---

## Walk-Forward Validation

### Why Walk-Forward?

Standard train/test split has a fatal flaw:
- Scaler fit on all training data (data leakage)
- Model sees "future" statistics
- Backtest looks great, live fails

### Walk-Forward Approach

```
Fold 1: Train [2015-2016] → Validate [2017]
Fold 2: Train [2016-2017] → Validate [2018]
Fold 3: Train [2017-2018] → Validate [2019]
Fold 4: Train [2018-2019] → Validate [2020]
...
```

Each fold:
1. Split data BEFORE indicators
2. Calculate indicators separately
3. Fit scaler ONLY on training data
4. Transform validation with training scaler
5. Train model
6. Evaluate on validation

### Pass Criteria

```python
# Model passes if:
mean_return > 5%           # Positive expectancy
profitable_folds > 60%     # Consistency
mean_max_drawdown < 30%    # Acceptable risk
lucky_win_rate < 20%       # Quality entries
```

### Running Walk-Forward

```bash
python src/walk_forward.py
```

Output:
- Per-fold results
- Aggregate metrics
- PASS/FAIL verdict
- Equity curves plot

---

## Feature Engineering

### Total Features: 73 (V2.1)

| Category | Count | Normalized |
|----------|-------|------------|
| Base Indicators (1H) | 28 | Yes |
| Multi-Timeframe (4H + Daily) | 16 | Yes |
| Confluence Signals | 5 | Yes |
| **Regime Flags (V2.1)** | **25** | **No** |
| Position Features | 4 | Mixed |

#### Base Indicators (1H) - Normalized
- Price: open, high, low, close, volume
- Momentum: RSI, MACD, Stochastic, ROC, CCI
- Volatility: BB (upper/middle/lower/width), ATR, volatility
- Trend: ADX, DI+, DI-, EMA alignment, regime

#### Multi-Timeframe (4H + Daily) - Normalized
- `htf_4h_trend` / `htf_1d_trend`
- `htf_4h_rsi` / `htf_1d_rsi`
- `htf_4h_adx` / `htf_1d_adx`
- Price vs EMA21/50/200

#### Confluence Signals - Normalized
- `mtf_trend_alignment` (all TFs agree?)
- `mtf_strong_bull` / `mtf_strong_bear`
- `htf_bias` (directional bias)
- `trend_conflict` (warning signal)

#### Regime Flags (V2.1) - NOT Normalized
These 25 flags remain as raw 0/1 values:

```
Trend:      regime_trending, regime_strong_trend, regime_ranging,
            regime_weak_trend, regime_trend_clarity

Volatility: regime_volatility_pct, regime_low_volatility,
            regime_high_volatility, regime_normal_volatility,
            regime_bb_squeeze

Momentum:   regime_overbought, regime_oversold, regime_rsi_neutral,
            regime_macd_bullish, regime_macd_increasing,
            regime_strong_momentum

HTF:        regime_htf_bullish, regime_htf_bearish, regime_htf_neutral,
            regime_full_confluence, regime_trend_conflict

Risk:       regime_ideal_setup, regime_chop, regime_dangerous,
            regime_caution

Score:      regime_quality_score
```

#### Position Features (added at runtime)
- `position_type` (-1, 0, 1)
- `unrealized_pnl` (normalized)
- `position_duration` (normalized)
- `htf_bias` (current bias)

---

## Configuration

### `config.yaml`

```yaml
# V2 Configuration

data:
  train_path: "data/btcusd_1-min_data.csv"
  timeframe: "1h"
  train_start: "2012-01-01"
  train_end: "2023-12-31"

environment:
  window_size: 48
  initial_capital: 10000
  max_position_duration: 72
  max_position_pct: 0.20

trading:
  # Fixed SL/TP (V2 - RL doesn't choose)
  sl_pct: 0.02       # 2%
  tp_pct: 0.04       # 4% (R:R = 1:2)
  spread_pct: 0.001
  commission_pct: 0.001
  max_trades_per_day: 3

model:
  algorithm: "PPO"
  policy: "MlpPolicy"
  learning_rate: 0.0003
  total_timesteps: 500000

reward:
  decision_weight: 0.4
  outcome_weight: 0.3
  quality_weight: 0.2
  frequency_weight: 0.1
```

---

## Key Design Decisions

### 1. Why Only 2 Actions?

**Problem:** More actions = more ways to overfit

**Solution:** Binary decision focuses learning on the core question: "Is this setup good?"

**Result:** Simpler policy, more robust to regime changes

### 2. Why Rule-Based Direction?

**Problem:** RL choosing direction leads to random-looking trades

**Solution:** Direction from HTF bias ensures alignment with market structure

**Result:** Trades always have structural reasoning

### 3. Why Fixed SL/TP?

**Problem:** RL optimizing SL/TP leads to dataset-specific values

**Solution:** Fixed 2%/4% provides consistent risk:reward

**Result:** Stable risk management across all conditions

### 4. Why Penalize Lucky Wins?

**Problem:** Lucky profits reinforce bad behavior

**Solution:** Cap profit reward + add penalty for bad entry + profit

**Result:** Model learns entry quality, not just outcomes

### 5. Why Walk-Forward?

**Problem:** Single train/test split has data leakage

**Solution:** Rolling folds with separate scalers

**Result:** Realistic out-of-sample performance estimate

---

## Metrics to Monitor

### Training Phase

| Metric | Good | Bad |
|--------|------|-----|
| Correct decision rate | > 60% | < 40% |
| Lucky win rate | < 10% | > 25% |
| Trade frequency | 1-3/day | > 5/day |
| Episode reward | Stable/increasing | Volatile/decreasing |

### Walk-Forward Phase

| Metric | Pass | Fail |
|--------|------|------|
| Mean return | > 5% | < 0% |
| Profitable folds | > 60% | < 50% |
| Mean max DD | < 30% | > 40% |
| Lucky win rate | < 20% | > 30% |

### Live Monitoring (Future)

| Metric | Action |
|--------|--------|
| 3+ consecutive losses | Review entries |
| Lucky win spike | Check bias calculation |
| Drawdown > 20% | Pause trading |
| Trade frequency > 3/day | Review decision quality |

---

## Decision Statistics

V2 tracks detailed decision quality:

```python
decisions = {
    'correct_no_trade': 0,   # Waited in bad conditions
    'correct_trade': 0,       # Traded in good conditions
    'wrong_no_trade': 0,      # Missed good opportunity
    'wrong_trade': 0,         # Traded in bad conditions
    'lucky_wins': 0,          # Profit despite bad entry
}
```

Access with:
```python
stats = env.get_decision_stats()
print(f"Correct rate: {stats['correct_rate']*100:.1f}%")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | - | Basic RL, 5-action space, PnL rewards |
| 2.0 | Dec 2024 | Gatekeeper model, 2 actions, quality rewards, walk-forward |
| 2.1 | Dec 2024 | **Decomposed market regime flags** (25 flags), interpretable quality |

### V2.1 Changelog

**New Features:**
- Added `add_market_regime_flags()` function in `indicators.py`
- 25 explicit regime flags with meaningful TA thresholds
- Pre-computed `regime_quality_score` for reward calculation
- Flags remain unnormalized (0/1) for interpretability
- Added `_get_quality_components()` method for debugging
- Updated `render()` to show active regime flags

**Breaking Changes:**
- Old scalers are incompatible (will show warning, need retraining)
- Observation space size increased (48 → 73 features + position)

**Files Modified:**
- `src/indicators.py` - Added regime flags calculation
- `src/trading_env.py` - Updated quality methods, render, info

---

*Last updated: December 2024*
