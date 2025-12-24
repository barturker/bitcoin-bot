# Bitcoin Trading Bot - Model Documentation (V2 - Gatekeeper)

> **Version 2.0** - Complete refactor from "RL as Trader" to "RL as Gatekeeper"

## Executive Summary

| Aspect | V1 (Old) | V2 (Current) |
|--------|----------|--------------|
| RL Role | Full trader (direction + timing + exit) | Gatekeeper (only entry timing) |
| Action Space | 5 actions | 2 actions (TRADE / NO_TRADE) |
| Direction | RL chooses | Rule-based (HTF bias) |
| SL/TP | RL chooses or fixed | Fixed (2% / 4%) |
| Exit | RL can close manually | Mechanical only (SL/TP/timeout) |
| Reward | PnL-centric | Quality-centric |
| Lucky Wins | Rewarded | Penalized |

---

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Architecture](#architecture)
3. [Action Space](#action-space)
4. [HTF Bias (Rule-Based Direction)](#htf-bias-rule-based-direction)
5. [Reward System](#reward-system)
6. [Market Quality Assessment](#market-quality-assessment)
7. [Trade Frequency Control](#trade-frequency-control)
8. [Walk-Forward Validation](#walk-forward-validation)
9. [Feature Engineering](#feature-engineering)
10. [Configuration](#configuration)
11. [Key Design Decisions](#key-design-decisions)
12. [Metrics to Monitor](#metrics-to-monitor)

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
┌─────────────────────────────────────────────────────────────────┐
│                     V2 GATEKEEPER ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │  Market Data │                                               │
│  │  (1H Candles)│                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │  Indicators  │───►│   Features   │                           │
│  │  (1H + MTF)  │    │ (Normalized) │                           │
│  └──────────────┘    └──────┬───────┘                           │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ HTF Bias     │    │   Market     │    │     RL       │       │
│  │ (Rule-Based) │    │   Quality    │    │   Agent      │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         │    LONG/SHORT/    │   -1 to +1        │  TRADE/       │
│         │       NONE        │                   │  NO_TRADE     │
│         │                   │                   │               │
│         └─────────┬─────────┴─────────┬─────────┘               │
│                   │                   │                          │
│                   ▼                   ▼                          │
│            ┌──────────────────────────────┐                      │
│            │      TRADE EXECUTION         │                      │
│            │  (Only if bias != NONE       │                      │
│            │   AND action == TRADE        │                      │
│            │   AND quality allows)        │                      │
│            └──────────────────────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│            ┌──────────────────────────────┐                      │
│            │     POSITION MANAGEMENT      │                      │
│            │  - Fixed SL: 2%              │                      │
│            │  - Fixed TP: 4%              │                      │
│            │  - Max Duration: 72h         │                      │
│            │  - NO manual close by RL     │                      │
│            └──────────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
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

### Scoring System

```python
def _get_market_quality(self) -> float:
    """Returns -1 (bad) to +1 (good)"""
    quality = 0.0

    # 1. ADX (trend strength)
    if adx > 0.5: quality += 0.25
    elif adx < -0.5: quality -= 0.3

    # 2. MTF alignment (most important)
    quality += mtf_alignment * 0.4

    # 3. Strong confluence
    if mtf_strong_bull or mtf_strong_bear:
        quality += 0.2

    # 4. Trend conflict penalty
    if trend_conflict:
        quality -= 0.3

    # 5. 4H ADX
    if htf_4h_adx > 0.5: quality += 0.15
    elif htf_4h_adx < -0.5: quality -= 0.15

    return clip(quality, -1, 1)
```

### Quality Interpretation

| Quality | Meaning | RL Should |
|---------|---------|-----------|
| > 0.5 | Excellent | Consider trading |
| 0 to 0.5 | Acceptable | Trade with caution |
| -0.5 to 0 | Poor | Probably wait |
| < -0.5 | Dangerous | Definitely wait |

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

### Total Features: 48

#### Base Indicators (1H)
- Price: open, high, low, close, volume
- Momentum: RSI, MACD, Stochastic, ROC, CCI
- Volatility: BB, ATR, volatility
- Trend: ADX, EMA alignment, regime

#### Multi-Timeframe (4H + Daily)
- `htf_4h_trend` / `htf_1d_trend`
- `htf_4h_rsi` / `htf_1d_rsi`
- `htf_4h_adx` / `htf_1d_adx`
- Price vs EMA21/50

#### Confluence Signals
- `mtf_trend_alignment` (all TFs agree?)
- `mtf_strong_bull` / `mtf_strong_bear`
- `htf_bias` (directional bias)
- `trend_conflict` (warning signal)

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
| 2.0 | Current | Gatekeeper model, 2 actions, quality rewards, walk-forward |

---

*Last updated: December 2024*
