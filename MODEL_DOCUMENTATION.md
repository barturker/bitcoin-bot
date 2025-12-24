# Bitcoin Trading Bot - Model Documentation (V2.3 - Gatekeeper)

> **Version 2.3** - HARD VETO system: dangerous conditions now OVERRIDE RL decisions completely

## Executive Summary

| Aspect | V1 (Old) | V2.0 | V2.1 | V2.2 | V2.3 (Current) |
|--------|----------|------|------|------|----------------|
| RL Role | Full trader | Gatekeeper | Gatekeeper | Gatekeeper | **Constrained Gatekeeper** |
| Action Space | 5 actions | 2 actions | 2 actions | 2 actions | 2 actions |
| Direction | RL chooses | Rule-based | Rule-based | Rule-based | Rule-based |
| SL/TP | RL chooses | Fixed 2%/4% | Fixed 2%/4% | Fixed 2%/4% | Fixed 2%/4% |
| Danger Handling | None | None | Flags only | Penalties | **HARD VETO** |
| Trade Frequency | Unlimited | 3/day | 3/day | 3/day | **1/day + 24h cooldown** |
| RL Freedom | Full | High | High | Medium | **Low (allowed zones only)** |

### V2.3 Key Improvement: HARD VETO

V2.2 tried to teach RL to avoid danger with penalties. **It didn't work** - model still lost in bear markets.

**V2.3 Solution:** Don't ask RL in dangerous conditions. **FORCE NO_TRADE.**

```python
HARD_VETO_FLAGS = [
    'regime_bear_trend',      # Bear market = NO TRADE
    'regime_shock',           # Crisis = NO TRADE
    'regime_trend_conflict',  # TFs disagree = NO TRADE
    'regime_chop',            # Range + squeeze = NO TRADE
    'regime_no_trade_zone',   # Composite danger = NO TRADE
]

# If ANY flag is active → RL decision is OVERRIDDEN → Forced NO_TRADE
```

**Philosophy change:**
- V2.2: "Let RL learn from penalties in dangerous conditions"
- V2.3: "RL is NOT ALLOWED to trade in dangerous conditions"

> **"Edge is thin. Thin edge needs hard rules."**

---

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Architecture](#architecture)
3. [HARD VETO System (V2.3)](#hard-veto-system-v23)
4. [Action Space](#action-space)
5. [HTF Bias (Rule-Based Direction)](#htf-bias-rule-based-direction)
6. [Reward System](#reward-system)
7. [Market Quality Assessment](#market-quality-assessment)
8. [Market Regime Flags](#market-regime-flags)
9. [Bear/Shock Detection (V2.2)](#bearshock-detection-v22)
10. [Trade Frequency Control](#trade-frequency-control)
11. [Walk-Forward Validation](#walk-forward-validation)
12. [Feature Engineering](#feature-engineering)
13. [Configuration](#configuration)
14. [Key Design Decisions](#key-design-decisions)
15. [Metrics to Monitor](#metrics-to-monitor)

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
│                   V2.3 GATEKEEPER ARCHITECTURE                       │
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
│  └──────────────┘    └──────────────┘    │  • 32 flags total    │   │
│                                          │  • Includes VETO     │   │
│                                          │    flags (V2.3)      │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│  ╔══════════════════════════════════════════════════╧═══════════╗   │
│  ║                    HARD VETO LAYER (V2.3)                    ║   │
│  ║  ┌─────────────────────────────────────────────────────────┐ ║   │
│  ║  │  IF any VETO flag active:                               │ ║   │
│  ║  │    • regime_bear_trend                                  │ ║   │
│  ║  │    • regime_shock                                       │ ║   │
│  ║  │    • regime_trend_conflict                              │ ║   │
│  ║  │    • regime_chop                                        │ ║   │
│  ║  │    • regime_no_trade_zone                               │ ║   │
│  ║  │  THEN: FORCE action = NO_TRADE (RL bypassed)            │ ║   │
│  ║  └─────────────────────────────────────────────────────────┘ ║   │
│  ╚══════════════════════════════════════════════════════════════╝   │
│                                   │                                  │
│         ┌─────────────────────────┼─────────────────────────┐       │
│         ▼                         ▼                         ▼       │
│  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐   │
│  │ HTF Bias     │    │   FREQUENCY LIMIT    │    │     RL       │   │
│  │ (Rule-Based) │    │   (V2.3)             │    │   Agent      │   │
│  └──────┬───────┘    │  • 1 trade/day max   │    │ (only in     │   │
│         │            │  • 24h cooldown      │    │ ALLOWED zone)│   │
│         │            └──────────┬───────────┘    └──────┬───────┘   │
│         │                       │                       │           │
│         └─────────┬─────────────┴─────────┬─────────────┘           │
│                   │                       │                          │
│                   ▼                       ▼                          │
│            ┌──────────────────────────────┐                          │
│            │      TRADE EXECUTION         │                          │
│            │  (Only if:                   │                          │
│            │   • NO veto active           │                          │
│            │   • bias != NONE             │                          │
│            │   • action == TRADE          │                          │
│            │   • cooldown passed)         │                          │
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

## HARD VETO System (V2.3)

### Why Hard Veto?

V2.2 results showed that **penalties don't work** for teaching RL to avoid danger:

| Version | Approach | Result |
|---------|----------|--------|
| V2.1 | Regime flags as features | Model ignores them |
| V2.2 | Massive penalties (-0.5) | Model still trades in bear, loses |
| **V2.3** | **HARD VETO (override RL)** | **RL cannot trade in danger** |

### The Insight

> **"If RL keeps making the same mistake despite penalties,
> remove the possibility of making that mistake."**

### Veto Flags

These 5 flags trigger absolute prohibition:

| Flag | Condition | Why Veto |
|------|-----------|----------|
| `regime_bear_trend` | ADX>25 + price<EMA50<EMA200 + DI->DI+ | Bear markets kill accounts |
| `regime_shock` | ATR spike + volume/candle spike | Crisis = unpredictable |
| `regime_trend_conflict` | 1H vs Daily disagree | No edge when TFs fight |
| `regime_chop` | Ranging + BB squeeze | Whipsaw city |
| `regime_no_trade_zone` | Composite of above | Multiple dangers |

### Implementation

```python
def _is_hard_veto(self) -> Tuple[bool, str]:
    """Check if ANY hard veto flag is active."""
    current = self.feature_df.iloc[self.current_step]

    for flag in self.HARD_VETO_FLAGS:
        if current.get(flag, 0) > 0.5:
            return True, flag.replace('regime_', '')

    return False, ''

def step(self, action):
    # Check HARD VETO before anything else
    is_vetoed, veto_reason = self._is_hard_veto()

    if is_vetoed and action == 1:
        # RL wanted to trade but we're vetoing it
        reward += 0.15  # Teach RL this is correct
        action = 0  # FORCE NO_TRADE
        self.decisions['hard_vetoes'] += 1
```

### Veto Statistics

Walk-forward now reports `hard_vetoes` count per fold:
- High veto count in bear years (2020, 2022) = **system working correctly**
- Low veto count in bull years = **RL has room to trade**

### What RL Learns

With hard veto, RL training becomes:
1. **Dangerous conditions:** RL cannot trade → learns "these states = no opportunity"
2. **Allowed conditions:** RL makes real decisions → learns actual edge

This is cleaner than penalties because:
- No gradient noise from impossible actions
- RL focuses on learnable patterns only
- Faster convergence in allowed zones

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
| NO_TRADE in danger zone (V2.2) | **+0.3** | **Excellent patience** |
| TRADE with no bias | -0.2 | Wrong decision |
| TRADE in bad quality | -0.15 × |quality| | Wrong decision |
| TRADE in good conditions | +0.05 × quality | Potentially good |
| **TRADE during shock (V2.2)** | **-0.5** | **TERRIBLE decision** |
| **TRADE in bear trend (V2.2)** | **-0.4** | **Very bad decision** |
| **TRADE in no_trade_zone (V2.2)** | **-0.3** | **Bad decision** |

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

## Market Regime Flags

### The Problem with Single Scalar Quality

In V2.0, market quality was a single number [-1, +1]. This had issues:
- Model couldn't distinguish WHY conditions were bad
- Different failure modes (chop vs conflict) looked the same
- No interpretability for debugging

### The Solution: Decomposed Regime Flags

V2.1 introduced **25 explicit regime flags**, V2.2 expanded to **32 flags** that:
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

#### 6. Shock/Crisis Detection (V2.2 - 3 flags)

| Flag | Condition | Meaning |
|------|-----------|---------|
| `regime_atr_spike` | ATR > 2× median | Abnormal volatility |
| `regime_volume_spike` | Volume > 3× mean | Panic volume |
| `regime_large_candles` | Candle size > 3× ATR | Extreme price movement |

#### 7. Bear/Bull Trend Direction (V2.2 - 4 flags)

| Flag | Condition | Meaning |
|------|-----------|---------|
| `regime_bull_trend` | ADX>25 + price>EMA50>EMA200 + DI+>DI- | **SAFE TO LONG** |
| `regime_bear_trend` | ADX>25 + price<EMA50<EMA200 + DI->DI+ | **AVOID TRADING** |
| `regime_shock` | ATR spike + (volume spike OR large candles) | **CRISIS - DO NOT TRADE** |
| `regime_no_trade_zone` | shock OR bear_trend OR (high_vol + ranging) | **HARD VETO** |

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

## Bear/Shock Detection (V2.2)

### The Problem: Bear Market Losses

Walk-forward validation revealed a critical issue:

| Validation Year | Return | Issue |
|-----------------|--------|-------|
| 2017 | +12.5% | Bull market ✓ |
| 2018 | -3.2% | Bear, some losses |
| 2019 | +8.4% | Sideways, OK |
| **2020** | **-15.8%** | **COVID crash** |
| 2021 | +15.2% | Bull ✓ |
| **2022** | **-21.4%** | **Bear market** |
| 2023 | +5.1% | Recovery |

The model knew "is there a trend?" but NOT "is this trend tradeable (bullish)?"

### The Solution: Bull vs Bear Separation

**Previous (V2.1):**
```python
regime_trending = ADX > 25  # "There is a trend" (but which direction?)
```

**New (V2.2):**
```python
# BULL trend: safe to trade
regime_bull_trend = (
    (ADX > 25) &
    (close > EMA_50) &
    (EMA_50 > EMA_200) &
    (DI+ > DI-)
)

# BEAR trend: AVOID trading
regime_bear_trend = (
    (ADX > 25) &
    (close < EMA_50) &
    (EMA_50 < EMA_200) &
    (DI- > DI+)
)
```

### Shock Detection

Crisis conditions (COVID crash, flash crashes):

```python
# ATR spike: volatility > 2x normal
regime_atr_spike = ATR > (2 × rolling_median_ATR)

# Volume spike: panic selling/buying
regime_volume_spike = Volume > (3 × rolling_mean_volume)

# Large candles: extreme moves
regime_large_candles = |high - low| > (3 × ATR)

# SHOCK = multiple crisis indicators
regime_shock = regime_atr_spike & (regime_volume_spike | regime_large_candles)
```

### No Trade Zone (Hard Veto)

```python
regime_no_trade_zone = (
    regime_shock |           # Crisis
    regime_bear_trend |      # Bear market
    (regime_high_volatility & regime_ranging)  # Chaos
)
```

### Quality Score Impact (V2.2)

```python
# MASSIVE penalties for dangerous conditions
quality -= regime_bear_trend * 0.60      # Bear = -60% quality
quality -= regime_shock * 0.80           # Shock = -80% quality
quality -= regime_no_trade_zone * 0.50   # No-trade = -50% quality
quality -= regime_atr_spike * 0.15       # ATR spike penalty
quality -= regime_volume_spike * 0.10    # Volume spike penalty
quality -= regime_large_candles * 0.10   # Large candles penalty
```

### Reward Penalties (V2.2)

```python
def _calculate_decision_reward(action):
    # Check danger flags
    is_shock = components.get('shock', 0) == 1
    is_bear_trend = components.get('bear_trend', 0) == 1
    is_no_trade_zone = components.get('no_trade_zone', 0) == 1

    if action == TRADE:
        if is_shock:
            return -0.5  # TERRIBLE: Trading during crisis
        elif is_bear_trend:
            return -0.4  # VERY BAD: Trading in bear market
        elif is_no_trade_zone:
            return -0.3  # BAD: Trading in no-trade zone

    elif action == NO_TRADE:
        if is_shock or is_bear_trend or is_no_trade_zone:
            return +0.3  # EXCELLENT: Staying out of danger
```

### Expected Behavior

| Market Condition | V2.1 Behavior | V2.2 Behavior |
|------------------|---------------|---------------|
| 2020 COVID crash | Tries to short, loses | Detects shock, stays out |
| 2022 bear market | Tries to short, loses | Detects bear_trend, stays out |
| Bull market dip | May enter too early | Waits for bull_trend to return |
| Flash crash | Gets stopped out | Shock detection prevents entry |

### Why Not Just Trade Bear Trends?

**Problem with shorting bear markets:**
1. **Retail platforms** often have poor short execution
2. **Bear rallies** are violent and unpredictable
3. **Timing** bear market entries is extremely difficult
4. **Model learned** on more bull data (Bitcoin's history)

**Better strategy:** Sit out bear markets entirely, preserve capital for bull runs.

---

## Trade Frequency Control

### V2.3: Brutal Limits

Walk-forward analysis showed model was **overtrading** (100-240 trades per year).
Quality > Quantity.

| Version | Max Trades/Day | Cooldown | Result |
|---------|----------------|----------|--------|
| V2.0-2.2 | 3 | 2 hours | Too many trades, losses accumulate |
| **V2.3** | **1** | **24 hours** | Forced selectivity |

### Limits

```python
max_trades_per_day = 1   # Maximum 1 trade per 24 hours (V2.3)
min_cooldown = 24        # Minimum 24 hours between trades (V2.3)
```

### `_can_trade()` Check

```python
def _can_trade(self) -> bool:
    # Already in position
    if self.position is not None:
        return False

    # V2.3: HARD VETO - absolute prohibition
    is_vetoed, _ = self._is_hard_veto()
    if is_vetoed:
        return False

    # Daily limit reached
    if self.daily_trade_count >= self.max_trades_per_day:
        return False

    # V2.3: Minimum cooldown between trades (24 hours)
    if self.current_step - self.last_trade_step < 24:
        return False

    return True
```

### Expected Trade Frequency

With 1/day limit + 24h cooldown + hard veto:

| Market Condition | Expected Trades/Month |
|------------------|----------------------|
| Bull market | 15-20 (good setups exist) |
| Sideways | 5-10 (fewer opportunities) |
| Bear market | 0-2 (mostly vetoed) |
| Crisis (shock) | 0 (hard veto blocks all) |

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

### Total Features: 80 (V2.2)

| Category | Count | Normalized |
|----------|-------|------------|
| Base Indicators (1H) | 28 | Yes |
| Multi-Timeframe (4H + Daily) | 16 | Yes |
| Confluence Signals | 5 | Yes |
| **Regime Flags (V2.2)** | **32** | **No** |
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

#### Regime Flags (V2.2) - NOT Normalized
These 32 flags remain as raw 0/1 values:

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

Shock:      regime_atr_spike, regime_volume_spike, regime_large_candles
(V2.2)

Direction:  regime_bull_trend, regime_bear_trend, regime_shock,
(V2.2)      regime_no_trade_zone

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
  max_trades_per_day: 1  # V2.3: Brutal limit

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

V2.3 tracks detailed decision quality including hard vetoes:

```python
decisions = {
    'correct_no_trade': 0,   # Waited in bad conditions
    'correct_trade': 0,       # Traded in good conditions
    'wrong_no_trade': 0,      # Missed good opportunity
    'wrong_trade': 0,         # Traded in bad conditions
    'lucky_wins': 0,          # Profit despite bad entry
    'hard_vetoes': 0,         # V2.3: Times RL was overridden
}
```

Access with:
```python
stats = env.get_decision_stats()
print(f"Correct rate: {stats['correct_rate']*100:.1f}%")
print(f"Hard vetoes: {stats['hard_vetoes']}")  # V2.3
```

### Interpreting Hard Vetoes

| Veto Count | Interpretation |
|------------|----------------|
| High (1000+) | Bear/crisis period, system protecting capital |
| Medium (100-500) | Mixed conditions, some danger zones |
| Low (<100) | Bull market, RL has room to trade |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | - | Basic RL, 5-action space, PnL rewards |
| 2.0 | Dec 2024 | Gatekeeper model, 2 actions, quality rewards, walk-forward |
| 2.1 | Dec 2024 | Decomposed market regime flags (25 flags), interpretable quality |
| 2.2 | Dec 2024 | Bear/shock detection (32 flags), massive penalties for danger zones |
| **2.3** | **Dec 2024** | **HARD VETO system, 1 trade/day limit, 24h cooldown** |

### V2.3 Changelog

**Problem Identified:**
V2.2 penalties didn't work. Model still lost in bear markets:
- Profitable folds dropped from 57% (V2.1) to 14% (V2.2)
- 2021 bull year: -25% (should have been profitable!)
- Model was confused by conflicting signals

**Root Cause:** Penalties teach RL "this is bad" but RL still tries. Need to **remove the option entirely**.

**Solution: HARD VETO**
- 5 veto flags that OVERRIDE RL decisions completely
- If ANY veto flag active → action FORCED to NO_TRADE
- RL only operates in "allowed zones"

**New Features:**
- `_is_hard_veto()` method checks 5 danger flags
- `HARD_VETO_FLAGS` class constant defines veto conditions
- `hard_vetoes` counter tracks overridden decisions
- Brutal frequency limits: 1 trade/day, 24h cooldown
- Simplified decision reward (no danger handling needed)

**Veto Flags:**
```python
HARD_VETO_FLAGS = [
    'regime_bear_trend',
    'regime_shock',
    'regime_trend_conflict',
    'regime_chop',
    'regime_no_trade_zone',
]
```

**Expected Impact:**
- Bear years (2020, 2022): Near-zero trades (vetoed)
- Bull years: Selective trades in allowed zones only
- Model learns actual edge, not noise

**Files Modified:**
- `src/trading_env.py` - Added hard veto system, frequency limits
- `src/walk_forward.py` - Reports hard_vetoes per fold
- `config.yaml` - max_trades_per_day: 1

### V2.2 Changelog

**Problem Identified:**
Walk-forward validation showed model losing in bear markets:
- 2020 (COVID): -15.8%
- 2022 (Bear): -21.4%

Root cause: Model knew "is there a trend?" but not "is this trend safe to trade?"

**New Features:**
- `regime_bull_trend` - Uptrend detection (safe to trade)
- `regime_bear_trend` - Downtrend detection (AVOID)
- `regime_shock` - Crisis detection (COVID crashes, flash crashes)
- `regime_no_trade_zone` - Hard veto combining all danger signals
- `regime_atr_spike`, `regime_volume_spike`, `regime_large_candles` - Crisis indicators
- Massive quality penalties: bear=-60%, shock=-80%, no_trade=-50%
- Reward penalties: trading in shock=-0.5, bear=-0.4, no_trade=-0.3
- Reward bonus: NOT trading in danger=+0.3

**Expected Impact:**
- Model learns "bear market = stay silent"
- Preserves capital during crashes
- Focuses on bull market opportunities

**Files Modified:**
- `src/indicators.py` - Added 7 new regime flags, updated quality calculation
- `src/trading_env.py` - Added danger zone penalties in reward system

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
- Observation space size increased (48 → 80 features + position)

**Files Modified:**
- `src/indicators.py` - Added regime flags calculation
- `src/trading_env.py` - Updated quality methods, render, info

---

*Last updated: December 2024*
