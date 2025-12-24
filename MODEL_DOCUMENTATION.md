# Bitcoin Trading Bot - Model Documentation (V3.0 - Simple Trader)

> **Version 3.0** - RL replaced with Logistic Regression. Same risk management, better results.

## Executive Summary

| Aspect | V1 (RL) | V2.x (RL Gatekeeper) | V3.0 (Simple Trader) |
|--------|---------|----------------------|----------------------|
| Decision Model | RL (PPO) | RL (PPO) | **Logistic Regression** |
| Mean Return | Negative | -4.44% | **+5.87%** |
| Profitable Folds | ~30% | 43% | **75%** |
| Max Drawdown | 30%+ | 23.7% | **6.16%** |
| Complexity | High | Very High | **Low** |
| Interpretability | None | Low | **High** |

### V3.0 Key Insight

> **"Feature space has edge (AUC=0.79). RL was the problem, not features."**

Edge test revealed:
- Logistic Regression: AUC = 0.791
- GradientBoosting: AUC = 0.738

Both >> 0.50 (random). **Edge exists in features.**

RL failed because:
- Too complex for thin edge
- Reward engineering is hard
- Overfits to noise

**Solution:** Replace RL with simple classifier. Keep risk management.

---

## Table of Contents

1. [Why Simple Trader?](#why-simple-trader)
2. [Architecture](#architecture)
3. [The Model](#the-model)
4. [Trading Rules](#trading-rules)
5. [Risk Management](#risk-management)
6. [Walk-Forward Results](#walk-forward-results)
7. [Feature Engineering](#feature-engineering)
8. [Configuration](#configuration)
9. [How to Run](#how-to-run)
10. [Version History](#version-history)

---

## Why Simple Trader?

### The RL Journey (V1-V2.3)

| Version | Approach | Result |
|---------|----------|--------|
| V1 | Full RL trader | Overfit, negative returns |
| V2.0 | RL Gatekeeper | Better, still negative |
| V2.1 | + Regime flags | No improvement |
| V2.2 | + Bear/shock penalties | Worse (-6.89%) |
| V2.3 | + Hard veto | Still negative (-4.44%) |

### The Breakthrough: Edge Test

Tested if features contain predictable alpha using simple classifiers:

```
Logistic Regression AUC: 0.791 (+29% over random)
GradientBoosting AUC:    0.738 (+24% over random)
```

**Conclusion:** Features have strong edge. RL was overcomplicating it.

### The Solution

| Component | RL Approach | Simple Approach |
|-----------|-------------|-----------------|
| Decision | PPO neural network | Logistic Regression |
| Training | Reward engineering | Binary classification |
| Inference | Policy network | Probability threshold |
| Complexity | 1000s of parameters | ~80 coefficients |

Same risk management (veto, SL/TP, cooldown). Different decision engine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   V3.0 SIMPLE TRADER ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │  Market Data │                                                   │
│  │  (1H OHLCV)  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Indicators  │───►│   Features   │───►│   REGIME FLAGS       │   │
│  │  (80+ cols)  │    │ (Normalized) │    │   (32 binary flags)  │   │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                     │               │
│  ╔══════════════════════════════════════════════════╧═══════════╗   │
│  ║                    HARD VETO LAYER                           ║   │
│  ║  IF any veto flag active:                                    ║   │
│  ║    • regime_bear_trend                                       ║   │
│  ║    • regime_shock                                            ║   │
│  ║    • regime_trend_conflict                                   ║   │
│  ║    • regime_chop                                             ║   │
│  ║    • regime_no_trade_zone                                    ║   │
│  ║  THEN: NO TRADE (skip this bar)                              ║   │
│  ╚══════════════════════════════════════════════════════════════╝   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 LOGISTIC REGRESSION MODEL                      │ │
│  │                                                                │ │
│  │  Input: 80+ normalized features                                │ │
│  │  Output: P(price gains 2%+ in 24h)                             │ │
│  │                                                                │ │
│  │  Training: Walk-forward (2 years train → 1 year test)          │ │
│  └────────────────────────────────┬───────────────────────────────┘ │
│                                   │                                  │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 CONFIDENCE THRESHOLD                           │ │
│  │                                                                │ │
│  │  IF probability >= 60%:                                        │ │
│  │      Check HTF bias for direction                              │ │
│  │      IF bias exists: ENTER TRADE                               │ │
│  │  ELSE:                                                         │ │
│  │      NO TRADE                                                  │ │
│  └────────────────────────────────┬───────────────────────────────┘ │
│                                   │                                  │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 POSITION MANAGEMENT                            │ │
│  │                                                                │ │
│  │  • Direction: HTF bias (long if bullish, short if bearish)    │ │
│  │  • Stop Loss: 2% (fixed)                                       │ │
│  │  • Take Profit: 4% (fixed, R:R = 1:2)                          │ │
│  │  • Max Duration: 72 hours                                      │ │
│  │  • Cooldown: 24 hours between trades                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Model

### Algorithm: Logistic Regression

**Why Logistic Regression?**

| Reason | Explanation |
|--------|-------------|
| **Proven edge** | AUC = 0.791 in walk-forward test |
| **Interpretable** | Coefficients show feature importance |
| **Fast** | Trains in seconds, infers in microseconds |
| **Robust** | Less prone to overfitting than complex models |
| **Probabilistic** | Outputs confidence, not just yes/no |

### Target Variable

```python
target = "Will price increase by 2%+ in next 24 hours?"

# Calculation:
future_max = rolling_max(close, window=24).shift(-24)
potential_return = (future_max - close) / close
target = (potential_return >= 0.02).astype(int)
```

### Training Process

```python
# For each walk-forward fold:
1. Split: 2 years train → 1 year test
2. Scale features (StandardScaler, fit on train only)
3. Train LogisticRegression(max_iter=1000)
4. Predict probabilities on test set
5. Trade when P >= 0.60 and conditions allow
```

### Model Coefficients

The model learns weights for each feature. Higher absolute weight = more important.

Example interpretation:
- `regime_bull_trend`: Positive weight → increases trade probability
- `regime_bear_trend`: Negative weight → decreases trade probability
- `rsi_14 > 70`: May have negative weight (overbought = risky)

---

## Trading Rules

### Entry Conditions (ALL must be true)

| # | Condition | Rationale |
|---|-----------|-----------|
| 1 | No veto flag active | Avoid dangerous conditions |
| 2 | Model probability >= 60% | High confidence only |
| 3 | HTF bias exists (≠ 0) | Need direction confirmation |
| 4 | Cooldown passed (24h) | Prevent overtrading |
| 5 | Not already in position | One trade at a time |

### Entry Logic

```python
def should_enter(row, probability):
    # Check veto
    if is_vetoed(row):
        return False, None

    # Check confidence
    if probability < 0.60:
        return False, None

    # Check HTF bias
    bias = get_htf_bias(row)
    if bias == 0:
        return False, None

    # Check cooldown
    if bars_since_last_trade < 24:
        return False, None

    direction = 'long' if bias > 0 else 'short'
    return True, direction
```

### Exit Conditions (ANY triggers exit)

| Condition | Action |
|-----------|--------|
| Price hits Stop Loss (2%) | Exit with loss |
| Price hits Take Profit (4%) | Exit with profit |
| Duration >= 72 hours | Exit at market |

### Exit Logic

```python
def check_exit(trade, current_bar):
    if trade.direction == 'long':
        sl_price = entry_price * 0.98  # -2%
        tp_price = entry_price * 1.04  # +4%

        if current_bar.low <= sl_price:
            return 'stop_loss'
        if current_bar.high >= tp_price:
            return 'take_profit'

    # Similar for short...

    if duration >= 72:
        return 'max_duration'

    return None  # Hold
```

---

## Risk Management

### Hard Veto System

These conditions **block all trading**, regardless of model confidence:

| Flag | Condition | Why Block |
|------|-----------|-----------|
| `regime_bear_trend` | ADX>25 + price<EMA50<EMA200 + DI->DI+ | Bear markets destroy accounts |
| `regime_shock` | ATR spike + volume/candle spike | Crisis = unpredictable |
| `regime_trend_conflict` | 1H vs Daily trend disagree | No edge when TFs fight |
| `regime_chop` | Ranging + BB squeeze | Whipsaw zone |
| `regime_no_trade_zone` | Composite of above | Multiple dangers |

### Position Sizing

```python
position_size = capital * 0.20  # 20% of capital per trade
```

With 2% SL, max loss per trade = 0.4% of capital.

### Trade Frequency

| Limit | Value | Rationale |
|-------|-------|-----------|
| Min cooldown | 24 hours | Prevent overtrading |
| Max concurrent | 1 position | Simplicity |

### Transaction Costs

```python
spread_pct = 0.001      # 0.1% spread
commission_pct = 0.001  # 0.1% commission
total_cost = 0.2% per trade  # Entry + exit
```

---

## Walk-Forward Results

### Per-Year Performance

| Year | Return | Max DD | Trades | Win Rate | Notes |
|------|--------|--------|--------|----------|-------|
| 2014 | -7.7% | 10.2% | 109 | 33% | Early data, less reliable |
| 2015 | +5.8% | 3.6% | 55 | 49% | ✓ |
| 2016 | +3.5% | 2.6% | 18 | 56% | ✓ Low activity |
| **2017** | **+27.2%** | 4.6% | 141 | 52% | **Bull run captured!** |
| 2018 | +0.6% | 12.4% | 121 | 39% | Bear year, almost flat ✓ |
| 2019 | +6.2% | 2.8% | 39 | 51% | ✓ |
| **2020** | **+13.4%** | 6.6% | 83 | 53% | **COVID year, profit!** |
| **2021** | **+13.1%** | 5.4% | 127 | 46% | **Bull captured!** |
| 2022 | -14.6% | 15.2% | 81 | 22% | Bear market loss |
| **2023** | **+16.9%** | 2.1% | 54 | 69% | **Recovery captured!** |
| 2024 | +11.2% | 1.3% | 26 | 77% | ✓ |
| 2025 | -5.2% | 7.1% | 56 | 34% | Partial year |

### Aggregate Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Mean Return | **+5.87%** | Positive expectancy ✓ |
| Std Return | 11.56% | Moderate volatility |
| Mean Max DD | **6.16%** | Excellent risk control ✓ |
| Mean Win Rate | 48.4% | With 2:1 R/R, profitable ✓ |
| Profitable Folds | **75% (9/12)** | Strong consistency ✓ |
| Total Trades | 910 | ~76/year average |
| Total Vetoes | 41,220 | System protected from danger |

### Comparison with RL

| Metric | RL (V2.3) | Simple Trader | Improvement |
|--------|-----------|---------------|-------------|
| Mean Return | -4.44% | +5.87% | **+10.3%** |
| Profitable Folds | 43% | 75% | **+32%** |
| Mean Max DD | 23.7% | 6.16% | **-17.5%** |
| Complexity | Very High | Low | **Much simpler** |

---

## Feature Engineering

### Feature Categories (80+ features)

| Category | Count | Examples |
|----------|-------|----------|
| Price | 5 | open, high, low, close, volume |
| Momentum | 10 | RSI, MACD, Stochastic, ROC, CCI |
| Volatility | 8 | BB bands, ATR, volatility |
| Trend | 8 | ADX, DI+, DI-, EMA alignment |
| Multi-Timeframe | 16 | 4H and Daily trend, RSI, ADX |
| Confluence | 5 | MTF alignment, bias, conflict |
| Regime Flags | 32 | All binary 0/1 flags |

### Regime Flags (32 total)

```
Trend:       regime_trending, regime_strong_trend, regime_ranging,
             regime_weak_trend, regime_trend_clarity

Volatility:  regime_volatility_pct, regime_low_volatility,
             regime_high_volatility, regime_normal_volatility,
             regime_bb_squeeze

Momentum:    regime_overbought, regime_oversold, regime_rsi_neutral,
             regime_macd_bullish, regime_macd_increasing,
             regime_strong_momentum

HTF:         regime_htf_bullish, regime_htf_bearish, regime_htf_neutral,
             regime_full_confluence, regime_trend_conflict

Risk:        regime_ideal_setup, regime_chop, regime_dangerous,
             regime_caution

Shock:       regime_atr_spike, regime_volume_spike, regime_large_candles

Direction:   regime_bull_trend, regime_bear_trend, regime_shock,
             regime_no_trade_zone

Score:       regime_quality_score
```

---

## Configuration

### Key Parameters

```yaml
# Model
confidence_threshold: 0.60   # Minimum probability to trade
target_horizon: 24           # Hours to look ahead
target_threshold: 0.02       # 2% gain target

# Risk Management
sl_pct: 0.02                 # 2% stop loss
tp_pct: 0.04                 # 4% take profit (R:R = 1:2)
max_duration: 72             # Max hours in position
min_cooldown: 24             # Hours between trades
position_size_pct: 0.20      # 20% of capital per trade

# Costs
spread_pct: 0.001            # 0.1%
commission_pct: 0.001        # 0.1%
```

### Veto Flags (Hard Rules)

```python
VETO_FLAGS = [
    'regime_bear_trend',
    'regime_shock',
    'regime_trend_conflict',
    'regime_chop',
    'regime_no_trade_zone',
]
```

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Edge Test (Verify feature space has alpha)

```bash
python src/edge_test.py
```

Expected: AUC > 0.55 indicates edge exists.

### Walk-Forward Validation

```bash
python src/simple_trader.py
```

Expected: Mean return > 0%, Profitable folds > 50%.

### Files

| File | Purpose |
|------|---------|
| `src/simple_trader.py` | Main trading system |
| `src/edge_test.py` | Feature space edge test |
| `src/indicators.py` | Feature engineering |
| `src/trading_env.py` | Legacy RL environment |
| `config.yaml` | Configuration |

---

## Version History

| Version | Date | Approach | Mean Return | Profitable Folds |
|---------|------|----------|-------------|------------------|
| 1.0 | Dec 2024 | Full RL | Negative | ~30% |
| 2.0 | Dec 2024 | RL Gatekeeper | Negative | ~40% |
| 2.1 | Dec 2024 | + Regime flags | -1.56% | 57% |
| 2.2 | Dec 2024 | + Bear penalties | -6.89% | 14% |
| 2.3 | Dec 2024 | + Hard veto | -4.44% | 43% |
| **3.0** | **Dec 2024** | **Simple Trader** | **+5.87%** | **75%** |

### V3.0 Changelog

**The Pivot:**
- Discovered features have strong edge (AUC = 0.79)
- RL was overcomplicating the problem
- Replaced RL with Logistic Regression

**Changes:**
- New file: `src/simple_trader.py`
- New file: `src/edge_test.py`
- Model: Logistic Regression (not PPO)
- Training: Binary classification (not RL)
- Decision: Probability threshold (not policy network)

**Results:**
- Mean return: -4.44% → **+5.87%**
- Profitable folds: 43% → **75%**
- Max drawdown: 23.7% → **6.16%**

**Risk Management (Unchanged):**
- Hard veto system
- Fixed SL/TP (2%/4%)
- 24h cooldown
- HTF bias for direction

---

## Key Lessons Learned

### 1. Simple > Complex

RL with complex reward engineering failed. Simple logistic regression worked.

### 2. Test Features First

`edge_test.py` proved features have predictive power before wasting time on model tuning.

### 3. Risk Management is Separate

Same veto/SL/TP system works with any decision model. Decouple decision from risk.

### 4. Walk-Forward is Essential

Out-of-sample testing catches overfitting. All results are walk-forward validated.

### 5. Edge is Thin

+5.87% mean return is not huge. Requires discipline, low costs, and consistent execution.

---

*Last updated: December 2024*
