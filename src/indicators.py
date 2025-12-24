"""
Data loading and technical indicators for Bitcoin trading bot.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.columns = df.columns.str.lower()
    return df


def resample_to_timeframe(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return resampled


def calculate_htf_indicators(df: pd.DataFrame, htf: str, prefix: str) -> pd.DataFrame:
    """
    Calculate higher timeframe indicators and merge back to base timeframe.

    Args:
        df: Base timeframe DataFrame with OHLCV
        htf: Higher timeframe string (e.g., '4h', '1D')
        prefix: Prefix for column names (e.g., 'htf_4h_')

    Returns:
        DataFrame with HTF indicators aligned to base timeframe
    """
    # Resample to higher timeframe
    htf_df = df.resample(htf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    # Calculate HTF indicators
    htf_df[f"{prefix}ema_21"] = ta.ema(htf_df["close"], length=21)
    htf_df[f"{prefix}ema_50"] = ta.ema(htf_df["close"], length=50)
    htf_df[f"{prefix}ema_200"] = ta.ema(htf_df["close"], length=200)
    htf_df[f"{prefix}rsi"] = ta.rsi(htf_df["close"], length=14)
    htf_df[f"{prefix}atr"] = ta.atr(htf_df["high"], htf_df["low"], htf_df["close"], length=14)

    # ADX for trend strength
    htf_adx = ta.adx(htf_df["high"], htf_df["low"], htf_df["close"], length=14)
    htf_df[f"{prefix}adx"] = htf_adx["ADX_14"]

    # Trend direction: 1 = bullish, -1 = bearish, 0 = neutral
    htf_df[f"{prefix}trend"] = np.where(
        htf_df["close"] > htf_df[f"{prefix}ema_50"],
        np.where(htf_df[f"{prefix}ema_50"] > htf_df[f"{prefix}ema_200"], 1, 0.5),
        np.where(htf_df[f"{prefix}ema_50"] < htf_df[f"{prefix}ema_200"], -1, -0.5)
    )

    # Price position relative to HTF EMAs
    htf_df[f"{prefix}price_vs_ema21"] = (htf_df["close"] / htf_df[f"{prefix}ema_21"] - 1) * 100
    htf_df[f"{prefix}price_vs_ema50"] = (htf_df["close"] / htf_df[f"{prefix}ema_50"] - 1) * 100

    # EMA slope (momentum)
    htf_df[f"{prefix}ema21_slope"] = htf_df[f"{prefix}ema_21"].pct_change(periods=3) * 100

    # Select only the indicator columns (not OHLCV)
    indicator_cols = [col for col in htf_df.columns if col.startswith(prefix)]
    htf_indicators = htf_df[indicator_cols]

    # Forward-fill to base timeframe (no look-ahead bias)
    htf_aligned = htf_indicators.reindex(df.index, method='ffill')

    return htf_aligned


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)
    
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    
    bbands = ta.bbands(df["close"], length=20, std=2)
    bb_cols = bbands.columns.tolist()
    bb_upper_col = [c for c in bb_cols if c.startswith("BBU")][0]
    bb_middle_col = [c for c in bb_cols if c.startswith("BBM")][0]
    bb_lower_col = [c for c in bb_cols if c.startswith("BBL")][0]
    df["bb_upper"] = bbands[bb_upper_col]
    df["bb_middle"] = bbands[bb_middle_col]
    df["bb_lower"] = bbands[bb_lower_col]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = adx["ADX_14"]
    df["di_plus"] = adx["DMP_14"]
    df["di_minus"] = adx["DMN_14"]
    
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["roc_10"] = ta.roc(df["close"], length=10)
    df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["returns"].rolling(window=20).std()
    
    df["price_to_ema12"] = df["close"] / df["ema_12"] - 1
    df["price_to_ema26"] = df["close"] / df["ema_26"] - 1
    df["price_to_ema50"] = df["close"] / df["ema_50"] - 1
    
    df["ema_trend"] = (df["ema_12"] > df["ema_26"]).astype(int)
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

    # ============ TREND FILTER - Bull/Bear Market Detection ============

    # Price position relative to EMA 200 (long-term trend)
    df["price_to_ema200"] = df["close"] / df["ema_200"] - 1

    # Trend direction: 1 = bull, 0 = bear
    df["trend_bull"] = (df["close"] > df["ema_200"]).astype(int)

    # EMA alignment (stronger trend signal)
    # Bull: EMA12 > EMA26 > EMA50 > EMA200
    df["ema_alignment_bull"] = (
        (df["ema_12"] > df["ema_26"]) &
        (df["ema_26"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    ).astype(int)

    # Bear: EMA12 < EMA26 < EMA50 < EMA200
    df["ema_alignment_bear"] = (
        (df["ema_12"] < df["ema_26"]) &
        (df["ema_26"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"])
    ).astype(int)

    # Trend strength (distance from EMA200, normalized by ATR)
    df["trend_strength"] = (df["close"] - df["ema_200"]) / df["atr_14"]
    df["trend_strength"] = df["trend_strength"].clip(-10, 10)  # Clip extremes

    # Market regime: 1 = strong bull, 0.5 = weak bull, -0.5 = weak bear, -1 = strong bear
    conditions = [
        df["ema_alignment_bull"] == 1,  # Strong bull
        df["trend_bull"] == 1,           # Weak bull
        df["ema_alignment_bear"] == 1,  # Strong bear
    ]
    choices = [1.0, 0.5, -1.0]
    df["market_regime"] = np.select(conditions, choices, default=-0.5)

    # Trend momentum (rate of change of EMA50)
    df["trend_momentum"] = df["ema_50"].pct_change(periods=10) * 100
    df["trend_momentum"] = df["trend_momentum"].clip(-5, 5)

    # Higher timeframe trend (20-period slope of EMA50)
    df["ema50_slope"] = (df["ema_50"] - df["ema_50"].shift(20)) / df["ema_50"].shift(20) * 100
    df["ema50_slope"] = df["ema50_slope"].clip(-10, 10)

    return df


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add higher timeframe (4H, Daily) features to the base timeframe data.
    This gives the model context about the bigger picture.
    """
    df = df.copy()

    # 4-Hour timeframe features
    htf_4h = calculate_htf_indicators(df, "4h", "htf_4h_")
    for col in htf_4h.columns:
        df[col] = htf_4h[col]

    # Daily timeframe features
    htf_1d = calculate_htf_indicators(df, "1D", "htf_1d_")
    for col in htf_1d.columns:
        df[col] = htf_1d[col]

    # ============ MULTI-TIMEFRAME CONFLUENCE ============

    # Trend alignment score: Do all timeframes agree?
    # +1 for each bullish TF, -1 for each bearish TF
    df["mtf_trend_alignment"] = (
        np.sign(df["market_regime"]) +  # 1h trend
        np.sign(df["htf_4h_trend"]) +    # 4h trend
        np.sign(df["htf_1d_trend"])      # Daily trend
    ) / 3  # Normalize to [-1, 1]

    # Strong confluence: All timeframes agree
    df["mtf_strong_bull"] = (
        (df["market_regime"] > 0) &
        (df["htf_4h_trend"] > 0) &
        (df["htf_1d_trend"] > 0)
    ).astype(int)

    df["mtf_strong_bear"] = (
        (df["market_regime"] < 0) &
        (df["htf_4h_trend"] < 0) &
        (df["htf_1d_trend"] < 0)
    ).astype(int)

    # HTF support/resistance: Price vs Daily EMA50
    df["htf_bias"] = np.where(
        df["htf_1d_trend"] > 0, 1,  # Daily bullish = long bias
        np.where(df["htf_1d_trend"] < 0, -1, 0)  # Daily bearish = short bias
    )

    # Trend conflict warning (1h vs Daily disagree)
    df["trend_conflict"] = (
        np.sign(df["market_regime"]) != np.sign(df["htf_1d_trend"])
    ).astype(int)

    return df


def add_market_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate explicit market regime flags for RL interpretability.

    These flags use RAW (unnormalized) indicator values with meaningful
    technical analysis thresholds. They remain as 0/1 values and are NOT
    normalized, preserving their semantic meaning for the RL agent.

    Prefix: 'regime_' - columns with this prefix skip normalization

    Categories:
        - Trend: is the market trending or ranging?
        - Volatility: low/normal/high based on ATR percentile
        - Momentum: overbought/oversold, MACD direction
        - HTF Alignment: do timeframes agree?
        - Risk Flags: conflict, squeeze, dangerous conditions

    This solves the "single scalar quality" problem by giving the RL model
    explicit flags it can learn to interpret individually.
    """
    df = df.copy()

    # ============ TREND REGIME ============
    # ADX interpretation (standard TA thresholds):
    # < 20: no trend (ranging)
    # 20-25: weak trend forming
    # 25-40: trending
    # > 40: strong trend
    df['regime_trending'] = (df['adx_14'] > 25).astype(np.float32)
    df['regime_strong_trend'] = (df['adx_14'] > 40).astype(np.float32)
    df['regime_ranging'] = (df['adx_14'] < 20).astype(np.float32)
    df['regime_weak_trend'] = (
        (df['adx_14'] >= 20) & (df['adx_14'] <= 25)
    ).astype(np.float32)

    # Trend clarity: |DI+ - DI-| / (DI+ + DI-)
    # High value = clear directional trend, low = choppy
    di_diff = np.abs(df['di_plus'] - df['di_minus'])
    di_sum = df['di_plus'] + df['di_minus']
    df['regime_trend_clarity'] = np.where(
        di_sum > 0,
        (di_diff / di_sum).clip(0, 1),
        0
    ).astype(np.float32)

    # ============ VOLATILITY REGIME ============
    # ATR percentile over rolling window (168 = 1 week for 1h data)
    window = min(168, max(24, len(df) // 4))
    atr_pct = df['atr_14'].rolling(window=window, min_periods=24).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    ).fillna(0.5)

    df['regime_volatility_pct'] = atr_pct.astype(np.float32)
    df['regime_low_volatility'] = (atr_pct < 0.25).astype(np.float32)
    df['regime_high_volatility'] = (atr_pct > 0.75).astype(np.float32)
    df['regime_normal_volatility'] = (
        (atr_pct >= 0.25) & (atr_pct <= 0.75)
    ).astype(np.float32)

    # Bollinger Band squeeze: consolidation before breakout
    bb_width_pct = df['bb_width'].rolling(window=window, min_periods=24).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    ).fillna(0.5)
    df['regime_bb_squeeze'] = (bb_width_pct < 0.2).astype(np.float32)

    # ============ MOMENTUM REGIME ============
    df['regime_overbought'] = (df['rsi_14'] > 70).astype(np.float32)
    df['regime_oversold'] = (df['rsi_14'] < 30).astype(np.float32)
    df['regime_rsi_neutral'] = (
        (df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)
    ).astype(np.float32)

    # MACD momentum state
    df['regime_macd_bullish'] = (df['macd_hist'] > 0).astype(np.float32)
    df['regime_macd_increasing'] = (
        df['macd_hist'] > df['macd_hist'].shift(1)
    ).fillna(0).astype(np.float32)

    # Momentum strength (relative MACD histogram)
    macd_strength = np.abs(df['macd_hist']) / df['close'] * 100
    macd_pct = macd_strength.rolling(window=window, min_periods=24).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    ).fillna(0.5)
    df['regime_strong_momentum'] = (macd_pct > 0.7).astype(np.float32)

    # ============ HTF ALIGNMENT ============
    # Categorical flags for HTF direction
    df['regime_htf_bullish'] = (df['mtf_trend_alignment'] > 0.5).astype(np.float32)
    df['regime_htf_bearish'] = (df['mtf_trend_alignment'] < -0.5).astype(np.float32)
    df['regime_htf_neutral'] = (
        (df['mtf_trend_alignment'] >= -0.5) & (df['mtf_trend_alignment'] <= 0.5)
    ).astype(np.float32)

    # Full confluence: ALL 3 timeframes aligned
    df['regime_full_confluence'] = (
        (df['mtf_strong_bull'] == 1) | (df['mtf_strong_bear'] == 1)
    ).astype(np.float32)

    # Trend conflict (1H vs Daily disagree) - critical risk flag
    df['regime_trend_conflict'] = df['trend_conflict'].astype(np.float32)

    # ============ COMPOSITE RISK FLAGS ============
    # Ideal setup: ALL conditions favorable
    df['regime_ideal_setup'] = (
        (df['regime_trending'] == 1) &
        (df['regime_trend_conflict'] == 0) &
        ((df['regime_htf_bullish'] == 1) | (df['regime_htf_bearish'] == 1)) &
        (df['regime_normal_volatility'] == 1)
    ).astype(np.float32)

    # Chop zone: ranging + BB squeeze (AVOID trading)
    df['regime_chop'] = (
        (df['regime_ranging'] == 1) &
        (df['regime_bb_squeeze'] == 1)
    ).astype(np.float32)

    # Dangerous conditions: conflict OR (high vol + ranging)
    df['regime_dangerous'] = (
        (df['regime_trend_conflict'] == 1) |
        ((df['regime_high_volatility'] == 1) & (df['regime_ranging'] == 1))
    ).astype(np.float32)

    # Caution: weak trend OR extreme RSI without HTF support
    df['regime_caution'] = (
        (df['regime_weak_trend'] == 1) |
        (((df['regime_overbought'] == 1) | (df['regime_oversold'] == 1)) &
         (df['regime_htf_neutral'] == 1))
    ).astype(np.float32)

    # ============ COMPOSITE QUALITY SCORE ============
    # Pre-computed quality for reward calculation
    quality = np.zeros(len(df), dtype=np.float32)

    # Positive contributions
    quality += df['regime_trending'].values * 0.20
    quality += df['regime_strong_trend'].values * 0.10
    quality += df['regime_full_confluence'].values * 0.25
    quality += df['regime_normal_volatility'].values * 0.10
    quality += df['regime_strong_momentum'].values * 0.05
    quality += df['regime_trend_clarity'].values * 0.10

    # Negative contributions
    quality -= df['regime_ranging'].values * 0.25
    quality -= df['regime_trend_conflict'].values * 0.40
    quality -= df['regime_chop'].values * 0.20
    quality -= df['regime_dangerous'].values * 0.15
    quality -= df['regime_bb_squeeze'].values * 0.05

    df['regime_quality_score'] = quality.clip(-1, 1)

    return df


def prepare_features(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = [
            # Price data
            "open", "high", "low", "close", "volume",
            # Momentum indicators
            "rsi_14", "macd", "macd_signal", "stoch_k", "stoch_d", "roc_10", "cci_20",
            # Volatility
            "bb_upper", "bb_middle", "bb_lower", "atr_14", "volatility",
            # Trend indicators
            "adx_14", "ema_trend", "returns",
            # Price relative to EMAs
            "price_to_ema12", "price_to_ema26", "price_to_ema50", "price_to_ema200",
            # TREND FILTER - Bull/Bear detection
            "trend_bull", "ema_alignment_bull", "ema_alignment_bear",
            "trend_strength", "market_regime", "trend_momentum", "ema50_slope",
            # ============ MULTI-TIMEFRAME FEATURES ============
            # 4H timeframe
            "htf_4h_trend", "htf_4h_rsi", "htf_4h_adx",
            "htf_4h_price_vs_ema21", "htf_4h_price_vs_ema50", "htf_4h_ema21_slope",
            # Daily timeframe
            "htf_1d_trend", "htf_1d_rsi", "htf_1d_adx",
            "htf_1d_price_vs_ema21", "htf_1d_price_vs_ema50", "htf_1d_ema21_slope",
            # Confluence signals
            "mtf_trend_alignment", "mtf_strong_bull", "mtf_strong_bear",
            "htf_bias", "trend_conflict",
            # ============ REGIME FLAGS (NOT NORMALIZED) ============
            # These provide explicit, interpretable signals for the RL agent
            # Trend regime
            "regime_trending", "regime_strong_trend", "regime_ranging",
            "regime_weak_trend", "regime_trend_clarity",
            # Volatility regime
            "regime_volatility_pct", "regime_low_volatility",
            "regime_high_volatility", "regime_normal_volatility", "regime_bb_squeeze",
            # Momentum regime
            "regime_overbought", "regime_oversold", "regime_rsi_neutral",
            "regime_macd_bullish", "regime_macd_increasing", "regime_strong_momentum",
            # HTF alignment
            "regime_htf_bullish", "regime_htf_bearish", "regime_htf_neutral",
            "regime_full_confluence", "regime_trend_conflict",
            # Composite risk flags
            "regime_ideal_setup", "regime_chop", "regime_dangerous", "regime_caution",
            # Quality score (composite)
            "regime_quality_score"
        ]
    
    feature_columns = [col for col in feature_columns if col in df.columns]
    feature_df = df[feature_columns].copy()
    
    # Clean infinity values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill()
    
    # Clip extreme values
    for col in feature_df.columns:
        col_std = feature_df[col].std()
        col_mean = feature_df[col].mean()
        if col_std > 0:
            feature_df[col] = feature_df[col].clip(
                lower=col_mean - 10 * col_std,
                upper=col_mean + 10 * col_std
            )
    
    feature_df = feature_df.fillna(0)
    return feature_df


def normalize_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True,
    save_path: str = None,
    skip_prefix: str = 'regime_'
) -> tuple:
    """
    Normalize features using StandardScaler.

    Columns starting with skip_prefix are NOT normalized (e.g., regime flags).
    This preserves their semantic meaning (0/1 binary values) for the RL agent.

    Args:
        df: Feature DataFrame
        scaler: Optional pre-fitted scaler
        fit: Whether to fit the scaler (True for training, False for inference)
        save_path: Path to save the fitted scaler
        skip_prefix: Column prefix to skip normalization (default: 'regime_')

    Returns:
        Tuple of (normalized DataFrame, scaler)
    """
    # Separate columns into normalize vs skip
    skip_cols = [col for col in df.columns if col.startswith(skip_prefix)]
    normalize_cols = [col for col in df.columns if not col.startswith(skip_prefix)]

    if scaler is None:
        scaler = StandardScaler()

    # Normalize only non-regime columns
    if normalize_cols:
        normalize_df = df[normalize_cols]

        if fit:
            normalized_values = scaler.fit_transform(normalize_df.values)
            if save_path:
                # Save scaler with column metadata for validation
                scaler_data = {
                    'scaler': scaler,
                    'columns': normalize_cols
                }
                joblib.dump(scaler_data, save_path)
        else:
            normalized_values = scaler.transform(normalize_df.values)

        normalized_df = pd.DataFrame(
            normalized_values,
            index=df.index,
            columns=normalize_cols
        )
    else:
        normalized_df = pd.DataFrame(index=df.index)

    # Add regime columns unchanged (preserve 0/1 values)
    if skip_cols:
        regime_df = df[skip_cols].copy()
        normalized_df = pd.concat([normalized_df, regime_df], axis=1)

    return normalized_df, scaler


def load_and_preprocess_data(
    csv_path: str,
    timeframe: str = "1h",
    start_date: str = None,
    end_date: str = None,
    normalize: bool = True,
    scaler: StandardScaler = None,
    scaler_path: str = None,
    use_multi_timeframe: bool = True
) -> tuple:
    """
    Load and preprocess trading data with technical indicators and regime flags.

    Pipeline:
        1. Load raw CSV data
        2. Resample to target timeframe
        3. Add technical indicators (RSI, MACD, ADX, etc.)
        4. Add multi-timeframe features (4H, Daily)
        5. Add market regime flags (trend, volatility, momentum, etc.)
        6. Prepare and normalize features

    Args:
        csv_path: Path to CSV file with OHLCV data
        timeframe: Target timeframe (e.g., '1h', '4h')
        start_date: Optional start date filter
        end_date: Optional end date filter
        normalize: Whether to normalize features
        scaler: Optional pre-fitted scaler
        scaler_path: Path to save/load scaler
        use_multi_timeframe: Whether to add 4H/Daily features

    Returns:
        Tuple of (OHLCV DataFrame, feature DataFrame, scaler)
    """
    df = load_raw_data(csv_path)
    df = resample_to_timeframe(df, timeframe)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    df = add_technical_indicators(df)

    # Add multi-timeframe features (4H, Daily context)
    if use_multi_timeframe:
        df = add_multi_timeframe_features(df)

    # Add market regime flags (decomposed quality signals)
    # These use raw indicator values with meaningful TA thresholds
    df = add_market_regime_flags(df)

    df = df.dropna()
    feature_df = prepare_features(df)

    if normalize:
        if scaler is None and scaler_path and os.path.exists(scaler_path):
            # Load scaler with backward compatibility
            loaded = joblib.load(scaler_path)
            if isinstance(loaded, dict):
                scaler = loaded['scaler']
                expected_cols = loaded.get('columns', None)
                # Validate column match (warning only, don't fail)
                if expected_cols:
                    current_cols = [c for c in feature_df.columns
                                    if not c.startswith('regime_')]
                    if set(current_cols) != set(expected_cols):
                        print(f"Warning: Scaler columns mismatch. "
                              f"Expected {len(expected_cols)}, got {len(current_cols)}. "
                              f"Consider re-training with new features.")
            else:
                # Old format - scaler only (pre-regime flags)
                scaler = loaded
                print("Warning: Loading old scaler format. "
                      "Re-train to use new regime flags.")
            feature_df, scaler = normalize_features(feature_df, scaler, fit=False)
        else:
            feature_df, scaler = normalize_features(
                feature_df, scaler, fit=True, save_path=scaler_path
            )

    return df, feature_df, scaler
