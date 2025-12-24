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
            "htf_bias", "trend_conflict"
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


def normalize_features(df: pd.DataFrame, scaler: StandardScaler = None,
                       fit: bool = True, save_path: str = None) -> tuple:
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        normalized_values = scaler.fit_transform(df.values)
        if save_path:
            joblib.dump(scaler, save_path)
    else:
        normalized_values = scaler.transform(df.values)
    
    normalized_df = pd.DataFrame(
        normalized_values,
        index=df.index,
        columns=df.columns
    )
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

    df = df.dropna()
    feature_df = prepare_features(df)
    
    if normalize:
        if scaler is None and scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            feature_df, scaler = normalize_features(feature_df, scaler, fit=False)
        else:
            feature_df, scaler = normalize_features(
                feature_df, scaler, fit=True, save_path=scaler_path
            )
    
    return df, feature_df, scaler
