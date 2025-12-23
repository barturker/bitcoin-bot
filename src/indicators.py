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
            # TREND FILTER - Bull/Bear detection (NEW)
            "trend_bull", "ema_alignment_bull", "ema_alignment_bear",
            "trend_strength", "market_regime", "trend_momentum", "ema50_slope"
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
    scaler_path: str = None
) -> tuple:
    df = load_raw_data(csv_path)
    df = resample_to_timeframe(df, timeframe)
    
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    df = add_technical_indicators(df)
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
