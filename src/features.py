import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Trend ---
    df["ma_7"]  = df["Close"].rolling(7).mean()   # 1 week moving average
    df["ma_21"] = df["Close"].rolling(21).mean()  # 1 month moving average
    df["ma_50"] = df["Close"].rolling(50).mean()  # 2 month moving average

    # --- Momentum ---
    df["roc_7"]  = df["Close"].pct_change(7)   # % change over 7 days
    df["roc_21"] = df["Close"].pct_change(21)  # % change over 21 days

    # --- Volatility ---
    df["volatility_7"]  = df["Close"].rolling(7).std()
    df["volatility_21"] = df["Close"].rolling(21).std()

    # --- Volume signal ---
    df["volume_ma_7"] = df["Volume"].rolling(7).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_7"]  # spike detector

    # --- RSI (relative strength index) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Target: next day's closing price ---
    # replace this line:
    df["target"] = df["Close"].pct_change(-1) * -1  # next day % change

    # drop rows with NaN (from rolling windows)
    df.dropna(inplace=True)

    return df

FEATURE_COLS = [
    "Close", "Volume",
    "ma_7", "ma_21", "ma_50",
    "roc_7", "roc_21",
    "volatility_7", "volatility_21",
    "volume_ratio", "rsi"
]