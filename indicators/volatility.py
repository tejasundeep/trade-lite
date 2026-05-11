import pandas as pd
from .market_context import get_market_data

def analyze_volatility() -> dict:
    df = get_market_data()
    if df is None or df.empty or len(df) < 30:
        return {"error": "Insufficient data"}
    period = 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothed ATR (industry standard)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    price = df["close"].iloc[-1]
    return {
        "atr":         float(atr.iloc[-1]),
        "atr_pct":     float(atr.iloc[-1] / price * 100),
        "std_dev":     float(df["close"].rolling(period).std().iloc[-1]),
        "std_dev_pct": float(df["close"].rolling(period).std().iloc[-1] / price * 100),
    }
