import pandas as pd
from .market_context import get_market_data

def calculate_bollinger(period: int = 20, std_dev: int = 2) -> dict:
    df = get_market_data()
    if df is None or len(df) < period: return {"mid": 0, "upper": 0, "lower": 0}
    mid = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return {"mid": float(mid.iloc[-1]), "upper": float(upper.iloc[-1]), "lower": float(lower.iloc[-1])}
