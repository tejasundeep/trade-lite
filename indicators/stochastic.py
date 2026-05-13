import pandas as pd
from .market_context import get_market_data

def calculate_stochastic(k_period: int = 14, d_period: int = 3) -> dict:
    df = get_market_data()
    if df is None or len(df) < k_period:
        return {"k": 50.0, "d": 50.0}
    
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(window=d_period).mean()
    
    return {
        "k": float(k.iloc[-1]),
        "d": float(d.iloc[-1]),
        "k_series": k.fillna(50.0).tolist(),
        "d_series": d.fillna(50.0).tolist()
    }
