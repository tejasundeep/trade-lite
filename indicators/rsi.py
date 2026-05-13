import pandas as pd
from .market_context import get_market_data

def calculate_rsi(period: int = 14) -> dict:
    df = get_market_data()
    if df is None or len(df) < period: 
        return {"value": 50.0, "series": [50.0] * 5}
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return {
        "value": float(rsi.iloc[-1]),
        "series": rsi.fillna(50.0).tolist()
    }
