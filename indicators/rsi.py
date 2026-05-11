import pandas as pd
from .market_context import get_market_data

def calculate_rsi(period: int = 14) -> float:
    df = get_market_data()
    if df is None or len(df) < period: return 50.0
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])
