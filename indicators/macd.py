import pandas as pd
from .market_context import get_market_data

def calculate_macd() -> dict:
    df = get_market_data()
    if df is None or len(df) < 26: return {"macd": 0, "signal": 0, "hist": 0}
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return {"macd": float(macd.iloc[-1]), "signal": float(signal.iloc[-1]), "hist": float(hist.iloc[-1])}
