import numpy as np
import pandas as pd
from .market_context import get_market_data

def analyze_trend_strength() -> dict:
    raw = get_market_data()
    if raw is None or raw.empty or len(raw) < 30:
        return {"error": "Insufficient data"}
    
    df = raw.copy()
    period = 14
    
    # 1. True Range
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = h_l.combine(h_pc, max).combine(l_pc, max)
    
    # 2. Directional Movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    p_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    n_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # 3. Wilder's Smoothing for ADX components
    smooth_tr = tr.ewm(alpha=1/period, adjust=False).mean()
    smooth_p_dm = pd.Series(p_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    smooth_n_dm = pd.Series(n_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    
    p_di = 100 * (smooth_p_dm / (smooth_tr + 1e-9))
    n_di = 100 * (smooth_n_dm / (smooth_tr + 1e-9))
    
    dx = 100 * (p_di - n_di).abs() / (p_di + n_di + 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    latest_adx = float(adx.iloc[-1])
    latest_p_di = float(p_di.iloc[-1])
    latest_n_di = float(n_di.iloc[-1])
    
    strength = "Strong Trend" if latest_adx > 25 else "Weak or No Trend"
    direction = "Bullish" if latest_p_di > latest_n_di else "Bearish"
    
    return {
        "adx": latest_adx,
        "p_di": latest_p_di,
        "n_di": latest_n_di,
        "trend_strength": strength,
        "trend_direction": direction
    }
