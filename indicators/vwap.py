import numpy as np
from .market_context import get_market_data

def calculate_vwap_analysis() -> dict:
    raw = get_market_data()
    if raw is None or raw.empty or len(raw) < 20:
        return {"error": "Insufficient data"}
    
    df = raw.copy()
    window = 24
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']
    
    rolling_tp_vol = tp_vol.rolling(window=window).sum()
    rolling_vol = df['volume'].rolling(window=window).sum()
    
    vwap = rolling_tp_vol / rolling_vol
    
    var = ((typical_price - vwap)**2).rolling(window=window).mean()
    vwap_std = np.sqrt(var)
    
    latest_px   = df['close'].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    latest_std  = vwap_std.iloc[-1]
    
    # Epsilon protection for zero-volatility environments
    denominator = latest_std if latest_std > 1e-9 else 1e-9
    z_score = float((latest_px - latest_vwap) / denominator)
    
    bias = "Fair Value"
    if z_score > 2.0: bias = "Overextended (Bearish Reversion)"
    elif z_score < -2.0: bias = "Undervalued (Bullish Reversion)"
    
    return {
        "vwap": float(latest_vwap),
        "z_score": z_score,
        "bias": bias,
        "upper_band_2": float(latest_vwap + 2 * latest_std),
        "lower_band_2": float(latest_vwap - 2 * latest_std)
    }
