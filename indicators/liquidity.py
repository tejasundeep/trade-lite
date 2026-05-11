import pandas as pd
from .market_context import get_market_data

def calculate_liquidity_map() -> dict:
    df = get_market_data()
    if df is None or df.empty or len(df) < 100: return {"error": "Insufficient data"}
    lookback = 20
    swing_highs = df[df['high'] == df['high'].rolling(lookback, center=True).max()]['high']
    swing_lows = df[df['low'] == df['low'].rolling(lookback, center=True).min()]['low']
    price_bins = pd.cut(df['close'], bins=50)
    vol_profile = df.groupby(price_bins)['volume'].sum()
    high_vol_nodes = vol_profile.nlargest(5)
    current_price = df.iloc[-1]['close']
    magnets = [{"price": float(i.mid), "strength": float(v)} for i, v in high_vol_nodes.items() if abs(i.mid - current_price) / current_price < 0.05]
    buy_liq = swing_highs[swing_highs > current_price].tail(3).tolist()
    sell_liq = swing_lows[swing_lows < current_price].tail(3).tolist()
    return {"buy_side_liquidity_targets": buy_liq, "sell_side_liquidity_targets": sell_liq, "high_volume_nodes": magnets}
