import numpy as np
import pandas as pd
from .market_context import get_market_data, is_backtest
from trading.adapters.crypto import CCXTCryptoAdapter

def analyze_order_flow(symbol: str = "BTC/USDT", adapter: CCXTCryptoAdapter = None, streamer = None) -> dict:
    if is_backtest():
        return {"summary": "Order flow bypassed in backtest", "poc": 0.0, "vah": 0.0, "val": 0.0}

    raw = get_market_data()
    if raw is None or raw.empty: return {"error": "No data"}
    df = raw.copy()

    # 1. Real Trade-Based Analysis (Zero Latency)
    delta, cvd, imbalance = 0.0, 0.0, "Neutral"
    poc, vah, val = 0.0, 0.0, 0.0
    absorption, iceberg = "None", "None"

    if streamer:
        trades_df = streamer.get_trades_df(symbol, limit=2000)
        if not trades_df.empty:
            # A. True Volume Profile from Trades
            prices = trades_df['price'].values
            volumes = trades_df['amount'].values
            hist, edges = np.histogram(prices, bins=30, weights=volumes)
            poc_idx = int(np.argmax(hist))
            poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
            
            # B. Iceberg Detection (Large volume at single tick)
            # Group by exact price (rounded to tick size)
            tick_size = (edges[1] - edges[0]) / 2
            trades_df['tick'] = trades_df['price'].apply(lambda x: round(x / tick_size) * tick_size)
            tick_groups = trades_df.groupby('tick')['amount'].sum().sort_values(ascending=False)
            if not tick_groups.empty:
                top_tick_vol = tick_groups.iloc[0]
                avg_tick_vol = tick_groups.mean()
                if top_tick_vol > avg_tick_vol * 4:
                    iceberg = f"Iceberg Detected @ {tick_groups.index[0]:.2f}"

            # C. Value Area (70%)
            target = hist.sum() * 0.70
            cur_vol, l, r = hist[poc_idx], poc_idx, poc_idx
            while cur_vol < target and (l > 0 or r < 29):
                v_l = hist[l-1] if l > 0 else 0
                v_r = hist[r+1] if r < 29 else 0
                if v_l >= v_r and l > 0: l -= 1; cur_vol += v_l
                elif r < 29: r += 1; cur_vol += v_r
                else: break
            vah, val = edges[r+1], edges[l]

            # D. Aggressive Imbalance & CVD
            trades_df['side_val'] = trades_df['side'].map({'buy': 1, 'sell': -1})
            delta = (trades_df['amount'] * trades_df['side_val']).sum()
            cvd = (trades_df['amount'] * trades_df['side_val']).sum()
            
            buy_vol = trades_df[trades_df['side'] == 'buy']['amount'].sum()
            sell_vol = trades_df[trades_df['side'] == 'sell']['amount'].sum()
            ratio = buy_vol / (sell_vol + 1e-9)
            if ratio > 2.5: imbalance = "Extreme Bullish Imbalance"
            elif ratio < 0.4: imbalance = "Extreme Bearish Imbalance"

            # E. Absorption (Elite Signal)
            if delta > (buy_vol + sell_vol) * 0.3 and df['close'].iloc[-1] <= df['open'].iloc[-1]:
                absorption = "Bullish Absorption"
            elif delta < -(buy_vol + sell_vol) * 0.3 and df['close'].iloc[-1] >= df['open'].iloc[-1]:
                absorption = "Bearish Absorption"

    return {
        "poc": float(poc), "vah": float(vah), "val": float(val),
        "delta": float(delta), "cvd": float(cvd),
        "imbalance": imbalance,
        "absorption": absorption,
        "iceberg": iceberg,
        "summary": f"POC: {poc:.2f} | Ice: {iceberg != 'None'}"
    }


def analyze_open_interest(symbol: str = "BTC/USDT", adapter: CCXTCryptoAdapter = None) -> dict:
    if is_backtest():
        return {"summary": "OI bypassed in backtest"}
    if adapter is None:
        adapter = CCXTCryptoAdapter()
    df = get_market_data()
    if df is None or df.empty:
        return {"error": "No data"}
    try:
        history = adapter.get_open_interest_history(symbol, limit=10)
        if not history:
            return {"error": "OI history unavailable"}
        cur  = float(history[-1]["openInterestAmount"])
        prev = float(history[-2]["openInterestAmount"])
        oi_chg  = (cur - prev) / prev
        px_chg  = df.iloc[-1]["close"] - df.iloc[-2]["close"]
        if   px_chg > 0 and oi_chg >  0.02: bias = "Aggressive Bullish"
        elif px_chg < 0 and oi_chg >  0.02: bias = "Aggressive Bearish"
        elif px_chg > 0 and oi_chg < -0.02: bias = "Short Covering"
        elif px_chg < 0 and oi_chg < -0.02: bias = "Long Liquidation"
        else:                                bias = "Neutral"
        return {"current_oi": cur, "oi_change_pct": round(oi_chg * 100, 2), "oi_bias": bias}
    except Exception as e:
        return {"error": str(e)}
