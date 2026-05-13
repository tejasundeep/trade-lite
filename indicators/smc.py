from .market_context import get_market_data

def analyze_smc_structure(htf_structure: str = "Neutral", df_override=None) -> dict:
    df = df_override if df_override is not None else get_market_data()
    if df is None or df.empty or len(df) < 50:
        return {"error": "Insufficient market data for SMC"}

    # 1. Improved Pivot Detection (Fixed Look-Ahead Bias)
    def get_swings(data, window=5):
        highs = data['high'].values
        lows = data['low'].values
        swings_h, swings_l = [], []
        
        # ELITE FIX: Swings can only be identified AFTER they are confirmed
        # i must be far enough back that [i+window] is STILL in the past relative to the latest candle
        for i in range(window, len(data) - window):
            # Check if index i is a local peak/trough
            if all(highs[i] >= highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
                swings_h.append({"price": highs[i], "index": i})
            if all(lows[i] <= lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
                swings_l.append({"price": lows[i], "index": i})
        return swings_h, swings_l

    swings_h, swings_l = get_swings(df)
    if not swings_h or not swings_l: return {"structure": "Neutral", "summary": "No swings"}

    last_sh, last_sl = swings_h[-1]["price"], swings_l[-1]["price"]
    price = df.iloc[-1]["close"]

    # 2. Fair Value Gaps (FVG)
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG: High of candle 1 < Low of candle 3
        if df.iloc[i-2]["high"] < df.iloc[i]["low"]:
            fvgs.append({"type": "bullish", "top": df.iloc[i]["low"], "bottom": df.iloc[i-2]["high"], "index": i-1})
        # Bearish FVG: Low of candle 1 > High of candle 3
        elif df.iloc[i-2]["low"] > df.iloc[i]["high"]:
            fvgs.append({"type": "bearish", "top": df.iloc[i-2]["low"], "bottom": df.iloc[i]["high"], "index": i-1})

    # 3. Order Blocks (OB) - Last opposite candle before BOS
    order_blocks = []
    # Simplified BOS check for OB identification
    for i in range(5, len(df)-1):
        # Bullish BOS: Close above last swing high
        if df.iloc[i]["close"] > last_sh:
            # Find last bearish candle before the move
            for j in range(i, i-10, -1):
                if df.iloc[j]["close"] < df.iloc[j]["open"]:
                    order_blocks.append({"type": "bullish", "top": df.iloc[j]["high"], "bottom": df.iloc[j]["low"], "index": j})
                    break
        # Bearish BOS: Close below last swing low
        elif df.iloc[i]["close"] < last_sl:
            for j in range(i, i-10, -1):
                if df.iloc[j]["close"] > df.iloc[j]["open"]:
                    order_blocks.append({"type": "bearish", "top": df.iloc[j]["high"], "bottom": df.iloc[j]["low"], "index": j})
                    break

    # 4. Market Structure & SFP
    structure = "Bullish" if price > last_sh else "Bearish" if price < last_sl else "Ranging"
    sweep_h = any(df.iloc[i]["high"] > last_sh and df.iloc[i]["close"] <= last_sh for i in range(-3, 0))
    sweep_l = any(df.iloc[i]["low"]  < last_sl and df.iloc[i]["close"] >= last_sl for i in range(-3, 0))

    # Dealing range & OTE
    rng_high, rng_low = swings_h[-1]["price"], swings_l[-1]["price"]
    eq = (rng_high + rng_low) / 2
    ote_705 = rng_low + (rng_high - rng_low) * 0.705 if structure == "Bullish" else rng_high - (rng_high - rng_low) * 0.705

    return {
        "structure": structure,
        "mtf_aligned": (structure == htf_structure),
        "zone": "Discount" if price < eq else "Premium",
        "ote_705": float(ote_705),
        "fvgs": fvgs[-3:], # Last 3 FVGs
        "order_blocks": order_blocks[-2:], # Last 2 OBs
        "liquidity_sweep": {"high": sweep_h, "low": sweep_l},
        "pd_array": {"high": float(rng_high), "low": float(rng_low), "eq": float(eq)},
        "summary": f"{structure} | OBs: {len(order_blocks)} | FVGs: {len(fvgs)}"
    }
