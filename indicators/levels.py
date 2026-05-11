from .market_context import get_market_data

def calculate_key_levels() -> dict:
    df = get_market_data()
    if df is None or df.empty or len(df) < 50: return {"error": "Insufficient data"}
    lookback = 50
    recent_df = df.tail(lookback)
    sh, sl = recent_df['high'].max(), recent_df['low'].min()
    diff = sh - sl
    fibs = {"fib_0": float(sh), "fib_0.382": float(sh - 0.382 * diff), "fib_0.5": float(sh - 0.5 * diff), "fib_0.618": float(sh - 0.618 * diff), "fib_1": float(sl)}
    prev_day = df.iloc[-48:-24] if len(df) >= 48 else df.iloc[:-24]
    pivots = {}
    if not prev_day.empty:
        ph, pl, pc = prev_day['high'].max(), prev_day['low'].min(), prev_day['close'].iloc[-1]
        pivot = (ph + pl + pc) / 3
        pivots = {"pivot": float(pivot), "r1": float((2 * pivot) - pl), "s1": float((2 * pivot) - ph)}
    return {"fibonacci_retracements": fibs, "daily_pivots": pivots, "ote_zone": [fibs["fib_0.618"], fibs.get("fib_0.786", fibs["fib_0.618"] * 0.9)]}
