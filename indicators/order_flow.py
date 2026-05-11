import numpy as np
import pandas as pd
from .market_context import get_market_data
from trading.adapters.crypto import CCXTCryptoAdapter

def _value_area_from_profile(hist: np.ndarray, edges: np.ndarray, poc_idx: int) -> tuple[float, float]:
    if hist.size == 0 or edges.size < 2:
        return 0.0, 0.0

    target = hist.sum() * 0.70
    cur_vol = float(hist[poc_idx])
    left = right = int(poc_idx)

    while cur_vol < target and (left > 0 or right < len(hist) - 1):
        left_vol = float(hist[left - 1]) if left > 0 else -1.0
        right_vol = float(hist[right + 1]) if right < len(hist) - 1 else -1.0
        if left_vol >= right_vol and left > 0:
            left -= 1
            cur_vol += left_vol
        elif right < len(hist) - 1:
            right += 1
            cur_vol += right_vol
        else:
            break

    return float(edges[right + 1]), float(edges[left])


def _profile_from_prices(prices: np.ndarray, volumes: np.ndarray, bins: int = 30) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    if prices.size == 0 or volumes.size == 0:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    unique_prices = max(len(np.unique(prices)), 1)
    bins = max(10, min(int(unique_prices), bins))
    if prices.min() == prices.max():
        bins = 1

    hist, edges = np.histogram(prices, bins=bins, weights=volumes)
    if hist.size == 0 or edges.size < 2:
        return float(prices[-1]), float(prices[-1]), float(prices[-1]), hist, edges

    poc_idx = int(np.argmax(hist))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    vah, val = _value_area_from_profile(hist, edges, poc_idx)
    return poc, vah, val, hist, edges


def _trade_order_flow(trades_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    prices = trades_df["price"].astype(float).to_numpy()
    volumes = trades_df["amount"].astype(float).to_numpy()
    poc, vah, val, hist, edges = _profile_from_prices(prices, volumes)

    tick_size = max((edges[1] - edges[0]) / 2 if edges.size >= 2 else 0.0, 1e-9)
    tick_prices = trades_df["price"].astype(float).apply(lambda x: round(x / tick_size) * tick_size)
    tick_groups = trades_df.assign(tick=tick_prices).groupby("tick")["amount"].sum().sort_values(ascending=False)

    iceberg = "None"
    if not tick_groups.empty and tick_groups.mean() > 0:
        top_tick_vol = float(tick_groups.iloc[0])
        avg_tick_vol = float(tick_groups.mean())
        if top_tick_vol > avg_tick_vol * 4:
            iceberg = f"Iceberg Detected @ {float(tick_groups.index[0]):.2f}"

    signed = trades_df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
    delta = float((trades_df["amount"].astype(float) * signed).sum())
    cvd = float((trades_df["amount"].astype(float) * signed).cumsum().iloc[-1]) if len(trades_df) else 0.0

    buy_vol = float(trades_df.loc[trades_df["side"] == "buy", "amount"].astype(float).sum())
    sell_vol = float(trades_df.loc[trades_df["side"] == "sell", "amount"].astype(float).sum())
    ratio = buy_vol / (sell_vol + 1e-9)
    imbalance = "Neutral"
    if ratio > 2.5:
        imbalance = "Extreme Bullish Imbalance"
    elif ratio < 0.4:
        imbalance = "Extreme Bearish Imbalance"

    absorption = "None"
    if len(df) > 0:
        last_open = float(df["open"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if delta > (buy_vol + sell_vol) * 0.3 and last_close <= last_open:
            absorption = "Bullish Absorption"
        elif delta < -(buy_vol + sell_vol) * 0.3 and last_close >= last_open:
            absorption = "Bearish Absorption"

    return {
        "source": "stream_trades",
        "method": "tick_volume_profile",
        "poc": float(poc),
        "vah": float(vah),
        "val": float(val),
        "delta": delta,
        "cvd": cvd,
        "imbalance": imbalance,
        "absorption": absorption,
        "iceberg": iceberg,
    }


def _historical_order_flow(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "No data"}

    closes = df["close"].astype(float).to_numpy()
    volumes = df["volume"].astype(float).to_numpy()
    opens = df["open"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()

    poc, vah, val, hist, edges = _profile_from_prices(closes, volumes)

    candle_range = np.maximum(highs - lows, 1e-9)
    body = closes - opens
    body_strength = np.clip(np.abs(body) / candle_range, 0.0, 1.0)
    direction = np.where(body >= 0, 1.0, -1.0)
    signed_volume = volumes * direction * np.where(body_strength > 0, body_strength, 0.25)

    delta = float(signed_volume.sum())
    cvd = float(np.cumsum(signed_volume)[-1]) if signed_volume.size else 0.0

    bullish_vol = float(volumes[direction > 0].sum())
    bearish_vol = float(volumes[direction < 0].sum())
    ratio = bullish_vol / (bearish_vol + 1e-9)
    imbalance = "Neutral"
    if ratio > 2.0:
        imbalance = "Extreme Bullish Imbalance"
    elif ratio < 0.5:
        imbalance = "Extreme Bearish Imbalance"

    avg_volume = pd.Series(volumes).rolling(20, min_periods=1).mean().to_numpy()
    avg_range = pd.Series(candle_range).rolling(20, min_periods=1).mean().to_numpy()
    last_volume = float(volumes[-1])
    last_body = float(body[-1])
    last_range = float(candle_range[-1])

    iceberg = "None"
    if last_volume > float(avg_volume[-1]) * 3 and last_range > 0 and abs(last_body) <= last_range * 0.2:
        iceberg = f"Volume Spike @ {float(closes[-1]):.2f}"

    absorption = "None"
    if delta > 0 and last_body <= 0 and last_volume > float(avg_volume[-1]):
        absorption = "Bullish Absorption"
    elif delta < 0 and last_body >= 0 and last_volume > float(avg_volume[-1]):
        absorption = "Bearish Absorption"

    return {
        "source": "historical_candles",
        "method": "candle_proxy",
        "poc": float(poc),
        "vah": float(vah),
        "val": float(val),
        "delta": delta,
        "cvd": cvd,
        "imbalance": imbalance,
        "absorption": absorption,
        "iceberg": iceberg,
        "summary": f"POC: {poc:.2f} | Ice: {iceberg != 'None'}",
    }


def analyze_order_flow(symbol: str = "BTC/USDT", adapter: CCXTCryptoAdapter = None, streamer=None) -> dict:
    raw = get_market_data()
    if raw is None or raw.empty:
        return {"error": "No data"}

    df = raw.copy()
    if streamer is not None:
        trades_df = streamer.get_trades_df(symbol, limit=2000)
        if not trades_df.empty:
            result = _trade_order_flow(trades_df, df)
            result["summary"] = f"POC: {result['poc']:.2f} | Ice: {result['iceberg'] != 'None'}"
            return result

    result = _historical_order_flow(df)
    if "summary" not in result:
        result["summary"] = f"POC: {result.get('poc', 0.0):.2f} | Ice: {result.get('iceberg') != 'None'}"
    return result


def analyze_open_interest(symbol: str = "BTC/USDT", adapter: CCXTCryptoAdapter = None) -> dict:
    if adapter is None:
        adapter = CCXTCryptoAdapter()
    df = get_market_data()
    if df is None or df.empty:
        return {"error": "No data"}
    if len(df) < 2:
        return {"error": "Insufficient price history"}
    try:
        history = adapter.get_open_interest_history(symbol, limit=10, period="5m")
        current = adapter.get_open_interest(symbol)
        if not history and not current:
            return {"error": "OI history unavailable"}

        def _row_amount(row: dict) -> float:
            for key in ("openInterestAmount", "sumOpenInterest", "openInterest", "sumOpenInterestValue", "openInterestValue"):
                if key in row and row.get(key) not in (None, ""):
                    try:
                        return float(row.get(key))
                    except Exception:
                        continue
            return 0.0

        cur = float(_row_amount(current)) if current else float(_row_amount(history[-1]))
        prev = float(_row_amount(history[-2])) if len(history) > 1 else cur
        oi_chg = ((cur - prev) / prev * 100.0) if prev > 0 else 0.0
        px_chg  = df.iloc[-1]["close"] - df.iloc[-2]["close"]
        if   px_chg > 0 and oi_chg >  2.0: bias = "Aggressive Bullish"
        elif px_chg < 0 and oi_chg >  2.0: bias = "Aggressive Bearish"
        elif px_chg > 0 and oi_chg < -2.0: bias = "Short Covering"
        elif px_chg < 0 and oi_chg < -2.0: bias = "Long Liquidation"
        else:                                bias = "Neutral"
        return {
            "source": "binance_futures_data",
            "current_oi": cur,
            "oi_change_pct": round(oi_chg, 2),
            "oi_bias": bias,
            "history_points": len(history),
        }
    except Exception as e:
        return {"error": str(e)}
