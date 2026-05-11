import asyncio
import time
import requests
from .market_context import is_backtest

_cache: dict = {"data": None, "ts": 0.0}
_TTL = 300  # 5-minute cache — macro data doesn't change tick-by-tick

_DEFAULT = {
    "sentiment": {"score": 50, "label": "Neutral"},
    "dominance":  {"btc": 52.5},
    "funding":    {"rate": 0.0001, "overcrowded": False},
}

def _fetch_sync() -> dict:
    """Runs in a thread pool — keeps event loop unblocked."""
    result = dict(_DEFAULT)
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=4)
        if r.status_code == 200:
            d = r.json()["data"][0]
            result["sentiment"] = {"score": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        pass
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=4)
        if r.status_code == 200:
            btc_dom = r.json().get("data", {}).get("market_cap_percentage", {}).get("btc", 52.0)
            result["dominance"]["btc"] = float(btc_dom)
    except Exception:
        pass
    return result

async def get_macro_analysis() -> dict:
    if is_backtest():
        return _DEFAULT
    now = time.monotonic()
    if _cache["data"] and now - _cache["ts"] < _TTL:
        return _cache["data"]
    # Off-load blocking HTTP calls to thread pool — event loop stays free
    result = await asyncio.get_event_loop().run_in_executor(None, _fetch_sync)
    _cache.update({"data": result, "ts": now})
    return result
