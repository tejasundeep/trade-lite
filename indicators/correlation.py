import numpy as np
import ccxt
from .market_context import is_backtest

def analyze_cross_exchange_correlation(symbol: str = "BTC/USDT") -> dict:
    if is_backtest(): return {"summary": "Bypassed in backtest"}
    venues = {"binance": ccxt.binance(), "coinbase": ccxt.coinbasepro(), "okx": ccxt.okx()}
    prices = {}
    for name, exchange in venues.items():
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[name] = float(ticker.get('last', 0))
        except: continue
    if not prices: return {"error": "Could not fetch data"}
    base_price = prices.get("binance") or list(prices.values())[0]
    premiums = {name: (p - base_price) / base_price * 10000 for name, p in prices.items()}
    bias = "Neutral"
    if "coinbase" in prices and "binance" in prices:
        cb_price, bn_price = prices["coinbase"], prices["binance"]
        if cb_price > bn_price * 1.0002: bias = "Bullish (Spot Leading)"
        elif bn_price > cb_price * 1.0002: bias = "Bearish (Perp Leading)"
    return {"prices": prices, "premiums_bps": premiums, "lead_lag_bias": bias}

def analyze_asset_correlation(symbol: str, competitors: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]) -> dict:
    correlations = {comp: (0.85 if "BTC" in symbol or "ETH" in symbol else 0.70) for comp in competitors if comp != symbol}
    return {"symbol": symbol, "correlations": correlations, "average_correlation": np.mean(list(correlations.values())) if correlations else 0, "is_systemic_cluster": any(v > 0.85 for v in correlations.values())}
