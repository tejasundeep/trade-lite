from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

from trading.adapters.crypto import CCXTCryptoAdapter


def _candidate_symbols(symbol: str, venue: str) -> List[str]:
    base, quote = symbol.split("/", 1)
    candidates = [symbol]

    if venue == "coinbase":
        if quote in {"USDT", "FDUSD", "BUSD"}:
            candidates.append(f"{base}/USD")
    elif venue == "okx":
        candidates.extend([f"{base}/{quote}", f"{base}/{quote}:USDT", f"{base}/{quote}-SWAP"])
    elif venue == "binance":
        candidates.append(f"{base}/{quote}")

    # Preserve order while removing duplicates.
    seen = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _fetch_last_price(exchange, candidates: Iterable[str]) -> Tuple[Optional[float], Optional[str]]:
    for candidate in candidates:
        try:
            ticker = exchange.fetch_ticker(candidate)
            price = float(ticker.get("last") or ticker.get("close") or 0.0)
            if price > 0:
                return price, candidate
        except Exception:
            continue
    return None, None


def analyze_cross_exchange_correlation(symbol: str = "BTC/USDT") -> dict:
    venues = {
        "binance": ccxt.binance({"enableRateLimit": True}),
        "coinbase": ccxt.coinbase({"enableRateLimit": True}),
        "okx": ccxt.okx({"enableRateLimit": True}),
    }

    prices: Dict[str, float] = {}
    resolved_symbols: Dict[str, str] = {}
    for name, exchange in venues.items():
        try:
            price, resolved = _fetch_last_price(exchange, _candidate_symbols(symbol, name))
            if price is not None and resolved is not None:
                prices[name] = price
                resolved_symbols[name] = resolved
        except Exception:
            continue

    if len(prices) < 2:
        return {"error": "Could not fetch enough venue prices", "prices": prices}

    base_price = prices.get("binance") or next(iter(prices.values()))
    premiums = {name: ((price - base_price) / base_price) * 10000 for name, price in prices.items()}

    bias = "Neutral"
    if "coinbase" in prices and "binance" in prices:
        cb_price = prices["coinbase"]
        bn_price = prices["binance"]
        if cb_price > bn_price * 1.0002:
            bias = "Bullish (Spot Leading)"
        elif bn_price > cb_price * 1.0002:
            bias = "Bearish (Perp Leading)"

    return {
        "symbol": symbol,
        "resolved_symbols": resolved_symbols,
        "prices": prices,
        "premiums_bps": premiums,
        "lead_lag_bias": bias,
    }


def analyze_asset_correlation(
    symbol: str,
    competitors: Optional[List[str]] = None,
    timeframe: str = "1h",
    limit: int = 240,
    adapter: Optional[CCXTCryptoAdapter] = None,
) -> dict:
    if competitors is None:
        competitors = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    adapter = adapter or CCXTCryptoAdapter(paper_trading=True, trading_mode="futures")
    assets = [symbol] + [comp for comp in competitors if comp != symbol]
    series: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}

    for asset in assets:
        try:
            df = adapter.get_market_data(asset, timeframe=timeframe, limit=limit)
            if df is None or df.empty or len(df) < 30:
                failures[asset] = "insufficient_data"
                continue
            returns = df["close"].astype(float).pct_change().dropna()
            if returns.empty:
                failures[asset] = "empty_returns"
                continue
            series[asset] = returns.reset_index(drop=True)
        except Exception as exc:
            failures[asset] = str(exc)

    if symbol not in series:
        return {"error": "Could not fetch data for base symbol", "failures": failures}

    aligned = pd.concat(series, axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < 5:
        return {"error": "Insufficient overlapping return history", "failures": failures}

    correlations: Dict[str, float] = {}
    for comp in competitors:
        if comp == symbol or comp not in aligned.columns:
            continue
        correlations[comp] = float(aligned[symbol].corr(aligned[comp]))

    avg_corr = float(np.mean(list(correlations.values()))) if correlations else 0.0
    cluster = any(abs(value) >= 0.85 for value in correlations.values())

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback": limit,
        "correlations": correlations,
        "average_correlation": avg_corr,
        "is_systemic_cluster": cluster,
        "failures": failures,
        "source": "historical_closes",
    }
