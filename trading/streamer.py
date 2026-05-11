import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
from collections import deque
import websockets
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

class BinanceStreamer:
    """
    State-of-the-Art Real-time Data Streamer for Binance.
    - Zero-Latency local candle building.
    - Optimized memory management with deques.
    - High-frequency pre-mapped symbol lookups.
    """
    def __init__(self, symbols: List[str]):
        self.orig_symbols = symbols
        # Map "btcusdt" -> "BTC/USDT"
        self.symbol_map = {s.lower().replace("/", ""): s for s in symbols}
        self.symbols_ws = list(self.symbol_map.keys())
        
        self.prices: Dict[str, float] = {s: 0.0 for s in symbols}
        self.trades: Dict[str, deque] = {s: deque(maxlen=2000) for s in symbols}
        self.orderbook: Dict[str, dict] = {s: {"bids": [], "asks": []} for s in symbols}
        
        # Local Candle Management (1m bars)
        self.candles: Dict[str, List[dict]] = {s: [] for s in symbols}
        self._current_candle: Dict[str, dict] = {s: None for s in symbols}
        self._max_candles = 500
        
        self._stop_event = asyncio.Event()
        self._callbacks: List[Callable] = []

    def add_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def _update_candle(self, symbol: str, price: float, volume: float, timestamp: int):
        # 1-minute bucket (60,000 ms)
        minute_ts = (timestamp // 60000) * 60000
        
        curr = self._current_candle[symbol]
        
        if curr is None or curr['timestamp'] != minute_ts:
            # Finalize previous candle if it exists
            if curr is not None:
                self.candles[symbol].append(curr)
                if len(self.candles[symbol]) > self._max_candles:
                    self.candles[symbol].pop(0)
            
            # Start new candle
            self._current_candle[symbol] = {
                "timestamp": minute_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "trades": 1
            }
        else:
            # Update existing candle
            curr['high'] = max(curr['high'], price)
            curr['low'] = min(curr['low'], price)
            curr['close'] = price
            curr['volume'] += volume
            curr['trades'] += 1

    async def _handle_trade(self, symbol_ws: str, data: dict):
        orig_symbol = self.symbol_map.get(symbol_ws)
        if not orig_symbol: return

        price = float(data['p'])
        qty = float(data['q'])
        ts = data['T']
        is_buyer_maker = data['m']
        
        self.prices[orig_symbol] = price
        
        trade = {
            "timestamp": ts,
            "price": price,
            "amount": qty,
            "side": "sell" if is_buyer_maker else "buy"
        }
        
        self.trades[orig_symbol].append(trade)
        self._update_candle(orig_symbol, price, qty, ts)
            
        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb): await cb(orig_symbol, "trade", trade)
                else: cb(orig_symbol, "trade", trade)
            except Exception as e: log.error(f"Callback error: {e}")

    async def _handle_depth(self, symbol_ws: str, data: dict):
        orig_symbol = self.symbol_map.get(symbol_ws)
        if not orig_symbol: return
        
        self.orderbook[orig_symbol] = {
            "bids": data['b'][:10],
            "asks": data['a'][:10]
        }

    async def _listen(self):
        stream_names = []
        for s in self.symbols_ws:
            stream_names.append(f"{s}@aggTrade")
            stream_names.append(f"{s}@depth10@100ms")
        
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(stream_names)}"
        
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info(f"Elite Streamer Connected: {url}")
                    while not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        stream = data.get("stream", "")
                        if "@aggTrade" in stream:
                            symbol_ws = stream.split("@")[0]
                            await self._handle_trade(symbol_ws, data['data'])
                        elif "@depth" in stream:
                            symbol_ws = stream.split("@")[0]
                            await self._handle_depth(symbol_ws, data['data'])
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error(f"WebSocket Connection Lost: {e}. Reconnecting...")
                    await asyncio.sleep(5)

    def start(self):
        asyncio.create_task(self._listen())

    def stop(self):
        self._stop_event.set()

    def get_candles(self, symbol: str) -> pd.DataFrame:
        """Returns the local OHLCV history as a DataFrame."""
        history = list(self.candles.get(symbol, []))
        # Include current forming candle to be real-time
        if self._current_candle.get(symbol):
            history.append(self._current_candle[symbol])
            
        if not history: return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_trades_df(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Efficiently converts deque of trades to DataFrame."""
        t_list = list(self.trades.get(symbol, []))
        if not t_list: return pd.DataFrame()
        
        df = pd.DataFrame(t_list[-limit:])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_spread(self, symbol: str) -> float:
        ob = self.orderbook.get(symbol)
        if not ob or not ob['bids'] or not ob['asks']: return 0.0
        return float(ob['asks'][0][0]) - float(ob['bids'][0][0])
