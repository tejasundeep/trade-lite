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
    def __init__(self, symbols: List[str], adapter: Optional[object] = None):
        self.orig_symbols = symbols
        self.adapter = adapter
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
        self._account_callbacks: List[Callable] = []
        self._listen_key: Optional[str] = None
        self._last_listen_key_update = 0

    def add_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def add_account_callback(self, cb: Callable):
        self._account_callbacks.append(cb)

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
            "bids": data['bids'] if 'bids' in data else data.get('b', [])[:10],
            "asks": data['asks'] if 'asks' in data else data.get('a', [])[:10]
        }

    async def _handle_account_update(self, data: dict):
        """Handle User Data Stream events (Spot and Futures)."""
        event_type = data.get("e")
        
        # Mapping common fields to a unified format
        update = {"event": event_type, "timestamp": data.get("E")}
        
        if event_type in ["ACCOUNT_UPDATE", "outboundAccountPosition"]:
            # Balance updates
            balances = []
            if event_type == "ACCOUNT_UPDATE":
                for b in data.get("a", {}).get("B", []):
                    balances.append({"asset": b["a"], "free": float(b["wb"]), "locked": 0.0})
            else:
                for b in data.get("B", []):
                    balances.append({"asset": b["a"], "free": float(b["f"]), "locked": float(b["l"])})
            update["balances"] = balances
        
        elif event_type in ["ORDER_TRADE_UPDATE", "executionReport"]:
            # Order updates
            o = data.get("o") if event_type == "ORDER_TRADE_UPDATE" else data
            update["order"] = {
                "symbol": self.symbol_map.get(o["s"].lower(), o["s"]),
                "side": o["S"].lower(),
                "status": o["X"],
                "price": float(o["p"]),
                "amount": float(o["q"]),
                "filled": float(o["z"]),
                "last_filled_price": float(o["L"]) if event_type == "executionReport" else float(o["ap"])
            }

        for cb in self._account_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb): await cb(update)
                else: cb(update)
            except Exception as e: log.error(f"Account callback error: {e}")

    async def _manage_listen_key(self):
        """Background task to keep the listenKey alive."""
        while not self._stop_event.is_set():
            try:
                if self.adapter:
                    if not self._listen_key:
                        self._listen_key = self.adapter.get_listen_key()
                        self._last_listen_key_update = asyncio.get_event_loop().time()
                        log.info(f"New ListenKey acquired: {self._listen_key}")
                    elif asyncio.get_event_loop().time() - self._last_listen_key_update > 1800: # 30 mins
                        if self.adapter.keep_alive_listen_key(self._listen_key):
                            self._last_listen_key_update = asyncio.get_event_loop().time()
                            log.info("ListenKey extended successfully.")
                        else:
                            self._listen_key = None # Force refresh next loop
                await asyncio.sleep(60)
            except Exception as e:
                log.error(f"ListenKey management error: {e}")
                await asyncio.sleep(10)

    def _is_futures(self) -> bool:
        if not self.adapter: return False
        return "fapi" in str(type(self.adapter.exchange)).lower() or "usdm" in self.adapter.exchange.id.lower() or "dpm" in self.adapter.exchange.id.lower()

    async def _listen(self):
        stream_names = []
        for s in self.symbols_ws:
            stream_names.append(f"{s}@aggTrade")
            stream_names.append(f"{s}@depth10@100ms")
        
        is_futures = self._is_futures()
        if is_futures:
            # Use routed market endpoint for regular streams (aggTrade, depth)
            base_url = "wss://fstream.binance.com/market/stream"
        else:
            base_url = "wss://stream.binance.com:9443/stream"
            
        url = f"{base_url}?streams={'/'.join(stream_names)}"
        
        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                # Add User Data Stream to the combined stream list if listenKey is available
                current_streams = list(stream_names)
                if self._listen_key:
                    # For Spot, we use the same URL but add the listenKey as a stream or a separate connection.
                    # Actually, for combined streams, we can add it. 
                    # But often it's cleaner to have a separate connection for User Data.
                    pass 

                # Connect to Market Data
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info(f"Elite Streamer Connected: {url}")
                    retry_delay = 1
                    
                    # Also connect to User Data in parallel if listenKey exists
                    user_ws_task = None
                    if self._listen_key:
                        user_url = f"wss://stream.binance.com:9443/ws/{self._listen_key}"
                        # Check if it's futures
                        if is_futures:
                            user_url = f"wss://fstream.binance.com/private/ws/{self._listen_key}"
                        
                        user_ws_task = asyncio.create_task(self._listen_user_data(user_url))

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
                
                if user_ws_task: user_ws_task.cancel()
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error(f"WebSocket Connection Lost: {e}. Reconnecting in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60) # Exponential backoff up to 60s

    async def _listen_user_data(self, url: str):
        """Listen to the User Data Stream."""
        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info(f"User Data Stream Connected: {url}")
                    retry_delay = 1
                    while not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)
                        await self._handle_account_update(data)
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error(f"User Data Stream Lost: {e}. Reconnecting in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)
                    # If we keep failing, force a listenKey refresh
                    if retry_delay > 30:
                        self._listen_key = None

    def start(self):
        asyncio.create_task(self._manage_listen_key())
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
