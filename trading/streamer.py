import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import websockets

log = logging.getLogger(__name__)


class BinanceStreamer:
    """
    Binance Spot streamer using the official market-data WebSocket streams and
    the Spot WebSocket API for user-data subscriptions.
    """

    def __init__(self, symbols: List[str], adapter: Optional[object] = None):
        self.orig_symbols = symbols
        self.adapter = adapter

        self.symbol_map = {s.lower().replace("/", ""): s for s in symbols}
        self.symbols_ws = list(self.symbol_map.keys())

        self.prices: Dict[str, float] = {s: 0.0 for s in symbols}
        self.trades: Dict[str, deque] = {s: deque(maxlen=2000) for s in symbols}
        self.orderbook: Dict[str, dict] = {
            s: {"bid": None, "ask": None, "bids": [], "asks": []} for s in symbols
        }

        self.candles: Dict[str, List[dict]] = {s: [] for s in symbols}
        self._current_candle: Dict[str, dict] = {s: None for s in symbols}
        self._max_candles = 500

        self._stop_event = asyncio.Event()
        self._callbacks: List[Callable] = []
        self._account_callbacks: List[Callable] = []

    def add_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def add_account_callback(self, cb: Callable):
        self._account_callbacks.append(cb)

    def _update_candle(self, symbol: str, price: float, volume: float, timestamp: int):
        minute_ts = (timestamp // 60000) * 60000
        curr = self._current_candle[symbol]

        if curr is None or curr["timestamp"] != minute_ts:
            if curr is not None:
                self.candles[symbol].append(curr)
                if len(self.candles[symbol]) > self._max_candles:
                    self.candles[symbol].pop(0)

            self._current_candle[symbol] = {
                "timestamp": minute_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "trades": 1,
            }
        else:
            curr["high"] = max(curr["high"], price)
            curr["low"] = min(curr["low"], price)
            curr["close"] = price
            curr["volume"] += volume
            curr["trades"] += 1

    async def _handle_trade(self, symbol_ws: str, data: dict):
        orig_symbol = self.symbol_map.get(symbol_ws)
        if not orig_symbol:
            return

        price = float(data["p"])
        qty = float(data["q"])
        ts = int(data["T"])
        is_buyer_maker = bool(data["m"])

        self.prices[orig_symbol] = price

        trade = {
            "timestamp": ts,
            "price": price,
            "amount": qty,
            "side": "sell" if is_buyer_maker else "buy",
        }

        self.trades[orig_symbol].append(trade)
        self._update_candle(orig_symbol, price, qty, ts)

        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(orig_symbol, "trade", trade)
                else:
                    cb(orig_symbol, "trade", trade)
            except Exception as e:
                log.error("Callback error: %s", e)

    async def _handle_book_ticker(self, symbol_ws: str, data: dict):
        orig_symbol = self.symbol_map.get(symbol_ws)
        if not orig_symbol:
            return

        bid = float(data.get("b", 0.0))
        ask = float(data.get("a", 0.0))
        bid_qty = float(data.get("B", 0.0))
        ask_qty = float(data.get("A", 0.0))

        self.orderbook[orig_symbol] = {
            "bid": bid,
            "ask": ask,
            "bids": [[str(bid), str(bid_qty)]],
            "asks": [[str(ask), str(ask_qty)]],
        }

        if bid > 0 and ask > 0:
            self.prices[orig_symbol] = (bid + ask) / 2.0

    def _normalize_account_event(self, payload: dict) -> Optional[dict]:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        event_type = event.get("e")
        if not event_type:
            return None

        update = {
            "event": event_type,
            "timestamp": event.get("E") or event.get("T"),
            "raw": event,
        }

        if event_type == "outboundAccountPosition":
            update["balances"] = [
                {"asset": b["a"], "free": float(b["f"]), "locked": float(b["l"])}
                for b in event.get("B", [])
            ]
        elif event_type == "balanceUpdate":
            update["balance_update"] = {
                "asset": event.get("a"),
                "delta": float(event.get("d", 0.0)),
                "clear_time": event.get("T"),
            }
        elif event_type == "executionReport":
            update["order"] = {
                "symbol": self.symbol_map.get(event.get("s", "").lower(), event.get("s")),
                "side": event.get("S", "").lower(),
                "status": event.get("X"),
                "type": event.get("o"),
                "price": float(event.get("p", 0.0)),
                "amount": float(event.get("q", 0.0)),
                "filled": float(event.get("z", 0.0)),
                "last_filled_price": float(event.get("L", 0.0)),
                "commission": float(event.get("n", 0.0)),
                "commission_asset": event.get("N"),
            }
        elif event_type == "externalLockUpdate":
            update["external_lock"] = {
                "asset": event.get("a"),
                "delta": float(event.get("d", 0.0)),
                "transaction_time": event.get("T"),
            }

        return update

    async def _dispatch_account_update(self, payload: dict):
        update = self._normalize_account_event(payload)
        if not update:
            return

        for cb in self._account_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(update)
                else:
                    cb(update)
            except Exception as e:
                log.error("Account callback error: %s", e)

    async def _listen_market(self):
        stream_names = []
        for s in self.symbols_ws:
            stream_names.append(f"{s}@aggTrade")
            stream_names.append(f"{s}@bookTicker")

        url = f"{self.adapter.ws_url if self.adapter and getattr(self.adapter, 'ws_url', None) else 'wss://stream.binance.com:9443/stream'}?streams={'/'.join(stream_names)}"

        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=None, ping_timeout=None) as ws:
                    log.info("Market stream connected: %s", url)
                    retry_delay = 1

                    while not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        if "@aggTrade" in stream:
                            symbol_ws = stream.split("@", 1)[0]
                            await self._handle_trade(symbol_ws, payload)
                        elif "@bookTicker" in stream:
                            symbol_ws = stream.split("@", 1)[0]
                            await self._handle_book_ticker(symbol_ws, payload)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error("Market WebSocket lost: %s. Reconnecting in %ss...", e, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)

    async def _listen_user_data(self):
        if not self.adapter or not getattr(self.adapter, "api_key", None) or not getattr(self.adapter, "secret", None):
            log.info("Skipping Spot user data stream because no API credentials are configured.")
            return

        url = getattr(self.adapter, "ws_api_url", "wss://ws-api.binance.com:443/ws-api/v3")
        retry_delay = 1

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=None, ping_timeout=None) as ws:
                    log.info("User data WS API connected: %s", url)
                    retry_delay = 1

                    ts = int(time.time() * 1000)
                    subscribe_params = {
                        "apiKey": self.adapter.api_key,
                        "timestamp": ts,
                        "recvWindow": getattr(self.adapter, "recv_window", 5000),
                    }
                    subscribe_params["signature"] = self.adapter._signed_request_signature(subscribe_params)

                    subscribe_payload = {
                        "id": "user-data-subscribe",
                        "method": "userDataStream.subscribe.signature",
                        "params": subscribe_params,
                    }
                    await ws.send(json.dumps(subscribe_payload))

                    subscribed = False
                    while not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)

                        if data.get("id") == "user-data-subscribe":
                            subscribed = data.get("status") == 200
                            if not subscribed:
                                raise RuntimeError(f"User data subscribe failed: {data}")
                            continue

                        if "event" in data:
                            await self._dispatch_account_update(data)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error("User data WebSocket lost: %s. Reconnecting in %ss...", e, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)

    def start(self):
        asyncio.create_task(self._listen_market())
        asyncio.create_task(self._listen_user_data())

    def stop(self):
        self._stop_event.set()

    def get_candles(self, symbol: str) -> pd.DataFrame:
        history = list(self.candles.get(symbol, []))
        if self._current_candle.get(symbol):
            history.append(self._current_candle[symbol])

        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_trades_df(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        t_list = list(self.trades.get(symbol, []))
        if not t_list:
            return pd.DataFrame()

        df = pd.DataFrame(t_list[-limit:])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_spread(self, symbol: str) -> float:
        ob = self.orderbook.get(symbol)
        if not ob:
            return 0.0

        bid = ob.get("bid")
        ask = ob.get("ask")
        if bid and ask:
            return float(ask) - float(bid)

        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return 0.0
        return float(asks[0][0]) - float(bids[0][0])
