from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote

import pandas as pd
import requests

log = logging.getLogger(__name__)


class CCXTCryptoAdapter:
    """
    Binance Spot REST adapter.

    The project name kept the historical CCXT adapter class name, but the
    implementation now speaks directly to the official Binance Spot REST API.
    That keeps the behavior aligned with the current docs instead of relying on
    wrapper-specific endpoint guesses.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper_trading: bool = True,
    ):
        self.exchange_id = (exchange_id or "binance").lower().strip()
        self.api_key = (api_key or "").strip()
        self.secret = (secret or "").strip()
        self.paper_trading = paper_trading

        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/stream"
        self.ws_api_url = "wss://ws-api.binance.com:443/ws-api/v3"
        self.recv_window = 5000
        self._paper_balance = float(os.getenv("PAPER_BALANCE", "10000"))

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _public_request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _signed_request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        if not self.api_key or not self.secret:
            raise ValueError("Binance API key and secret are required for signed requests.")

        payload = self._signed_payload(params)
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, params=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def _signed_payload(self, params: Optional[dict] = None) -> Dict[str, Any]:
        payload = dict(params or {})
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("recvWindow", self.recv_window)

        query = urlencode(payload, doseq=True, quote_via=quote)
        signature = self._sign_query(query)
        payload["signature"] = signature
        return payload

    def _sign_query(self, query: str) -> str:
        return hmac.new(
            self.secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_request_signature(self, params: Dict[str, Any]) -> str:
        query = self._canonical_query(params)
        return self._sign_query(query)

    @staticmethod
    def _canonical_query(params: Dict[str, Any]) -> str:
        return "&".join(f"{k}={params[k]}" for k in sorted(params))

    def _exchange_info(self) -> Dict[str, Any]:
        if self._exchange_info_cache is None:
            self._exchange_info_cache = self._public_request("GET", "/api/v3/exchangeInfo")
        return self._exchange_info_cache

    def _symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        m_symbol = self.get_market_symbol(symbol)
        if m_symbol in self._symbol_cache:
            return self._symbol_cache[m_symbol]

        info = self._exchange_info()
        for entry in info.get("symbols", []):
            if entry.get("symbol") == m_symbol:
                self._symbol_cache[m_symbol] = entry
                return entry
        return None

    @staticmethod
    def _floor_to_step(value: float, step: str) -> str:
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        if d_step == 0:
            return format(d_value, "f")
        quantized = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return format(quantized.normalize(), "f")

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        info = self._symbol_info(symbol)
        step = "0.000001"
        if info:
            for f in info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    step = f.get("stepSize", step)
                    break
        return self._floor_to_step(amount, step)

    def price_to_precision(self, symbol: str, price: float) -> str:
        info = self._symbol_info(symbol)
        tick = "0.01"
        if info:
            for f in info.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = f.get("tickSize", tick)
                    break
        return self._floor_to_step(price, tick)

    # ------------------------------------------------------------------
    # Symbol and market data
    # ------------------------------------------------------------------
    def get_market_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper().strip()

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        raw_ohlcv = self._public_request(
            "GET",
            "/api/v3/klines",
            {"symbol": self.get_market_symbol(symbol), "interval": timeframe, "limit": limit},
        )
        if not raw_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(raw_ohlcv)
        df = df.iloc[:, :12]
        df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_ticker(self, symbol: str) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        last_price = self._public_request("GET", "/api/v3/ticker/price", {"symbol": m_symbol})
        book_ticker = self._public_request("GET", "/api/v3/ticker/bookTicker", {"symbol": m_symbol})

        bid = float(book_ticker.get("bidPrice", 0.0))
        ask = float(book_ticker.get("askPrice", 0.0))
        price = float(last_price.get("price", 0.0))
        if price == 0.0 and bid > 0 and ask > 0:
            price = (bid + ask) / 2.0

        return {
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "timestamp": int(time.time() * 1000),
        }

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        return self._public_request(
            "GET",
            "/api/v3/depth",
            {"symbol": self.get_market_symbol(symbol), "limit": limit},
        )

    def get_real_delta(self, symbol: str, since_ms: int) -> dict:
        try:
            trades = self._public_request(
                "GET",
                "/api/v3/aggTrades",
                {"symbol": self.get_market_symbol(symbol), "startTime": since_ms},
            )
            if not trades:
                return {"delta": 0, "buy_vol": 0, "sell_vol": 0, "trades_count": 0}

            buy_vol = 0.0
            sell_vol = 0.0
            for t in trades:
                qty = float(t.get("q", 0.0))
                if t.get("m"):
                    sell_vol += qty
                else:
                    buy_vol += qty

            avg_size = (buy_vol + sell_vol) / max(len(trades), 1)
            large_trades = [t for t in trades if float(t.get("q", 0.0)) > avg_size * 5]
            return {
                "delta": buy_vol - sell_vol,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "trades_count": len(trades),
                "institutional_activity": len(large_trades) > 0,
                "large_trade_count": len(large_trades),
            }
        except Exception as e:
            return {"delta": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Account and order management
    # ------------------------------------------------------------------
    def get_account_balance(self) -> Dict:
        if self.paper_trading and not (self.api_key and self.secret):
            return {
                "free": {"USDT": self._paper_balance},
                "locked": {"USDT": 0.0},
                "paper": True,
            }

        account = self._signed_request("GET", "/api/v3/account")
        free = {b["asset"]: float(b["free"]) for b in account.get("balances", [])}
        locked = {b["asset"]: float(b["locked"]) for b in account.get("balances", [])}
        account["free"] = free
        account["locked"] = locked
        return account

    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        quantity = self.amount_to_precision(m_symbol, amount)

        if self.paper_trading:
            return {
                "symbol": m_symbol,
                "side": side.upper(),
                "type": "MARKET",
                "status": "FILLED",
                "executedQty": quantity,
                "origQty": quantity,
                "cummulativeQuoteQty": "0",
                "paper": True,
            }

        return self._signed_request(
            "POST",
            "/api/v3/order",
            {
                "symbol": m_symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": quantity,
                "newOrderRespType": "FULL",
            },
        )

    def _place_entry_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        quantity = self.amount_to_precision(m_symbol, amount)
        limit_price = self.price_to_precision(m_symbol, price)

        if self.paper_trading:
            return {
                "symbol": m_symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "status": "FILLED",
                "executedQty": quantity,
                "origQty": quantity,
                "price": limit_price,
                "paper": True,
            }

        return self._signed_request(
            "POST",
            "/api/v3/order",
            {
                "symbol": m_symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": quantity,
                "price": limit_price,
                "newOrderRespType": "FULL",
            },
        )

    def _place_oco_exit(self, symbol: str, side: str, quantity: str, stop_loss: float, take_profit: float) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        if side.lower() == "buy":
            # Closing a short: stop above, take profit below.
            above_price = self.price_to_precision(m_symbol, stop_loss)
            above_stop = self.price_to_precision(m_symbol, stop_loss)
            below_price = self.price_to_precision(m_symbol, take_profit)
            below_stop = self.price_to_precision(m_symbol, take_profit)
        else:
            # Closing a long: take profit above, stop below.
            above_price = self.price_to_precision(m_symbol, take_profit)
            above_stop = self.price_to_precision(m_symbol, take_profit)
            below_price = self.price_to_precision(m_symbol, stop_loss)
            below_stop = self.price_to_precision(m_symbol, stop_loss)

        if self.paper_trading:
            return {
                "symbol": m_symbol,
                "type": "OCO",
                "status": "EXECUTING",
                "paper": True,
                "abovePrice": above_price,
                "belowPrice": below_price,
            }

        return self._signed_request(
            "POST",
            "/api/v3/orderList/oco",
            {
                "symbol": m_symbol,
                "side": side.upper(),
                "quantity": quantity,
                "aboveType": "TAKE_PROFIT_LIMIT" if side.lower() == "sell" else "STOP_LOSS_LIMIT",
                "abovePrice": above_price,
                "aboveStopPrice": above_stop,
                "aboveTimeInForce": "GTC",
                "belowType": "STOP_LOSS_LIMIT" if side.lower() == "sell" else "TAKE_PROFIT_LIMIT",
                "belowPrice": below_price,
                "belowStopPrice": below_stop,
                "belowTimeInForce": "GTC",
            },
        )

    def place_order_with_sl_tp(self, symbol: str, side: str, amount: float, price: float, stop_loss: float, take_profit: float) -> Dict:
        try:
            entry_order = self._place_entry_order(symbol, side, amount, price)
            filled_qty = Decimal(str(entry_order.get("executedQty", entry_order.get("origQty", "0"))))
            order_status = str(entry_order.get("status", "")).upper()

            exit_side = "sell" if side.lower() == "buy" else "buy"
            sl_order = None
            tp_order = None

            if order_status == "FILLED" and filled_qty > 0:
                exit_qty = self.amount_to_precision(symbol, float(filled_qty))
                oco = self._place_oco_exit(symbol, exit_side, exit_qty, stop_loss, take_profit)
                sl_order = oco
                tp_order = oco

            return {
                "entry": entry_order,
                "sl": sl_order,
                "tp": tp_order,
                "strategy": "Spot OCO protection",
            }
        except Exception as e:
            log.error("place_order_with_sl_tp error: %s", e)
            return {"error": str(e)}

    def cancel_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict:
        params = {"symbol": self.get_market_symbol(symbol)}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._signed_request("DELETE", "/api/v3/order", params)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {}
        if symbol:
            params["symbol"] = self.get_market_symbol(symbol)
        return self._signed_request("GET", "/api/v3/openOrders", params)

    def get_open_interest_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        log.warning("Open interest history is not available on Binance Spot REST API.")
        return []

    def get_listen_key(self) -> Optional[str]:
        log.warning("Spot user data now uses the Binance WebSocket API; listenKey is not used.")
        return None

    def keep_alive_listen_key(self, listen_key: str) -> bool:
        return False
