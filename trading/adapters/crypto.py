from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import pandas as pd
import requests

log = logging.getLogger(__name__)


class CCXTCryptoAdapter:
    """
    Binance trading adapter with explicit spot/futures support.

    The project originally used spot-only execution semantics. This adapter
    now supports both Binance Spot and Binance USD-M Futures so the strategy
    layer can express true long/short execution without pretending that spot
    sells are shorts.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper_trading: bool = True,
        trading_mode: str = "futures",
        leverage: int = 3,
        margin_type: str = "ISOLATED",
        position_mode: str = "ONE_WAY",
    ):
        self.exchange_id = (exchange_id or "binance").lower().strip()
        self.api_key = (api_key or "").strip()
        self.secret = (secret or "").strip()
        self.paper_trading = paper_trading
        self.trading_mode = (trading_mode or "futures").lower().strip()
        if self.trading_mode not in {"spot", "futures"}:
            raise ValueError("trading_mode must be 'spot' or 'futures'")

        self.leverage = max(int(leverage or 1), 1)
        self.margin_type = (margin_type or "ISOLATED").upper().strip()
        self.position_mode = (position_mode or "ONE_WAY").upper().strip()

        if self.trading_mode == "futures":
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/stream"
            self.ws_user_data_url = "wss://fstream.binance.com/ws"
        else:
            self.base_url = "https://api.binance.com"
            self.ws_url = "wss://stream.binance.com:9443/stream"
            self.ws_user_data_url = "wss://ws-api.binance.com/ws-api/v3"

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
    def _request(self, method: str, path: str, params: Optional[dict] = None, signed: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        payload = self._signed_payload(params) if signed else params
        response = self.session.request(method, url, params=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def _public_request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        return self._request(method, path, params=params, signed=False)

    def _futures_public_request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        url = f"https://fapi.binance.com{path}"
        response = self.session.request(method, url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _signed_request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        if not self.api_key or not self.secret:
            raise ValueError("Binance API key and secret are required for signed requests.")
        return self._request(method, path, params=params, signed=True)

    def _listen_key_request(self, method: str, path: str) -> Any:
        if not self.api_key:
            raise ValueError("Binance API key is required for listen key requests.")
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, timeout=10)
        response.raise_for_status()
        return response.json()

    def _signed_payload(self, params: Optional[dict] = None) -> Dict[str, Any]:
        payload = dict(params or {})
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("recvWindow", self.recv_window)

        query = urlencode(payload, doseq=True, quote_via=quote)
        payload["signature"] = self._sign_query(query)
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
            path = "/fapi/v1/exchangeInfo" if self.trading_mode == "futures" else "/api/v3/exchangeInfo"
            self._exchange_info_cache = self._public_request("GET", path)
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

    def to_display_symbol(self, symbol: str) -> str:
        m_symbol = self.get_market_symbol(symbol)
        info = self._symbol_info(m_symbol)
        if info:
            base = info.get("baseAsset")
            quote = info.get("quoteAsset")
            if base and quote:
                return f"{base}/{quote}"
        if "/" in symbol:
            return symbol.strip().upper()
        return symbol.upper().strip()

    @staticmethod
    def _floor_to_step(value: float, step: str) -> str:
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        if d_step == 0:
            return format(d_value, "f")
        quantized = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return format(quantized.normalize(), "f")

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        if amount <= 0:
            return "0"
        info = self._symbol_info(symbol)
        step = "0.000001"
        if info:
            for f in info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    step = f.get("stepSize", step)
                    break
        return self._floor_to_step(amount, step)

    def _validate_order_quantity(self, symbol: str, amount: float) -> tuple[Optional[str], Optional[str]]:
        quantity = self.amount_to_precision(symbol, amount)
        try:
            if Decimal(quantity) <= 0:
                return None, "Order amount rounds to zero"
        except Exception:
            return None, "Unable to validate order amount"
        return quantity, None

    def price_to_precision(self, symbol: str, price: float) -> str:
        info = self._symbol_info(symbol)
        tick = "0.01"
        if info:
            for f in info.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = f.get("tickSize", tick)
                    break
        return self._floor_to_step(price, tick)

    def _market_path(self, resource: str) -> str:
        prefix = "/fapi/v1" if self.trading_mode == "futures" else "/api/v3"
        return f"{prefix}/{resource}"

    def _account_path(self, resource: str) -> str:
        if self.trading_mode == "futures":
            if resource == "balance":
                return "/fapi/v2/balance"
            if resource == "positions":
                return "/fapi/v2/positionRisk"
            return f"/fapi/v1/{resource}"
        if resource == "balance":
            return "/api/v3/account"
        return f"/api/v3/{resource}"

    # ------------------------------------------------------------------
    # Symbol and market data
    # ------------------------------------------------------------------
    def get_market_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper().strip()

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        raw_ohlcv = self._public_request(
            "GET",
            self._market_path("klines"),
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
        last_price = self._public_request("GET", self._market_path("ticker/price"), {"symbol": m_symbol})
        book_ticker = self._public_request("GET", self._market_path("ticker/bookTicker"), {"symbol": m_symbol})

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
            self._market_path("depth"),
            {"symbol": self.get_market_symbol(symbol), "limit": limit},
        )

    def get_real_delta(self, symbol: str, since_ms: int) -> dict:
        try:
            trades = self._public_request(
                "GET",
                self._market_path("aggTrades"),
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
    # Account, positions, and orders
    # ------------------------------------------------------------------
    def get_account_balance(self) -> Dict:
        if self.paper_trading:
            return {
                "free": {"USDT": self._paper_balance},
                "locked": {"USDT": 0.0},
                "paper": True,
                "mode": self.trading_mode,
            }

        if self.trading_mode == "futures":
            account = self._signed_request("GET", self._account_path("balance"))
            free = {
                row["asset"]: float(row.get("availableBalance", row.get("balance", 0.0)))
                for row in account
            }
            locked = {row["asset"]: float(row.get("balance", 0.0)) for row in account}
            return {"free": free, "locked": locked, "mode": "futures"}

        account = self._signed_request("GET", self._account_path("balance"))
        free = {b["asset"]: float(b["free"]) for b in account.get("balances", [])}
        locked = {b["asset"]: float(b["locked"]) for b in account.get("balances", [])}
        account["free"] = free
        account["locked"] = locked
        account["mode"] = "spot"
        return account

    def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> Dict:
        if self.trading_mode != "futures" or self.paper_trading or not (self.api_key and self.secret):
            return {"mode": self.trading_mode, "paper": self.paper_trading, "leverage": leverage or self.leverage}

        return self._signed_request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": self.get_market_symbol(symbol), "leverage": int(leverage or self.leverage)},
        )

    def set_margin_type(self, symbol: str, margin_type: Optional[str] = None) -> Dict:
        if self.trading_mode != "futures" or self.paper_trading or not (self.api_key and self.secret):
            return {"mode": self.trading_mode, "paper": self.paper_trading, "marginType": margin_type or self.margin_type}

        try:
            return self._signed_request(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": self.get_market_symbol(symbol), "marginType": (margin_type or self.margin_type).upper()},
            )
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 400:
                body = response.text or ""
                if "No need to change margin type" in body:
                    return {"status": "unchanged", "marginType": (margin_type or self.margin_type).upper()}
            raise

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        if self.trading_mode != "futures":
            return []

        rows = self._signed_request("GET", self._account_path("positions"))
        m_symbol = self.get_market_symbol(symbol) if symbol else None
        positions = []
        for row in rows:
            market_symbol = row.get("symbol")
            if m_symbol and market_symbol != m_symbol:
                continue

            position_amt = float(row.get("positionAmt", 0.0))
            if abs(position_amt) <= 1e-12:
                continue

            position_side = row.get("positionSide", "BOTH").upper()
            side = "long" if (position_side == "LONG" or position_amt > 0) else "short"
            positions.append(
                {
                    "symbol": self.to_display_symbol(market_symbol),
                    "market_symbol": market_symbol,
                    "side": side,
                    "amount": abs(position_amt),
                    "entry_price": float(row.get("entryPrice", 0.0)),
                    "unrealized_pnl": float(row.get("unRealizedProfit", 0.0)),
                    "leverage": float(row.get("leverage", 0.0)),
                    "margin_type": row.get("marginType", ""),
                    "position_side": position_side,
                }
            )
        return positions

    def _futures_order_side(self, side: str) -> str:
        return side.upper().strip()

    def _position_side_param(self, side: str) -> Optional[str]:
        if self.trading_mode != "futures":
            return None
        if self.position_mode != "HEDGE":
            return None
        return "LONG" if side.lower() == "buy" else "SHORT"

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        quantity, quantity_error = self._validate_order_quantity(m_symbol, amount)
        if quantity_error:
            return {"error": quantity_error, "symbol": m_symbol, "side": side.upper(), "type": "MARKET"}

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
                "reduceOnly": reduce_only,
                "positionSide": position_side or self._position_side_param(side),
            }

        if self.trading_mode == "futures":
            params: Dict[str, Any] = {
                "symbol": m_symbol,
                "side": self._futures_order_side(side),
                "type": "MARKET",
                "quantity": quantity,
                "newOrderRespType": "RESULT",
            }
            if reduce_only and self.position_mode != "HEDGE":
                params["reduceOnly"] = "true"
            if self.position_mode == "HEDGE":
                params["positionSide"] = (position_side or self._position_side_param(side) or "").upper()
            return self._signed_request("POST", "/fapi/v1/order", params)

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

    def _place_entry_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: Optional[str] = None,
    ) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        quantity, quantity_error = self._validate_order_quantity(m_symbol, amount)
        if quantity_error:
            return {"error": quantity_error, "symbol": m_symbol, "side": side.upper(), "type": "LIMIT"}
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

        if self.trading_mode == "futures":
            return self.place_market_order(symbol, side, amount, reduce_only=False, position_side=position_side)

        # Use market semantics for live spot entries so the local position state
        # remains aligned with the actual exchange fill. Limit entries can sit
        # open indefinitely and break the protective-exit flow.
        return self.place_market_order(symbol, side, amount, reduce_only=False)

    def _place_spot_oco_exit(self, symbol: str, side: str, quantity: str, stop_loss: float, take_profit: float) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        if side.lower() == "buy":
            above_price = self.price_to_precision(m_symbol, stop_loss)
            above_stop = self.price_to_precision(m_symbol, stop_loss)
            below_price = self.price_to_precision(m_symbol, take_profit)
            below_stop = self.price_to_precision(m_symbol, take_profit)
        else:
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

    def _place_futures_exit_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        trigger_price: float,
        order_type: str,
        reduce_only: bool = True,
    ) -> Dict:
        if self.paper_trading:
            return {
                "symbol": self.get_market_symbol(symbol),
                "type": order_type,
                "status": "NEW",
                "paper": True,
                "side": side.upper(),
                "quantity": quantity,
                "triggerPrice": trigger_price,
                "reduceOnly": reduce_only,
            }

        params: Dict[str, Any] = {
            "symbol": self.get_market_symbol(symbol),
            "side": side.upper(),
            "type": order_type,
            "quantity": quantity,
            "stopPrice": self.price_to_precision(symbol, trigger_price),
            "workingType": "MARK_PRICE",
            "priceProtect": "TRUE",
            "newOrderRespType": "RESULT",
        }
        if self.position_mode != "HEDGE":
            params["reduceOnly"] = "true" if reduce_only else "false"
        if self.position_mode == "HEDGE":
            params["positionSide"] = "LONG" if side.lower() == "sell" else "SHORT"
        return self._signed_request("POST", "/fapi/v1/order", params)

    def place_exit_orders(
        self,
        symbol: str,
        position_side: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        quantity, quantity_error = self._validate_order_quantity(m_symbol, amount)
        if quantity_error:
            return {"error": quantity_error, "symbol": m_symbol}
        exit_side = "sell" if position_side.lower() == "long" else "buy"

        if self.trading_mode == "futures":
            return {
                "symbol": m_symbol,
                "sl": self._place_futures_exit_order(symbol, exit_side, quantity, stop_loss, "STOP_MARKET", reduce_only=True),
                "tp": self._place_futures_exit_order(symbol, exit_side, quantity, take_profit, "TAKE_PROFIT_MARKET", reduce_only=True),
                "strategy": "Futures bracket exits",
            }

        return {
            "symbol": m_symbol,
            "oco": self._place_spot_oco_exit(symbol, exit_side, quantity, stop_loss, take_profit),
            "strategy": "Spot OCO protection",
        }

    def place_order_with_sl_tp(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        attach_exit_orders: bool = True,
        position_side: Optional[str] = None,
    ) -> Dict:
        try:
            entry_order = self._place_entry_order(symbol, side, amount, price, position_side=position_side)
            if entry_order.get("error"):
                return {"entry": entry_order, "error": entry_order["error"]}
            executed_qty = Decimal(str(entry_order.get("executedQty", entry_order.get("origQty", "0")) or "0"))
            order_status = str(entry_order.get("status", "")).upper()

            if executed_qty <= 0:
                return {"entry": entry_order, "error": "Entry not filled"}

            if order_status and order_status not in {"FILLED", "PARTIALLY_FILLED", "NEW"}:
                return {"entry": entry_order, "error": f"Unexpected entry status: {order_status}"}

            exit_side = "sell" if side.lower() == "buy" else "buy"
            if self.trading_mode == "futures":
                if not attach_exit_orders:
                    return {
                        "entry": entry_order,
                        "strategy": "Futures market order",
                        "position_side": (position_side or self._position_side_param(side) or "").lower(),
                    }

                exit_position_side = "long" if side.lower() == "buy" else "short"
                try:
                    exit_orders = self.place_exit_orders(symbol, exit_position_side, float(executed_qty), stop_loss, take_profit)
                except Exception as exc:
                    log.warning("Failed to place protective exits for %s: %s", symbol, exc)
                    exit_orders = {"error": str(exc)}

                exit_error = exit_orders.get("error")
                if exit_error and not self.paper_trading:
                    try:
                        self.cancel_all_open_orders(symbol)
                    except Exception as exc:
                        log.warning("Failed to cancel open orders during protective unwind for %s: %s", symbol, exc)
                    unwind_side = "sell" if side.lower() == "buy" else "buy"
                    try:
                        unwind = self.place_market_order(
                            symbol,
                            unwind_side,
                            float(executed_qty),
                            reduce_only=True,
                            position_side=("LONG" if side.lower() == "buy" else "SHORT"),
                        )
                    except Exception as exc:
                        unwind = {"error": str(exc)}
                    return {
                        "entry": entry_order,
                        "sl": exit_orders.get("sl"),
                        "tp": exit_orders.get("tp"),
                        "strategy": "Futures entry + protective exits",
                        "position_side": exit_position_side,
                        "exit_error": exit_error,
                        "unwind_result": unwind,
                        "error": f"Protective exits failed; position auto-unwind attempted: {exit_error}",
                    }
                return {
                    "entry": entry_order,
                    "sl": exit_orders.get("sl"),
                    "tp": exit_orders.get("tp"),
                    "strategy": "Futures entry + protective exits",
                    "position_side": exit_position_side,
                    "exit_error": exit_orders.get("error"),
                }

            try:
                oco = self._place_spot_oco_exit(
                    symbol,
                    exit_side,
                    self.amount_to_precision(symbol, float(executed_qty)),
                    stop_loss,
                    take_profit,
                )
            except Exception as exc:
                log.warning("Failed to place spot OCO exits for %s: %s", symbol, exc)
                oco = {"error": str(exc)}
            return {
                "entry": entry_order,
                "sl": oco,
                "tp": oco,
                "strategy": "Spot OCO protection",
                "exit_error": oco.get("error"),
            }
        except Exception as e:
            log.error("place_order_with_sl_tp error: %s", e)
            return {"error": str(e)}

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        params = {"symbol": self.get_market_symbol(symbol)}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        path = "/fapi/v1/order" if self.trading_mode == "futures" else "/api/v3/order"
        return self._signed_request("DELETE", path, params)

    def cancel_all_open_orders(self, symbol: str) -> Dict:
        params = {"symbol": self.get_market_symbol(symbol)}
        path = "/fapi/v1/allOpenOrders" if self.trading_mode == "futures" else "/api/v3/openOrders"
        return self._signed_request("DELETE", path, params)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {}
        if symbol:
            params["symbol"] = self.get_market_symbol(symbol)
        path = "/fapi/v1/openOrders" if self.trading_mode == "futures" else "/api/v3/openOrders"
        return self._signed_request("GET", path, params)

    def get_my_trades(self, symbol: str, from_id: Optional[int] = None, limit: int = 1000) -> List[Dict]:
        params: Dict[str, Any] = {
            "symbol": self.get_market_symbol(symbol),
            "limit": min(max(int(limit or 1), 1), 1000),
        }
        if from_id is not None:
            params["fromId"] = int(from_id)

        path = "/fapi/v1/userTrades" if self.trading_mode == "futures" else "/api/v3/myTrades"
        raw_trades = self._signed_request("GET", path, params)
        if not raw_trades:
            return []

        normalized = []
        for trade in raw_trades:
            side_value = trade.get("side")
            if not side_value:
                if "isBuyer" in trade:
                    side_value = "BUY" if bool(trade.get("isBuyer")) else "SELL"
                elif "buyer" in trade:
                    side_value = "BUY" if bool(trade.get("buyer")) else "SELL"
                else:
                    side_value = ""
            normalized.append(
                {
                    "id": int(trade.get("id", trade.get("tradeId", 0)) or 0),
                    "orderId": int(trade.get("orderId", 0) or 0),
                    "symbol": self.to_display_symbol(trade.get("symbol", symbol)),
                    "side": str(side_value).upper(),
                    "qty": float(trade.get("qty", trade.get("q", 0.0)) or 0.0),
                    "price": float(trade.get("price", trade.get("p", 0.0)) or 0.0),
                    "realizedPnl": float(trade.get("realizedPnl", 0.0) or 0.0),
                    "commission": float(trade.get("commission", 0.0) or 0.0),
                    "commissionAsset": trade.get("commissionAsset"),
                    "time": int(trade.get("time", trade.get("T", 0)) or 0),
                    "quoteQty": float(trade.get("quoteQty", trade.get("quoteQty", 0.0)) or 0.0),
                    "positionSide": str(trade.get("positionSide", trade.get("positionSide", "")) or "").upper(),
                }
            )
        return normalized

    def get_open_interest(self, symbol: str) -> Dict:
        m_symbol = self.get_market_symbol(symbol)
        try:
            return self._futures_public_request("GET", "/fapi/v1/openInterest", {"symbol": m_symbol})
        except Exception as exc:
            log.warning("Open interest snapshot unavailable for %s: %s", symbol, exc)
            return {}

    def get_open_interest_history(self, symbol: str, limit: int = 10, period: str = "5m") -> List[Dict]:
        m_symbol = self.get_market_symbol(symbol)
        params = {
            "symbol": m_symbol,
            "period": period,
            "limit": min(max(int(limit or 1), 1), 500),
        }
        try:
            raw = self._futures_public_request("GET", "/futures/data/openInterestHist", params)
        except Exception as exc:
            log.warning("Open interest history unavailable for %s: %s", symbol, exc)
            return []

        normalized: List[Dict] = []
        for row in raw or []:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "symbol": m_symbol,
                    "period": period,
                    "timestamp": int(row.get("timestamp", row.get("time", 0)) or 0),
                    "openInterestAmount": float(
                        row.get(
                            "sumOpenInterest",
                            row.get("openInterest", row.get("openInterestAmount", 0.0)),
                        )
                        or 0.0
                    ),
                    "openInterestValue": float(
                        row.get(
                            "sumOpenInterestValue",
                            row.get("openInterestValue", 0.0),
                        )
                        or 0.0
                    ),
                    "raw": row,
                }
            )
        return normalized

    def get_listen_key(self) -> Optional[str]:
        if self.paper_trading or not (self.api_key and self.secret):
            return None

        if self.trading_mode == "futures":
            data = self._listen_key_request("POST", "/fapi/v1/listenKey")
            return data.get("listenKey")

        log.warning("Spot user data now uses the Binance WebSocket API; listenKey is not used.")
        return None

    def keep_alive_listen_key(self, listen_key: str) -> bool:
        if self.trading_mode == "futures":
            try:
                self._listen_key_request("PUT", "/fapi/v1/listenKey")
                return True
            except Exception as e:
                log.warning("Failed to refresh futures listen key: %s", e)
                return False
        return False
