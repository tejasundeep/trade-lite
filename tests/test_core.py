import asyncio
import os
import unittest
from unittest.mock import patch

import pandas as pd

from bot import TradeXProClone
from db import Position, get_session
from engine.tools import TradingTools
from indicators.risk import calculate_risk_parameters
from trading.streamer import BinanceStreamer
from trading.adapters.crypto import CCXTCryptoAdapter


class _FakeAdapter:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self.paper_trading = True
        self.trading_mode = "futures"

    def get_market_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")

    def get_market_data(self, symbol: str, timeframe: str, limit: int):
        return self._frame.copy()

    def get_account_balance(self):
        return {"free": {"USDT": 10000.0}, "locked": {"USDT": 0.0}, "paper": True, "mode": "futures"}


class CoreBehaviorTests(unittest.TestCase):
    def test_paper_trading_balance_isolated_from_credentials(self):
        adapter = CCXTCryptoAdapter(
            exchange_id="binance",
            api_key="live_key",
            secret="live_secret",
            paper_trading=True,
            trading_mode="futures",
        )

        balance = adapter.get_account_balance()
        self.assertTrue(balance["paper"])
        self.assertEqual(balance["free"]["USDT"], 10000.0)

    def test_market_data_cleanup_sorts_and_drops_invalid_rows(self):
        raw = pd.DataFrame(
            [
                {"timestamp": "2024-01-02T00:00:00Z", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 7},
                {"timestamp": "invalid", "open": 999, "high": 999, "low": 999, "close": 999, "volume": 999},
                {"timestamp": "2024-01-01T00:00:00Z", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 5},
            ]
        )
        tools = TradingTools(api_key="x", secret="y", paper_trading=True, exchange_id="binance", trading_mode="futures")
        tools.adapter = _FakeAdapter(raw)

        cleaned = tools.get_market_data("BTC/USDT", timeframe="1h", limit=10)

        self.assertEqual(len(cleaned), 2)
        self.assertTrue(cleaned["timestamp"].is_monotonic_increasing)
        self.assertIn("EMA_50", cleaned.columns)
        self.assertIn("EMA_200", cleaned.columns)
        self.assertFalse(cleaned[["open", "high", "low", "close", "volume"]].isna().any().any())

    def test_daily_loss_limit_uses_configured_threshold(self):
        tools = TradingTools(
            api_key="x",
            secret="y",
            paper_trading=True,
            exchange_id="binance",
            trading_mode="futures",
            max_daily_loss_pct=0.02,
        )

        class _TradeBlockAdapter:
            paper_trading = True
            trading_mode = "futures"

            def get_market_symbol(self, symbol: str) -> str:
                return symbol.replace("/", "")

        tools.adapter = _TradeBlockAdapter()
        tools._get_day_balance_snapshot = lambda: 1000.0
        tools.get_todays_realized_pnl = lambda: -25.0

        result = tools.execute_trade(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            reason="test",
        )

        self.assertIn("error", result)
        self.assertIn("Daily loss limit reached", result["error"])

    def test_primary_symbol_is_always_in_symbol_universe(self):
        env = {
            "SYMBOL": "ADA/USDT",
            "SYMBOLS": "BTC/USDT,ETH/USDT",
            "PAPER_TRADING": "true",
            "FUTURES_LEVERAGE": "not-an-integer",
        }
        with patch.dict(os.environ, env, clear=False):
            bot = TradeXProClone()

        self.assertIn("ADA/USDT", bot.symbols)
        self.assertEqual(bot.tools.adapter.leverage, 3)

    def test_invalid_stop_loss_is_rejected_by_risk_sizing(self):
        result = calculate_risk_parameters(
            free_balance=1000.0,
            current_price=100.0,
            confidence_score=0.8,
            stop_loss=100.0,
        )

        self.assertEqual(result["action"], "lock")
        self.assertIn("Invalid stop loss", result["reason"])

    def test_zero_sized_order_is_rejected_before_submission(self):
        adapter = CCXTCryptoAdapter(
            exchange_id="binance",
            paper_trading=True,
            trading_mode="futures",
        )

        result = adapter.place_market_order("BTC/USDT", "buy", 0.0)

        self.assertIn("error", result)
        self.assertIn("rounds to zero", result["error"])

    def test_reconcile_preserves_local_positions_when_remote_snapshot_fails(self):
        class _BrokenAdapter:
            paper_trading = False
            trading_mode = "futures"

            def get_open_positions(self):
                raise RuntimeError("remote unavailable")

            def get_market_symbol(self, symbol: str) -> str:
                return symbol.replace("/", "")

            def get_my_trades(self, *args, **kwargs):
                return []

            def get_open_orders(self, *args, **kwargs):
                return []

            def get_market_data(self, *args, **kwargs):
                return pd.DataFrame()

            def get_ticker(self, *args, **kwargs):
                return {"price": 0.0, "bid": 0.0, "ask": 0.0}

        tools = TradingTools(api_key="x", secret="y", paper_trading=False, exchange_id="binance", trading_mode="futures")
        tools.adapter = _BrokenAdapter()

        session = get_session()
        try:
            session.query(Position).filter_by(symbol="XRP/USDT").delete()
            session.add(
                Position(
                    symbol="XRP/USDT",
                    avg_price=1.0,
                    amount=10.0,
                    side="long",
                    stop_loss=0.9,
                    take_profit=1.1,
                    tp1_hit=False,
                )
            )
            session.commit()
        finally:
            session.close()

        summary = tools.reconcile_execution_state()

        self.assertTrue(summary["reconciled"])
        self.assertTrue(any(item["symbol"] == "XRP/USDT" and item["position_action"] == "preserved" for item in summary["symbols"]))

        session = get_session()
        try:
            pos = session.query(Position).filter_by(symbol="XRP/USDT").first()
            self.assertIsNotNone(pos)
        finally:
            session.query(Position).filter_by(symbol="XRP/USDT").delete()
            session.commit()
            session.close()

    def test_startup_reconciliation_records_status(self):
        bot = TradeXProClone()
        bot.tools.adapter.paper_trading = False

        calls = {"count": 0}

        def _fake_reconcile(symbols):
            calls["count"] += 1
            return {"reconciled": True, "cursor_updated": False, "symbols": [{"symbol": s} for s in symbols]}

        bot.tools.reconcile_execution_state = _fake_reconcile
        asyncio.run(bot._startup_reconcile())

        self.assertEqual(calls["count"], 1)
        self.assertIn("Startup reconciled", bot.stats["last_action"])

    def test_circuit_breaker_trips_on_toxic_spread(self):
        from engine.safety import TradingCircuitBreaker, TradingSafetyConfig

        class _SpreadStreamer:
            prices = {"BTC/USDT": 100.0}

            def market_age_seconds(self, symbol: str) -> float:
                return 1.0

            def get_spread(self, symbol: str) -> float:
                return 2.0

        breaker = TradingCircuitBreaker(
            TradingSafetyConfig(max_spread_bps=15.0, max_stale_seconds=20),
            streamer=_SpreadStreamer(),
        )
        result = breaker.evaluate(["BTC/USDT"], balance=1000.0, day_balance=1000.0)

        self.assertFalse(result["allowed"])
        self.assertIn("toxic_spread", result["reason"])

    def test_spot_user_data_ws_uses_canonical_url(self):
        adapter = CCXTCryptoAdapter(
            exchange_id="binance",
            paper_trading=True,
            trading_mode="spot",
        )

        self.assertEqual(adapter.ws_user_data_url, "wss://ws-api.binance.com/ws-api/v3")

    def test_market_symbol_normalizes_to_display_symbol(self):
        adapter = CCXTCryptoAdapter(
            exchange_id="binance",
            paper_trading=True,
            trading_mode="futures",
        )
        adapter._exchange_info_cache = {
            "symbols": [
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
                {"symbol": "ADAUSDT", "baseAsset": "ADA", "quoteAsset": "USDT"},
            ]
        }

        self.assertEqual(adapter.to_display_symbol("BTCUSDT"), "BTC/USDT")
        self.assertEqual(adapter.to_display_symbol("ADA/USDT"), "ADA/USDT")

    def test_hedge_mode_market_orders_do_not_send_reduce_only(self):
        adapter = CCXTCryptoAdapter(
            exchange_id="binance",
            paper_trading=False,
            trading_mode="futures",
            position_mode="HEDGE",
        )

        captured = {}

        def _fake_signed_request(method, path, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["params"] = params
            return {"status": "ok"}

        adapter._signed_request = _fake_signed_request
        adapter.amount_to_precision = lambda symbol, amount: "0.1"

        adapter.place_market_order("BTC/USDT", "sell", 0.1, reduce_only=True, position_side="LONG")

        self.assertEqual(captured["path"], "/fapi/v1/order")
        self.assertEqual(captured["params"]["positionSide"], "LONG")
        self.assertNotIn("reduceOnly", captured["params"])

    def test_closing_trade_skips_protective_exits(self):
        tools = TradingTools(api_key="x", secret="y", paper_trading=True, exchange_id="binance", trading_mode="futures")
        tools.get_todays_realized_pnl = lambda: 0.0
        tools._get_day_balance_snapshot = lambda: 10000.0

        calls = {"count": 0, "attach_exit_orders": None, "position_side": None}

        def _fake_place_order_with_sl_tp(*args, **kwargs):
            calls["count"] += 1
            calls["attach_exit_orders"] = kwargs.get("attach_exit_orders")
            calls["position_side"] = kwargs.get("position_side")
            return {"entry": {"executedQty": "1", "status": "FILLED"}}

        tools.adapter.place_order_with_sl_tp = _fake_place_order_with_sl_tp

        session = get_session()
        try:
            session.query(Position).filter_by(symbol="SOL/USDT").delete()
            session.add(
                Position(
                    symbol="SOL/USDT",
                    avg_price=100.0,
                    amount=1.0,
                    side="long",
                    stop_loss=90.0,
                    take_profit=120.0,
                    tp1_hit=False,
                )
            )
            session.commit()
        finally:
            session.close()

        try:
            result = tools.execute_trade(
                symbol="SOL/USDT",
                side="sell",
                amount=1.0,
                price=95.0,
                stop_loss=90.0,
                take_profit=120.0,
                reason="close",
            )
            self.assertNotIn("error", result)
            self.assertEqual(calls["count"], 1)
            self.assertFalse(calls["attach_exit_orders"])
            self.assertEqual(calls["position_side"], "LONG")
        finally:
            session = get_session()
            try:
                session.query(Position).filter_by(symbol="SOL/USDT").delete()
                session.commit()
            finally:
                session.close()

    def test_futures_account_update_includes_positions(self):
        streamer = BinanceStreamer(["BTC/USDT"])
        payload = {
            "e": "ACCOUNT_UPDATE",
            "E": 1,
            "T": 1,
            "a": {
                "m": "ORDER",
                "B": [{"a": "USDT", "wb": "120.5", "cw": "100.0", "bc": "20.5"}],
                "P": [{"s": "BTCUSDT", "pa": "0.25", "ep": "65000.0", "bep": "65200.0", "up": "12.5", "mt": "isolated", "iw": "50.0", "ps": "LONG"}],
            },
        }

        update = streamer._normalize_account_event(payload)

        self.assertEqual(update["event"], "ACCOUNT_UPDATE")
        self.assertEqual(update["balances"][0]["free"], 120.5)
        self.assertEqual(update["positions"][0]["symbol"], "BTC/USDT")
        self.assertEqual(update["positions"][0]["side"], "long")
        self.assertEqual(update["positions"][0]["amount"], 0.25)

    def test_account_update_applies_position_snapshot(self):
        bot = TradeXProClone()
        session = get_session()
        try:
            session.query(Position).filter_by(symbol="ADA/USDT").delete()
            session.commit()
        finally:
            session.close()

        asyncio.run(
            bot._handle_account_update(
                {
                    "event": "ACCOUNT_UPDATE",
                    "balances": [{"asset": "USDT", "free": 1234.5, "locked": 0.0}],
                    "positions": [
                        {
                            "symbol": "ADA/USDT",
                            "side": "long",
                            "amount": 50.0,
                            "entry_price": 0.45,
                            "breakeven_price": 0.452,
                        }
                    ],
                }
            )
        )

        self.assertEqual(bot.cached_balance, 1234.5)
        session = get_session()
        try:
            pos = session.query(Position).filter_by(symbol="ADA/USDT").first()
            self.assertIsNotNone(pos)
            self.assertEqual(pos.amount, 50.0)
            self.assertEqual(pos.side, "long")
            self.assertEqual(pos.avg_price, 0.45)
        finally:
            session.query(Position).filter_by(symbol="ADA/USDT").delete()
            session.commit()
            session.close()


if __name__ == "__main__":
    unittest.main()
