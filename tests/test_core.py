import unittest

import pandas as pd

from engine.tools import TradingTools
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


if __name__ == "__main__":
    unittest.main()
