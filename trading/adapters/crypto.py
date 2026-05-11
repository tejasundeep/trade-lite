from typing import Dict, List, Optional
import ccxt
import pandas as pd

class CCXTCryptoAdapter:
    def __init__(self, exchange_id: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None, paper_trading: bool = True):
        self.exchange = getattr(ccxt, exchange_id)({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        })
        if paper_trading:
            self.exchange.set_sandbox_mode(True)

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Mocking taker volume for non-Binance or simplified analysis if needed
        # But for 1:1 we should try to get real taker volume if available
        if "binance" in self.exchange.id:
            # Binance raw OHLCV has more columns
            raw_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw_ohlcv)
            df = df[[0, 1, 2, 3, 4, 5, 9]]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]
        else:
            df["taker_buy_volume"] = df["volume"] * 0.5
            
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_real_delta(self, symbol: str, since_ms: int) -> dict:
        try:
            trades = self.exchange.fetch_trades(symbol, since=since_ms, limit=1000)
            if not trades: return {"delta": 0, "buy_vol": 0, "sell_vol": 0, "trades_count": 0}
            buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
            sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
            avg_size = sum(t['amount'] for t in trades) / len(trades)
            large_trades = [t for t in trades if t['amount'] > avg_size * 5]
            return {
                "delta": buy_vol - sell_vol, "buy_vol": buy_vol, "sell_vol": sell_vol,
                "trades_count": len(trades), "institutional_activity": len(large_trades) > 0,
                "large_trade_count": len(large_trades)
            }
        except Exception as e:
            return {"delta": 0, "error": str(e)}

    def place_order_with_sl_tp(self, symbol: str, side: str, amount: float, price: float, stop_loss: float, take_profit: float) -> Dict:
        # 1:1 logic from original
        try:
            entry_order = self.exchange.create_order(
                symbol, "limit", side, 
                self.exchange.amount_to_precision(symbol, amount), 
                self.exchange.price_to_precision(symbol, price)
            )
            exit_side = "sell" if side == "buy" else "buy"
            sl_params = {'stopPrice': self.exchange.price_to_precision(symbol, stop_loss)}
            sl_order = self.exchange.create_order(
                symbol=symbol, type='stop_market', side=exit_side,
                amount=self.exchange.amount_to_precision(symbol, amount),
                params=sl_params
            )
            tp_order = self.exchange.create_order(
                symbol=symbol, type='limit', side=exit_side,
                amount=self.exchange.amount_to_precision(symbol, amount),
                price=self.exchange.price_to_precision(symbol, take_profit)
            )
            return {"entry": entry_order, "sl": sl_order, "tp": tp_order, "strategy": "Dual-Leg Protection"}
        except Exception as e:
            return {"error": str(e)}
            
    def get_ticker(self, symbol: str) -> Dict:
        ticker = self.exchange.fetch_ticker(symbol)
        return {"symbol": symbol, "price": ticker.get("last", 0.0), "timestamp": ticker.get("timestamp")}

    def get_account_balance(self) -> Dict:
        return self.exchange.fetch_balance()

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def get_open_interest_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        try:
            symbol_f = symbol.replace("/", "")
            return self.exchange.fapiPublicGetOpenInterestHist({"symbol": symbol_f, "period": "1h", "limit": limit})
        except: return []
