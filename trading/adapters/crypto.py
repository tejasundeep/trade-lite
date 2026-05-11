from typing import Dict, List, Optional
import ccxt
import pandas as pd
import logging

log = logging.getLogger(__name__)

class CCXTCryptoAdapter:
    def __init__(self, exchange_id: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None, paper_trading: bool = True):
        self.exchange = getattr(ccxt, exchange_id)({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        })
        if paper_trading:
            self.exchange.set_sandbox_mode(True)
        
        try:
            self.exchange.load_markets()
        except Exception as e:
            log.error(f"Failed to load markets: {e}")
            
    def get_market_symbol(self, symbol: str) -> str:
        """Map user symbol (BTC/USDT) to exchange symbol (BTC/USDT:USDT)."""
        if symbol in self.exchange.markets:
            return symbol
        # Try to find a match (e.g. BTC/USDT -> BTC/USDT:USDT)
        for s in self.exchange.markets:
            if s.startswith(symbol):
                return s
        return symbol

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        raw_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw_ohlcv: return pd.DataFrame()
        
        df = pd.DataFrame(raw_ohlcv)
        # CCXT's fetch_ohlcv for Binance returns 10 columns:
        # [timestamp, open, high, low, close, volume, quote_vol, trade_count, taker_base_vol, taker_quote_vol]
        if "binance" in self.exchange.id and df.shape[1] >= 9:
            df = df[[0, 1, 2, 3, 4, 5, 8]]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]
        else:
            # Fallback for other exchanges or unexpected formats
            df = df.iloc[:, :6]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
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

    def _is_futures(self) -> bool:
        return "fapi" in str(type(self.exchange)).lower() or "usdm" in self.exchange.id.lower() or "dpm" in self.exchange.id.lower()

    def place_order_with_sl_tp(self, symbol: str, side: str, amount: float, price: float, stop_loss: float, take_profit: float) -> Dict:
        try:
            is_futures = self._is_futures()
            
            # Entry Order
            entry_order = self.exchange.create_order(
                symbol, "limit", side, 
                self.exchange.amount_to_precision(symbol, amount), 
                self.exchange.price_to_precision(symbol, price)
            )
            
            exit_side = "sell" if side == "buy" else "buy"
            
            # SL Order
            sl_params = {'stopPrice': self.exchange.price_to_precision(symbol, stop_loss)}
            if is_futures:
                sl_params['reduceOnly'] = True
            
            sl_order = self.exchange.create_order(
                symbol=symbol, type='stop_market' if is_futures else 'stop_loss_limit', 
                side=exit_side,
                amount=self.exchange.amount_to_precision(symbol, amount),
                price=self.exchange.price_to_precision(symbol, stop_loss) if not is_futures else None,
                params=sl_params
            )
            
            # TP Order
            tp_params = {}
            if is_futures:
                tp_params['reduceOnly'] = True
                
            tp_order = self.exchange.create_order(
                symbol=symbol, type='limit', side=exit_side,
                amount=self.exchange.amount_to_precision(symbol, amount),
                price=self.exchange.price_to_precision(symbol, take_profit),
                params=tp_params
            )
            
            return {"entry": entry_order, "sl": sl_order, "tp": tp_order, "strategy": "Dual-Leg Protection"}
        except Exception as e:
            log.error(f"place_order_with_sl_tp error: {e}")
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

    def get_listen_key(self) -> Optional[str]:
        """Fetch a new listenKey from Binance."""
        try:
            # For binanceusdm and binance, fapiPrivatePostListenKey and v3PrivatePostListenKey are standard.
            if hasattr(self.exchange, "fapiPrivatePostListenKey"):
                res = self.exchange.fapiPrivatePostListenKey()
            elif hasattr(self.exchange, "v3PrivatePostListenKey"):
                res = self.exchange.v3PrivatePostListenKey()
            else:
                # Fallback to direct request if implicit method fails
                res = self.exchange.privatePostListenKey() if hasattr(self.exchange, "privatePostListenKey") else {}
            
            return res.get("listenKey")
        except Exception as e:
            log.error(f"Error fetching listenKey: {e}")
        return None

    def keep_alive_listen_key(self, listen_key: str) -> bool:
        """Extend the validity of a listenKey."""
        try:
            if hasattr(self.exchange, "fapiPrivatePutListenKey"):
                self.exchange.fapiPrivatePutListenKey({"listenKey": listen_key})
            else:
                self.exchange.v3PrivatePutListenKey({"listenKey": listen_key})
            return True
        except Exception as e:
            log.error(f"Error extending listenKey: {e}")
        return False
