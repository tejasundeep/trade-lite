import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, date
import json

from trading.adapters.crypto import CCXTCryptoAdapter
from indicators.market_context import set_market_data
from db import get_session, Trade, Position, SystemState

log = logging.getLogger(__name__)

_FEE_BPS   = 0.001
_SLIP_BPS  = 0.0005


class TradingTools:
    def __init__(self, api_key: str = None, secret: str = None, paper_trading: bool = True, exchange_id: str = "binance"):
        self.adapter      = CCXTCryptoAdapter(exchange_id=exchange_id, api_key=api_key, secret=secret, paper_trading=paper_trading)
        self._day_balance: Optional[float] = None
        self._snap_date:   Optional[date]  = None

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        m_symbol = self.adapter.get_market_symbol(symbol)
        df = self.adapter.get_market_data(m_symbol, timeframe, limit)
        df["EMA_50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
        set_market_data(df)
        return df

    def get_institutional_levels(self, symbol: str) -> Dict:
        levels = {}
        try:
            m_symbol = self.adapter.get_market_symbol(symbol)
            df_d = self.adapter.get_market_data(m_symbol, "1d", 5)
            if not df_d.empty:
                levels.update({
                    "pdh": float(df_d.iloc[-2]["high"]),
                    "pdl": float(df_d.iloc[-2]["low"]),
                    "pdo": float(df_d.iloc[-2]["open"]),
                    "pdc": float(df_d.iloc[-2]["close"]),
                })
            df_w = self.adapter.get_market_data(m_symbol, "1w", 3)
            if not df_w.empty:
                levels.update({
                    "weekly_open": float(df_w.iloc[-1]["open"]),
                    "pwh": float(df_w.iloc[-2]["high"]),
                    "pwl": float(df_w.iloc[-2]["low"]),
                })
        except Exception: pass
        return levels

    def get_ticker(self, symbol: str) -> Dict:
        m_symbol = self.adapter.get_market_symbol(symbol)
        return self.adapter.get_ticker(m_symbol)

    def get_account_balance(self) -> Dict:
        return self.adapter.get_account_balance()

    def get_balance(self) -> float:
        try:
            bal = self.adapter.get_account_balance()
            # Binance Spot users often hold FDUSD or USDC now
            for asset in ["USDT", "FDUSD", "USDC", "BUSD"]:
                free = float(bal.get("free", {}).get(asset, 0.0))
                if free > 0.01:
                    log.info(f"Found balance: {free:.2f} {asset}")
                    return free
            return 0.0
        except Exception as e:
            log.error(f"get_balance error: {e}")
            return 0.0

    def _get_day_balance_snapshot(self) -> float:
        today = datetime.utcnow().date()
        if self._snap_date == today and self._day_balance is not None:
            return self._day_balance
        session = get_session()
        try:
            key = f"day_balance_{today}"
            row = session.query(SystemState).filter_by(key=key).first()
            if row:
                self._day_balance = float(json.loads(row.value))
            else:
                snap = self.get_balance()
                session.merge(SystemState(key=key, value=json.dumps(snap)))
                session.commit()
                self._day_balance = snap
            self._snap_date = today
            return self._day_balance
        finally:
            session.close()

    # ─── Position Management ──────────────────────────────────────────────────
    def get_open_position(self, symbol: str) -> Optional[Dict]:
        session = get_session()
        try:
            pos = session.query(Position).filter_by(symbol=symbol).first()
            if pos:
                return {"symbol": pos.symbol, "avg_price": pos.avg_price,
                        "amount": pos.amount, "side": pos.side,
                        "stop_loss": pos.stop_loss, "take_profit": pos.take_profit,
                        "tp1_hit": pos.tp1_hit, "trailing_stop": pos.trailing_stop_price}
            return None
        finally:
            session.close()

    def manage_open_positions(self, prices: Dict[str, float], atr_map: Dict[str, float], smc_map: Dict[str, Dict] = None):
        """Scale out at 1:1 R/R, move SL to breakeven+, and trail the remainder structurally."""
        session = get_session()
        smc_map = smc_map or {}
        try:
            positions = session.query(Position).all()
            for pos in positions:
                price = prices.get(pos.symbol)
                atr = atr_map.get(pos.symbol, price * 0.01) if price else 0
                if not price or not atr: continue
                
                risk = abs(pos.avg_price - pos.stop_loss)
                if risk < (pos.avg_price * 0.0001): continue
                
                # 1. Target 1 (1:1 R/R) -> Breakeven+
                if not pos.tp1_hit:
                    tp1_px = pos.avg_price + risk if pos.side == "long" else pos.avg_price - risk
                    is_tp1 = (pos.side == "long" and price >= tp1_px) or (pos.side == "short" and price <= tp1_px)
                    
                    if is_tp1:
                        log.info(f"TP1 for {pos.symbol}. Move to BE+ (fees coverage).")
                        m_symbol = self.adapter.get_market_symbol(pos.symbol)
                        close_amt = pos.amount * 0.5
                        side = "sell" if pos.side == "long" else "buy"
                        self.adapter.place_market_order(m_symbol, side, close_amt)
                        
                        # BE+ (entry + 15bps to cover slippage/fees)
                        be_plus = pos.avg_price * 1.0015 if pos.side == "long" else pos.avg_price * 0.9985
                        pos.amount -= close_amt
                        pos.stop_loss = be_plus
                        pos.tp1_hit = True
                        pos.trailing_stop_price = be_plus
                        session.commit()
                        continue

                # 2. Structural Trailing Stop
                if pos.tp1_hit:
                    # Combined trail: Max(Structural Pivot, ATR Trail)
                    smc = smc_map.get(pos.symbol, {})
                    pd_arr = smc.get("pd_array", {})
                    
                    if pos.side == "long":
                        atr_trail = price - atr * 2.0
                        struct_trail = pd_arr.get("low", 0) # Support pivot
                        potential_trail = max(atr_trail, struct_trail, pos.trailing_stop_price or 0)
                        if potential_trail > pos.trailing_stop_price:
                            pos.trailing_stop_price = potential_trail
                        if price < pos.trailing_stop_price:
                            self._execute_full_close(session, pos, price, "Structural Trail Exit")
                    else:
                        atr_trail = price + atr * 2.0
                        struct_trail = pd_arr.get("high", 999999999) # Resistance pivot
                        potential_trail = min(atr_trail, struct_trail, pos.trailing_stop_price or 999999999)
                        if potential_trail < pos.trailing_stop_price:
                            pos.trailing_stop_price = potential_trail
                        if price > pos.trailing_stop_price:
                            self._execute_full_close(session, pos, price, "Structural Trail Exit")
                    session.commit()

        except Exception as e:
            log.error(f"manage_open_positions error: {e}")
        finally:
            session.close()

    def _execute_full_close(self, session, pos, price, reason):
        log.info(f"Exiting full position {pos.symbol} via {reason}")
        m_symbol = self.adapter.get_market_symbol(pos.symbol)
        side = "sell" if pos.side == "long" else "buy"
        self.adapter.place_market_order(m_symbol, side, pos.amount)
        fill_px = price * (1 - (_SLIP_BPS + _FEE_BPS)) if side == "sell" else price * (1 + (_SLIP_BPS + _FEE_BPS))
        pnl = (fill_px - pos.avg_price) * pos.amount * (1 if pos.side == "long" else -1)
        session.add(Trade(symbol=pos.symbol, side=side, price=fill_px, amount=pos.amount,
                         pnl=pnl, status="closed", reason=reason))
        session.delete(pos)

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        pos = self.get_open_position(symbol)
        if not pos: return 0.0
        factor = 1 if pos["side"] == "long" else -1
        return (current_price - pos["avg_price"]) * pos["amount"] * factor

    def get_performance_metrics(self) -> Dict:
        session = get_session()
        try:
            trades = session.query(Trade).filter(Trade.pnl.isnot(None)).all()
            if not trades:
                return {"win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0, "total_pnl": 0.0}
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [abs(t.pnl) for t in trades if t.pnl < 0]
            total_won, total_lost = sum(wins), sum(losses)
            return {
                "win_rate": len(wins) / len(trades),
                "profit_factor": total_won / total_lost if total_lost > 0 else (2.0 if total_won > 0 else 0.0),
                "total_trades": len(trades),
                "total_pnl": total_won - total_lost,
            }
        finally: session.close()

    def get_todays_realized_pnl(self) -> float:
        session = get_session()
        try:
            today = datetime.utcnow().date()
            trades = session.query(Trade).filter(Trade.pnl.isnot(None)).all()
            return sum(t.pnl or 0.0 for t in trades if t.timestamp.date() == today)
        finally: session.close()

    def get_recent_trades(self, limit: int = 5) -> List[Dict]:
        session = get_session()
        try:
            trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
            return [{"symbol": t.symbol, "side": t.side, "price": t.price,
                     "pnl": t.pnl, "reason": t.reason, "timestamp": t.timestamp}
                    for t in trades]
        finally: session.close()

    def execute_trade(self, symbol: str, side: str, amount: float, price: float,
                      stop_loss: float, take_profit: float, reason: str = "Edge Signal") -> Dict:
        session = get_session()
        try:
            today_pnl = self.get_todays_realized_pnl()
            day_balance = self._get_day_balance_snapshot()
            if today_pnl <= -(day_balance * 0.05):
                return {"error": "Daily loss limit reached. Trading suspended."}

            m_symbol = self.adapter.get_market_symbol(symbol)
            pos = session.query(Position).filter_by(symbol=symbol).first()
            is_closing = pos and ((pos.side == "long" and side == "sell") or (pos.side == "short" and side == "buy"))

            result = self.adapter.place_order_with_sl_tp(m_symbol, side, amount, price, stop_loss, take_profit)
            fill_price = price * (1 + (_SLIP_BPS + _FEE_BPS)) if side == "buy" else price * (1 - (_SLIP_BPS + _FEE_BPS))

            realized_pnl = None
            if is_closing and pos:
                factor = 1 if pos.side == "long" else -1
                realized_pnl = (fill_price - pos.avg_price) * min(amount, pos.amount) * factor

            session.add(Trade(symbol=symbol, side=side, price=fill_price, amount=amount,
                             pnl=realized_pnl, status="closed" if is_closing else "open", reason=reason))

            if is_closing and pos:
                if amount >= pos.amount: session.delete(pos)
                else: pos.amount -= amount
            elif pos and not is_closing:
                total = pos.amount + amount
                pos.avg_price = (pos.avg_price * pos.amount + fill_price * amount) / total
                pos.amount = total
                pos.stop_loss, pos.take_profit = stop_loss, take_profit
            else:
                session.add(Position(symbol=symbol, avg_price=fill_price, amount=amount,
                                    side="long" if side == "buy" else "short",
                                    stop_loss=stop_loss, take_profit=take_profit, tp1_hit=False))

            session.commit()
            return result
        except Exception as e:
            session.rollback()
            log.error(f"execute_trade error: {e}")
            return {"error": str(e)}
        finally: session.close()
