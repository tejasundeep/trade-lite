import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json
import time

from trading.adapters.crypto import CCXTCryptoAdapter
from indicators.market_context import set_market_data
from db import get_session, Trade, Position, SystemState

log = logging.getLogger(__name__)

_FEE_BPS   = 0.001
_SLIP_BPS  = 0.0005
_EXECUTION_STATE_KEY = "execution_reconciliation_state"


class TradingTools:
    def __init__(
        self,
        api_key: str = None,
        secret: str = None,
        paper_trading: bool = True,
        exchange_id: str = "binance",
        trading_mode: str = "futures",
        leverage: int = 3,
        margin_type: str = "ISOLATED",
        position_mode: str = "ONE_WAY",
    ):
        self.adapter = CCXTCryptoAdapter(
            exchange_id=exchange_id,
            api_key=api_key,
            secret=secret,
            paper_trading=paper_trading,
            trading_mode=trading_mode,
            leverage=leverage,
            margin_type=margin_type,
            position_mode=position_mode,
        )
        self._day_balance: Optional[float] = None
        self._snap_date:   Optional[date]  = None

    def get_day_balance_snapshot(self) -> float:
        return self._get_day_balance_snapshot()

    def build_institutional_levels(self, df_d: Optional[pd.DataFrame] = None, df_w: Optional[pd.DataFrame] = None) -> Dict:
        levels = {}
        try:
            if df_d is not None and not df_d.empty:
                levels.update({
                    "pdh": float(df_d.iloc[:-1]["high"].max()) if len(df_d) > 1 else float(df_d.iloc[-1]["high"]),
                    "pdl": float(df_d.iloc[:-1]["low"].min()) if len(df_d) > 1 else float(df_d.iloc[-1]["low"]),
                    "pdo": float(df_d.iloc[0]["open"]),
                    "pdc": float(df_d.iloc[-1]["close"]),
                })
            if df_w is not None and not df_w.empty:
                levels.update({
                    "weekly_open": float(df_w.iloc[-1]["open"]),
                    "pwh": float(df_w.iloc[:-1]["high"].max()) if len(df_w) > 1 else float(df_w.iloc[-1]["high"]),
                    "pwl": float(df_w.iloc[:-1]["low"].min()) if len(df_w) > 1 else float(df_w.iloc[-1]["low"]),
                })
        except Exception as e:
            log.debug("build_institutional_levels fallback: %s", e)
        return levels

    def _load_json_state(self, key: str, default: Optional[dict] = None) -> dict:
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key=key).first()
            if not row:
                return dict(default or {})
            try:
                payload = json.loads(row.value)
                return payload if isinstance(payload, dict) else dict(default or {})
            except Exception:
                return dict(default or {})
        finally:
            session.close()

    def _save_json_state(self, key: str, payload: dict):
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key=key).first()
            value = json.dumps(payload, default=str)
            if row:
                row.value = value
            else:
                session.add(SystemState(key=key, value=value))
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("Could not persist %s: %s", key, e)
        finally:
            session.close()

    def _current_exit_order_prices(self, symbol: str) -> tuple[float, float]:
        stop_loss = 0.0
        take_profit = 0.0
        try:
            open_orders = self.adapter.get_open_orders(symbol)
            for order in open_orders:
                order_type = str(order.get("type", "")).upper()
                trigger = float(order.get("stopPrice", order.get("price", 0.0)) or 0.0)
                if order_type in {"STOP_MARKET", "STOP_LOSS_MARKET", "STOP_LOSS_LIMIT"}:
                    stop_loss = trigger
                elif order_type in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT_LIMIT"}:
                    take_profit = trigger
        except Exception as e:
            log.debug("current_exit_order_prices error for %s: %s", symbol, e)
        return stop_loss, take_profit

    def _sync_local_position_snapshot(self, session, symbol: str, remote: Optional[Dict[str, Any]], stop_loss: float = 0.0, take_profit: float = 0.0) -> str:
        local = session.query(Position).filter_by(symbol=symbol).first()
        if remote and remote.get("amount", 0.0) > 0:
            side = remote.get("side", "long")
            amount = float(remote.get("amount", 0.0))
            avg_price = float(remote.get("entry_price", 0.0))

            if local:
                local.avg_price = avg_price or local.avg_price
                local.amount = amount
                local.side = side
                if stop_loss > 0:
                    local.stop_loss = stop_loss
                if take_profit > 0:
                    local.take_profit = take_profit
                return "updated"

            session.add(
                Position(
                    symbol=symbol,
                    avg_price=avg_price,
                    amount=amount,
                    side=side,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    tp1_hit=False,
                )
            )
            return "created"

        if local:
            session.delete(local)
            return "deleted"

        return "unchanged"

    def _mark_symbol_trade_cursor(self, symbol: str):
        if self.adapter.paper_trading or not (self.adapter.api_key and self.adapter.secret):
            return
        try:
            trades = self.adapter.get_my_trades(symbol, limit=10)
            if not trades:
                return
            latest_id = max(int(t.get("id", 0) or 0) for t in trades)
            if latest_id <= 0:
                return
            state = self._load_json_state(_EXECUTION_STATE_KEY, {})
            market_symbol = self.adapter.get_market_symbol(symbol)
            symbol_state = state.get(market_symbol, {})
            symbol_state["last_trade_id"] = max(int(symbol_state.get("last_trade_id", 0) or 0), latest_id)
            symbol_state["updated_at"] = time.time()
            state[market_symbol] = symbol_state
            self._save_json_state(_EXECUTION_STATE_KEY, state)
        except Exception as e:
            log.debug("mark_symbol_trade_cursor failed for %s: %s", symbol, e)

    def reconcile_execution_state(self, symbols: Optional[List[str]] = None, trade_lookback: int = 500) -> Dict[str, Any]:
        """
        Reconcile exchange truth against local DB state.

        The reconciler prioritizes remote position snapshots, but also advances
        a per-symbol trade cursor so we do not double count our own fills.
        """
        if self.adapter.paper_trading:
            return {"mode": "paper", "reconciled": False, "reason": "paper_trading"}

        session = get_session()
        summary = {"mode": self.adapter.trading_mode, "reconciled": True, "symbols": [], "cursor_updated": False}
        try:
            state = self._load_json_state(_EXECUTION_STATE_KEY, {})
            target_symbols = list(dict.fromkeys(symbols or []))
            if not target_symbols:
                target_symbols = [p["symbol"] for p in self.adapter.get_open_positions()]

            remote_positions = {p["market_symbol"]: p for p in self.adapter.get_open_positions()}
            if target_symbols:
                allowed = {self.adapter.get_market_symbol(s) for s in target_symbols}
                remote_positions = {ms: pos for ms, pos in remote_positions.items() if ms in allowed}

            for symbol in target_symbols or [self.adapter.get_market_symbol(ms) for ms in remote_positions.keys()]:
                market_symbol = self.adapter.get_market_symbol(symbol)
                cursor_state = state.get(market_symbol, {})
                last_trade_id = int(cursor_state.get("last_trade_id", 0) or 0)

                new_fills: List[Dict[str, Any]] = []
                try:
                    new_fills = self.adapter.get_my_trades(symbol, from_id=(last_trade_id + 1 if last_trade_id > 0 else None), limit=trade_lookback)
                except Exception as e:
                    log.warning("Could not fetch trades for %s: %s", symbol, e)

                if new_fills:
                    newest = max(int(t.get("id", 0) or 0) for t in new_fills)
                    cursor_state["last_trade_id"] = max(last_trade_id, newest)
                    cursor_state["updated_at"] = time.time()
                    state[market_symbol] = cursor_state
                    summary["cursor_updated"] = True

                stop_loss, take_profit = self._current_exit_order_prices(symbol)
                remote = remote_positions.get(market_symbol)
                action = self._sync_local_position_snapshot(session, symbol, remote, stop_loss=stop_loss, take_profit=take_profit)
                summary["symbols"].append(
                    {
                        "symbol": symbol,
                        "market_symbol": market_symbol,
                        "fills_seen": len(new_fills),
                        "position_action": action,
                        "remote_position": bool(remote and remote.get("amount", 0.0) > 0),
                        "last_trade_id": cursor_state.get("last_trade_id", 0),
                    }
                )

            if summary["cursor_updated"]:
                self._save_json_state(_EXECUTION_STATE_KEY, state)
            session.commit()
            return summary
        except Exception as e:
            session.rollback()
            log.warning("reconcile_execution_state error: %s", e)
            summary["error"] = str(e)
            summary["reconciled"] = False
            return summary
        finally:
            session.close()

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        m_symbol = self.adapter.get_market_symbol(symbol)
        df = self.adapter.get_market_data(m_symbol, timeframe, limit)
        if df is None or df.empty:
            log.warning("No market data returned for %s @ %s", symbol, timeframe)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        expected = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected:
            if col not in df.columns:
                log.warning("Market data for %s missing column %s", symbol, col)
                return pd.DataFrame(columns=expected)

        df["EMA_50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
        set_market_data(df)
        return df

    def get_institutional_levels(self, symbol: str) -> Dict:
        try:
            m_symbol = self.adapter.get_market_symbol(symbol)
            df_d = self.adapter.get_market_data(m_symbol, "1d", 5)
            df_w = self.adapter.get_market_data(m_symbol, "1w", 3)
            return self.build_institutional_levels(df_d, df_w)
        except Exception as e:
            log.debug("get_institutional_levels fallback for %s: %s", symbol, e)
        return {}

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

    def sync_positions_from_exchange(self, symbols: Optional[List[str]] = None):
        """
        Backward-compatible wrapper for the execution reconciler.
        """
        return self.reconcile_execution_state(symbols=symbols)

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
                        result = self.adapter.place_market_order(m_symbol, side, close_amt, reduce_only=True)
                        if result.get("error"):
                            log.warning("TP1 partial close failed for %s: %s", pos.symbol, result.get("error"))
                            continue
                        closed_qty = float(result.get("executedQty", close_amt) or close_amt)
                        if closed_qty <= 0:
                            log.warning("TP1 partial close returned zero fill for %s", pos.symbol)
                            continue
                        
                        # BE+ (entry + 15bps to cover slippage/fees)
                        be_plus = pos.avg_price * 1.0015 if pos.side == "long" else pos.avg_price * 0.9985
                        pos.amount = max(pos.amount - closed_qty, 0.0)
                        pos.stop_loss = be_plus
                        pos.tp1_hit = True
                        pos.trailing_stop_price = be_plus
                        session.commit()
                        self._refresh_exchange_exits(pos)
                        continue

                # 2. Structural Trailing Stop
                if pos.tp1_hit:
                    # Combined trail: Max(Structural Pivot, ATR Trail)
                    smc = smc_map.get(pos.symbol, {})
                    pd_arr = smc.get("pd_array", {})
                    
                    if pos.side == "long":
                        atr_trail = price - atr * 2.0
                        struct_trail = pd_arr.get("low", 0) # Support pivot
                        current_trail = pos.trailing_stop_price or 0
                        potential_trail = max(atr_trail, struct_trail, current_trail)
                        trail_changed = potential_trail > current_trail
                        if trail_changed:
                            pos.trailing_stop_price = potential_trail
                        if pos.trailing_stop_price and price < pos.trailing_stop_price:
                            self._execute_full_close(session, pos, price, "Structural Trail Exit")
                        elif trail_changed:
                            session.commit()
                            self._refresh_exchange_exits(pos)
                    else:
                        atr_trail = price + atr * 2.0
                        struct_trail = pd_arr.get("high", 999999999) # Resistance pivot
                        current_trail = pos.trailing_stop_price or 999999999
                        potential_trail = min(atr_trail, struct_trail, current_trail)
                        trail_changed = potential_trail < current_trail
                        if trail_changed:
                            pos.trailing_stop_price = potential_trail
                        if pos.trailing_stop_price and price > pos.trailing_stop_price:
                            self._execute_full_close(session, pos, price, "Structural Trail Exit")
                        elif trail_changed:
                            session.commit()
                            self._refresh_exchange_exits(pos)
                    session.commit()

        except Exception as e:
            log.error(f"manage_open_positions error: {e}")
        finally:
            session.close()

    def _execute_full_close(self, session, pos, price, reason):
        log.info(f"Exiting full position {pos.symbol} via {reason}")
        m_symbol = self.adapter.get_market_symbol(pos.symbol)
        side = "sell" if pos.side == "long" else "buy"
        original_amount = pos.amount
        result = self.adapter.place_market_order(m_symbol, side, pos.amount, reduce_only=True)
        if result.get("error"):
            log.warning("Full close failed for %s: %s", pos.symbol, result.get("error"))
            return False
        closed_qty = float(result.get("executedQty", original_amount) or original_amount)
        if closed_qty <= 0:
            log.warning("Full close returned zero fill for %s", pos.symbol)
            return False
        fill_px = price * (1 - (_SLIP_BPS + _FEE_BPS)) if side == "sell" else price * (1 + (_SLIP_BPS + _FEE_BPS))
        pnl = (fill_px - pos.avg_price) * closed_qty * (1 if pos.side == "long" else -1)
        session.add(Trade(symbol=pos.symbol, side=side, price=fill_px, amount=closed_qty,
                         pnl=pnl, status="closed", reason=reason))
        if closed_qty >= original_amount * 0.999:
            session.delete(pos)
        else:
            pos.amount = max(pos.amount - closed_qty, 0.0)
        session.commit()
        if closed_qty >= original_amount * 0.999:
            self._cancel_exchange_orders(pos.symbol)
        else:
            self._refresh_exchange_exits(pos)
        return True

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
            if result.get("error"):
                session.rollback()
                return result
            if result.get("exit_error"):
                log.warning("Protective exit placement failed for %s: %s", symbol, result.get("exit_error"))

            executed_qty = float(
                result.get("entry", {}).get("executedQty", result.get("entry", {}).get("origQty", amount)) or amount
            )
            if executed_qty <= 0:
                session.rollback()
                return {"error": "Order did not fill"}

            fill_price = price
            entry_payload = result.get("entry", {})
            try:
                entry_price = float(entry_payload.get("avgPrice") or entry_payload.get("price") or 0.0)
                if entry_price > 0:
                    fill_price = entry_price
            except Exception as e:
                log.debug("Could not parse entry fill price for %s: %s", symbol, e)

            if self.adapter.trading_mode == "futures" and not self.adapter.paper_trading and pos is None:
                try:
                    self.adapter.set_margin_type(symbol)
                    self.adapter.set_leverage(symbol)
                except Exception as e:
                    log.warning("futures symbol config skipped for %s: %s", symbol, e)

            if side == "buy":
                fill_price = fill_price * (1 + (_SLIP_BPS + _FEE_BPS))
            else:
                fill_price = fill_price * (1 - (_SLIP_BPS + _FEE_BPS))

            realized_pnl = None
            if is_closing and pos:
                factor = 1 if pos.side == "long" else -1
                realized_pnl = (fill_price - pos.avg_price) * min(executed_qty, pos.amount) * factor

            session.add(Trade(symbol=symbol, side=side, price=fill_price, amount=executed_qty,
                             pnl=realized_pnl, status="closed" if is_closing else "open", reason=reason))

            if is_closing and pos:
                if executed_qty >= pos.amount:
                    residual = max(executed_qty - pos.amount, 0.0)
                    session.delete(pos)
                    self._cancel_exchange_orders(symbol)
                    if residual > 0:
                        residual_pos = Position(
                            symbol=symbol,
                            avg_price=fill_price,
                            amount=residual,
                            side="long" if side == "buy" else "short",
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            tp1_hit=False,
                        )
                        session.add(residual_pos)
                        if self.adapter.trading_mode == "futures" and not self.adapter.paper_trading:
                            self._refresh_exchange_exits(residual_pos)
                else:
                    pos.amount -= executed_qty
                    self._refresh_exchange_exits(pos)
            elif pos and not is_closing:
                total = pos.amount + executed_qty
                pos.avg_price = (pos.avg_price * pos.amount + fill_price * executed_qty) / total
                pos.amount = total
                pos.stop_loss, pos.take_profit = stop_loss, take_profit
                self._refresh_exchange_exits(pos)
            else:
                new_pos = Position(symbol=symbol, avg_price=fill_price, amount=executed_qty,
                                   side="long" if side == "buy" else "short",
                                   stop_loss=stop_loss, take_profit=take_profit, tp1_hit=False)
                session.add(new_pos)
                if self.adapter.trading_mode == "futures" and not self.adapter.paper_trading:
                    self._refresh_exchange_exits(new_pos)

            session.commit()
            self._mark_symbol_trade_cursor(symbol)
            return result
        except Exception as e:
            session.rollback()
            log.error(f"execute_trade error: {e}")
            return {"error": str(e)}
        finally: session.close()

    def _cancel_exchange_orders(self, symbol: str):
        if self.adapter.paper_trading:
            return
        try:
            self.adapter.cancel_all_open_orders(symbol)
        except Exception as e:
            log.warning("cancel_all_open_orders failed for %s: %s", symbol, e)

    def _refresh_exchange_exits(self, pos):
        if self.adapter.paper_trading:
            return
        if self.adapter.trading_mode != "futures":
            return
        if pos.amount <= 0 or pos.stop_loss <= 0:
            return
        try:
            self.adapter.cancel_all_open_orders(pos.symbol)
            if pos.take_profit and pos.take_profit > 0:
                self.adapter.place_exit_orders(pos.symbol, pos.side, pos.amount, pos.stop_loss, pos.take_profit)
        except Exception as e:
            log.warning("refresh_exchange_exits failed for %s: %s", pos.symbol, e)
