from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from db import Position, SystemState, Trade, get_session

log = logging.getLogger(__name__)

_GRID_STATE_PREFIX = "grid_state"
_FEE_BPS = 0.001
_SLIP_BPS = 0.0005


@dataclass(frozen=True)
class GridConfig:
    enabled: bool = True
    levels: int = 8
    range_pct: float = 0.08
    total_quote_pct: float = 0.20
    min_order_quote: float = 10.0
    recenter_threshold_pct: float = 0.035
    refresh_seconds: int = 3
    trade_lookback: int = 500
    max_inventory_pct: float = 0.15


class GridManager:
    """
    Long-only grid manager.

    The grid places a ladder of buy limits below the anchor price. When a buy
    fills, the manager places a paired take-profit sell. Once that sell fills,
    the slot resets back to buy mode.
    """

    def __init__(self, tools, config: GridConfig):
        self.tools = tools
        self.config = config

    def _market_symbol(self, symbol: str) -> str:
        return self.tools.adapter.get_market_symbol(symbol)

    def _state_key(self, symbol: str) -> str:
        return f"{_GRID_STATE_PREFIX}_{self._market_symbol(symbol)}"

    def _load_state(self, symbol: str) -> dict:
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key=self._state_key(symbol)).first()
            if not row:
                return {}
            try:
                payload = json.loads(row.value)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
        finally:
            session.close()

    def _save_state(self, symbol: str, payload: dict):
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key=self._state_key(symbol)).first()
            value = json.dumps(payload, default=str)
            if row:
                row.value = value
            else:
                session.add(SystemState(key=self._state_key(symbol), value=value))
            session.commit()
        except Exception as exc:
            session.rollback()
            log.warning("Could not persist grid state for %s: %s", symbol, exc)
        finally:
            session.close()

    def _cleanup_open_orders(self, symbol: str):
        try:
            self.tools.adapter.cancel_all_open_orders(symbol)
        except Exception as exc:
            log.warning("Could not cancel existing grid orders for %s: %s", symbol, exc)

    def _normalize_price(self, symbol: str, price: float) -> float:
        try:
            return float(self.tools.adapter.price_to_precision(symbol, price))
        except Exception:
            return float(price)

    def _normalize_amount(self, symbol: str, amount: float) -> float:
        try:
            return float(self.tools.adapter.amount_to_precision(symbol, amount))
        except Exception:
            return float(amount)

    def _available_quote(self, balance: float) -> float:
        budget = balance * min(max(self.config.total_quote_pct, 0.0), max(self.config.max_inventory_pct, 0.0))
        return max(budget, self.config.min_order_quote * max(self.config.levels, 1))

    def _grid_step_pct(self) -> float:
        return max(self.config.range_pct / max(self.config.levels, 1), 0.001)

    def _current_position(self, symbol: str) -> Optional[dict]:
        return self.tools.get_open_position(symbol)

    def _update_position_on_buy(self, symbol: str, qty: float, price: float, reason: str):
        session = get_session()
        try:
            pos = session.query(Position).filter_by(symbol=symbol).first()
            if pos:
                total = pos.amount + qty
                if total > 0:
                    pos.avg_price = (pos.avg_price * pos.amount + price * qty) / total
                pos.amount = total
                pos.side = "long"
            else:
                session.add(
                    Position(
                        symbol=symbol,
                        avg_price=price,
                        amount=qty,
                        side="long",
                        stop_loss=0.0,
                        take_profit=0.0,
                        tp1_hit=False,
                    )
                )
            session.add(
                Trade(
                    symbol=symbol,
                    side="buy",
                    price=price * (1 + (_FEE_BPS + _SLIP_BPS)),
                    amount=qty,
                    pnl=None,
                    status="open",
                    reason=reason,
                )
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            log.warning("Could not update buy fill for %s: %s", symbol, exc)
        finally:
            session.close()

    def _update_position_on_sell(self, symbol: str, qty: float, entry_price: float, exit_price: float, reason: str):
        session = get_session()
        try:
            pos = session.query(Position).filter_by(symbol=symbol).first()
            if not pos:
                return

            closing_qty = min(qty, pos.amount)
            if closing_qty <= 0:
                return

            factor = 1 if pos.side == "long" else -1
            fill_px = exit_price * (1 - (_FEE_BPS + _SLIP_BPS))
            pnl = (fill_px - entry_price) * closing_qty * factor
            session.add(
                Trade(
                    symbol=symbol,
                    side="sell",
                    price=fill_px,
                    amount=closing_qty,
                    pnl=pnl,
                    status="closed",
                    reason=reason,
                )
            )

            if closing_qty >= pos.amount:
                session.delete(pos)
            else:
                pos.amount = max(pos.amount - closing_qty, 0.0)
            session.commit()
        except Exception as exc:
            session.rollback()
            log.warning("Could not update sell fill for %s: %s", symbol, exc)
        finally:
            session.close()

    def _build_levels(self, symbol: str, anchor_price: float, per_order_quote: float) -> List[dict]:
        step_pct = self._grid_step_pct()
        levels: List[dict] = []
        for idx in range(1, self.config.levels + 1):
            buy_price = self._normalize_price(symbol, anchor_price * (1 - step_pct * idx))
            if buy_price <= 0:
                continue
            qty = self._normalize_amount(symbol, per_order_quote / buy_price)
            if qty <= 0:
                continue
            sell_price = self._normalize_price(symbol, buy_price * (1 + step_pct))
            levels.append(
                {
                    "index": idx,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "order_qty": qty,
                    "buy_order_id": None,
                    "sell_order_id": None,
                    "buy_filled_qty": 0.0,
                    "sell_filled_qty": 0.0,
                    "entry_price": None,
                    "status": "buy_open",
                    "updated_at": time.time(),
                }
            )
        return levels

    def _place_buy_order(self, symbol: str, level: dict) -> Dict[str, Any]:
        order = self.tools.adapter.place_limit_order(
            symbol,
            "buy",
            level["order_qty"],
            level["buy_price"],
            reduce_only=False,
        )
        if not order.get("error"):
            order_id = int(order.get("orderId", 0) or 0)
            level["buy_order_id"] = order_id or None
            level["status"] = "buy_open"
            level["updated_at"] = time.time()
        return order

    def _place_sell_order(self, symbol: str, level: dict, quantity: float) -> Dict[str, Any]:
        order = self.tools.adapter.place_limit_order(
            symbol,
            "sell",
            quantity,
            level["sell_price"],
            reduce_only=self.tools.adapter.trading_mode == "futures",
        )
        if not order.get("error"):
            order_id = int(order.get("orderId", 0) or 0)
            level["sell_order_id"] = order_id or None
            level["status"] = "sell_open"
            level["updated_at"] = time.time()
        return order

    def _reset_grid(self, symbol: str, anchor_price: float, balance: float) -> dict:
        self._cleanup_open_orders(symbol)
        per_order_quote = self._available_quote(balance) / max(self.config.levels, 1)
        levels = self._build_levels(symbol, anchor_price, per_order_quote)
        if not levels:
            raise RuntimeError("Unable to build grid levels; increase balance or lower grid size")
        for level in levels:
            order = self._place_buy_order(symbol, level)
            if order.get("error"):
                self._cleanup_open_orders(symbol)
                raise RuntimeError(f"Failed to place grid buy order at {level['buy_price']}: {order['error']}")

        state = {
            "symbol": symbol,
            "market_symbol": self._market_symbol(symbol),
            "active": True,
            "anchor_price": anchor_price,
            "lower_bound": levels[-1]["buy_price"] if levels else anchor_price,
            "upper_bound": anchor_price,
            "levels": asdict(self.config),
            "grid_step_pct": self._grid_step_pct(),
            "quote_per_order": per_order_quote,
            "order_count": len(levels),
            "last_trade_id": 0,
            "levels_state": levels,
            "updated_at": time.time(),
        }
        self._save_state(symbol, state)
        return state

    def _find_level_by_order_id(self, state: dict, order_id: int, order_kind: str) -> Optional[dict]:
        for level in state.get("levels_state", []):
            if order_kind == "buy" and int(level.get("buy_order_id") or 0) == order_id:
                return level
            if order_kind == "sell" and int(level.get("sell_order_id") or 0) == order_id:
                return level
        return None

    def _apply_buy_fill(self, symbol: str, state: dict, level: dict, qty: float, price: float, order_id: int):
        level["buy_filled_qty"] = min(float(level.get("buy_filled_qty", 0.0)) + qty, float(level["order_qty"]))
        level["entry_price"] = price if level.get("entry_price") is None else level["entry_price"]
        self._update_position_on_buy(symbol, qty, price, "grid_buy_fill")

        if level["buy_filled_qty"] + 1e-12 >= float(level["order_qty"]):
            level["buy_order_id"] = None
            level["buy_filled_qty"] = float(level["order_qty"])
            sell_qty = self._normalize_amount(symbol, float(level["order_qty"]))
            sell_order = self._place_sell_order(symbol, level, sell_qty)
            if sell_order.get("error"):
                raise RuntimeError(f"Failed to place paired sell order: {sell_order['error']}")
        else:
            # Keep the paired exit aligned to the currently filled inventory.
            sell_qty = self._normalize_amount(symbol, float(level["buy_filled_qty"]))
            if sell_qty > 0:
                if level.get("sell_order_id"):
                    try:
                        self.tools.adapter.cancel_order(symbol, order_id=int(level["sell_order_id"]))
                    except Exception:
                        pass
                sell_order = self._place_sell_order(symbol, level, sell_qty)
                if sell_order.get("error"):
                    raise RuntimeError(f"Failed to place partial paired sell order: {sell_order['error']}")

        state["updated_at"] = time.time()

    def _apply_sell_fill(self, symbol: str, state: dict, level: dict, qty: float, price: float, order_id: int):
        entry_price = float(level.get("entry_price") or level.get("buy_price") or price)
        level["sell_filled_qty"] = min(float(level.get("sell_filled_qty", 0.0)) + qty, float(level["order_qty"]))
        self._update_position_on_sell(symbol, qty, entry_price, price, "grid_sell_fill")

        if level["sell_filled_qty"] + 1e-12 >= float(level["order_qty"]):
            level["sell_order_id"] = None
            level["sell_filled_qty"] = 0.0
            level["buy_filled_qty"] = 0.0
            level["entry_price"] = None
            buy_order = self._place_buy_order(symbol, level)
            if buy_order.get("error"):
                raise RuntimeError(f"Failed to recycle buy order: {buy_order['error']}")
        else:
            # Re-issue the remaining exit quantity so inventory stays protected.
            if level.get("sell_order_id"):
                try:
                    self.tools.adapter.cancel_order(symbol, order_id=int(level["sell_order_id"]))
                except Exception:
                    pass
            remaining_qty = self._normalize_amount(symbol, float(level["order_qty"]) - float(level["sell_filled_qty"]))
            if remaining_qty > 0:
                sell_order = self._place_sell_order(symbol, level, remaining_qty)
                if sell_order.get("error"):
                    raise RuntimeError(f"Failed to reissue sell order: {sell_order['error']}")

        state["updated_at"] = time.time()

    def _simulate_paper_fills(self, symbol: str, state: dict, price: float) -> List[dict]:
        fills: List[dict] = []
        for level in state.get("levels_state", []):
            if level.get("status") == "buy_open" and price <= float(level["buy_price"]):
                qty = float(level["order_qty"])
                fills.append(
                    {
                        "orderId": int(level.get("buy_order_id") or 0),
                        "side": "BUY",
                        "qty": qty,
                        "price": float(level["buy_price"]),
                    }
                )
            elif level.get("status") == "sell_open" and price >= float(level["sell_price"]):
                qty = float(level["order_qty"])
                fills.append(
                    {
                        "orderId": int(level.get("sell_order_id") or 0),
                        "side": "SELL",
                        "qty": qty,
                        "price": float(level["sell_price"]),
                    }
                )
        return fills

    def _reconcile_live_fills(self, symbol: str, state: dict) -> List[dict]:
        last_trade_id = int(state.get("last_trade_id", 0) or 0)
        try:
            trades = self.tools.adapter.get_my_trades(
                symbol,
                from_id=(last_trade_id + 1 if last_trade_id > 0 else None),
                limit=self.config.trade_lookback,
            )
        except Exception as exc:
            log.warning("Grid trade reconciliation failed for %s: %s", symbol, exc)
            return []

        fills: List[dict] = []
        for trade in trades or []:
            trade_id = int(trade.get("id", 0) or 0)
            if trade_id > 0:
                state["last_trade_id"] = max(int(state.get("last_trade_id", 0) or 0), trade_id)
            fills.append(
                {
                    "orderId": int(trade.get("orderId", 0) or 0),
                    "side": str(trade.get("side", "")).upper(),
                    "qty": float(trade.get("qty", 0.0) or 0.0),
                    "price": float(trade.get("price", 0.0) or 0.0),
                }
            )
        return fills

    def _refresh_missing_orders(self, symbol: str, state: dict):
        try:
            open_orders = self.tools.adapter.get_open_orders(symbol)
        except Exception as exc:
            log.warning("Could not inspect open grid orders for %s: %s", symbol, exc)
            return

        open_ids = {int(order.get("orderId", 0) or 0) for order in open_orders or []}
        for level in state.get("levels_state", []):
            if level.get("status") == "buy_open":
                order_id = int(level.get("buy_order_id") or 0)
                if order_id and order_id not in open_ids:
                    try:
                        self._place_buy_order(symbol, level)
                    except Exception as exc:
                        log.warning("Could not restore buy grid order for %s: %s", symbol, exc)
            elif level.get("status") == "sell_open":
                order_id = int(level.get("sell_order_id") or 0)
                if order_id and order_id not in open_ids:
                    remaining_qty = float(level.get("order_qty", 0.0)) - float(level.get("sell_filled_qty", 0.0))
                    if remaining_qty > 0:
                        try:
                            self._place_sell_order(symbol, level, remaining_qty)
                        except Exception as exc:
                            log.warning("Could not restore sell grid order for %s: %s", symbol, exc)

    def _maybe_recenter(self, symbol: str, state: dict, price: float, balance: float):
        if not state.get("levels_state"):
            return
        lower_bound = float(state.get("lower_bound", 0.0) or 0.0)
        upper_bound = float(state.get("upper_bound", 0.0) or 0.0)
        position = self._current_position(symbol)
        inventory = float(position.get("amount", 0.0) or 0.0) if position else 0.0

        if inventory > 0:
            return
        if lower_bound <= 0 or upper_bound <= 0:
            return

        if price < lower_bound * (1 - self.config.recenter_threshold_pct) or price > upper_bound * (1 + self.config.recenter_threshold_pct):
            log.info("Recentering grid for %s around %.4f", symbol, price)
            self._reset_grid(symbol, price, balance)

    def sync(self, symbol: str, price: float, balance: float, allow_new_orders: bool = True) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"symbol": symbol, "enabled": False, "reason": "grid_disabled"}

        if self.tools.adapter.trading_mode == "futures" and getattr(self.tools.adapter, "position_mode", "ONE_WAY") == "HEDGE":
            return {
                "symbol": symbol,
                "enabled": False,
                "reason": "grid_mode_requires_spot_or_futures_one_way",
            }

        state = self._load_state(symbol)
        if not state or not state.get("levels_state"):
            if not allow_new_orders:
                return {"symbol": symbol, "enabled": False, "reason": "grid_paused"}
            state = self._reset_grid(symbol, price, balance)

        if not allow_new_orders:
            return {
                "symbol": symbol,
                "enabled": True,
                "active": False,
                "reason": "grid_paused",
                "anchor_price": state.get("anchor_price"),
                "levels": len(state.get("levels_state", [])),
            }

        fills = self._simulate_paper_fills(symbol, state, price) if self.tools.adapter.paper_trading else self._reconcile_live_fills(symbol, state)
        applied = 0
        for fill in fills:
            order_id = int(fill.get("orderId", 0) or 0)
            side = str(fill.get("side", "")).upper()
            qty = float(fill.get("qty", 0.0) or 0.0)
            fill_price = float(fill.get("price", 0.0) or price)

            level = self._find_level_by_order_id(state, order_id, "buy" if side == "BUY" else "sell")
            if not level or qty <= 0:
                continue

            try:
                if side == "BUY":
                    self._apply_buy_fill(symbol, state, level, qty, fill_price, order_id)
                else:
                    self._apply_sell_fill(symbol, state, level, qty, fill_price, order_id)
                applied += 1
            except Exception as exc:
                log.warning("Grid fill application failed for %s: %s", symbol, exc)

        if not self.tools.adapter.paper_trading:
            self._refresh_missing_orders(symbol, state)

        self._maybe_recenter(symbol, state, price, balance)

        state["updated_at"] = time.time()
        self._save_state(symbol, state)

        position = self._current_position(symbol)
        return {
            "symbol": symbol,
            "enabled": True,
            "active": True,
            "anchor_price": state.get("anchor_price"),
            "price": price,
            "levels": len(state.get("levels_state", [])),
            "fills_applied": applied,
            "open_position": position,
            "grid_step_pct": state.get("grid_step_pct"),
            "quote_per_order": state.get("quote_per_order"),
            "last_trade_id": state.get("last_trade_id", 0),
            "state": state,
        }

    def get_state(self, symbol: str) -> dict:
        return self._load_state(symbol)
