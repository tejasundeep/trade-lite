from typing import Dict, Optional
import time
import json
from db import get_session, SystemState

_COOLDOWN_KEY = "risk_cooldown"
_DEFAULT_STATE = {"last_loss_time": 0, "consecutive_losses": 0, "lock_until": 0}

def _get_state() -> dict:
    session = get_session()
    try:
        row = session.query(SystemState).filter_by(key=_COOLDOWN_KEY).first()
        if not row:
            initial = dict(_DEFAULT_STATE)
            session.add(SystemState(key=_COOLDOWN_KEY, value=json.dumps(initial)))
            session.commit()
            return initial
        try:
            loaded = json.loads(row.value)
        except Exception:
            loaded = {}
        if not isinstance(loaded, dict):
            loaded = {}
        state = dict(_DEFAULT_STATE)
        state.update({k: loaded.get(k, state[k]) for k in state})
        return state
    finally:
        session.close()

def _save_state(data: dict):
    session = get_session()
    try:
        row = session.query(SystemState).filter_by(key=_COOLDOWN_KEY).first()
        if row:
            row.value = json.dumps(data)
        else:
            session.add(SystemState(key=_COOLDOWN_KEY, value=json.dumps(data)))
        session.commit()
    finally:
        session.close()

def calculate_risk_parameters(
    free_balance: float,
    current_price: float,
    confidence_score: float,
    atr_pct: float = 1.0,
    stop_loss: Optional[float] = None,
    expected_r: float = 2.5,
    historical_win_rate: float = 0.55,
    total_trades: int = 0,
    is_last_trade_loss: bool = False,
    spread_bps: float = 5.0, # Spread in basis points
    max_risk_per_trade_pct: float = 0.015,
    max_position_pct: float = 0.15,
    min_order_quote: float = 10.0,
) -> Dict:
    state = _get_state()
    now   = time.time()
    lock_until = float(state.get("lock_until", 0) or 0)

    stop_loss_value = None
    if stop_loss is not None:
        try:
            stop_loss_value = float(stop_loss)
        except (TypeError, ValueError):
            return {
                "action": "lock",
                "reason": "Invalid stop loss value",
                "recommended_amount": 0.0,
                "risk_pct": 0.0,
                "notional_value": 0.0,
                "sl_distance_pct": 0,
                "liquidity_penalty": False,
            }
        if stop_loss_value <= 0 or abs(current_price - stop_loss_value) <= 1e-12:
            return {
                "action": "lock",
                "reason": "Invalid stop loss value",
                "recommended_amount": 0.0,
                "risk_pct": 0.0,
                "notional_value": 0.0,
                "sl_distance_pct": 0,
                "liquidity_penalty": False,
            }

    if free_balance <= 0 or current_price <= 0:
        return {
            "action": "lock",
            "reason": "Invalid balance or price",
            "recommended_amount": 0.0,
            "risk_pct": 0.0,
            "notional_value": 0.0,
            "sl_distance_pct": 0,
            "liquidity_penalty": False,
        }

    if is_last_trade_loss:
        state["consecutive_losses"] += 1
        state["last_loss_time"]      = now
        state["lock_until"] = now + (4 ** state["consecutive_losses"]) * 3600
        _save_state(state)
    elif state["consecutive_losses"] > 0:
        state["consecutive_losses"] = 0
        _save_state(state)

    if now < lock_until:
        return {"action": "lock", "reason": "System in Cooldown",
                "unlock_in_seconds": round(lock_until - now)}

    # 1. Base Kelly / Flat %
    if total_trades < 30:
        final_risk_pct = min(0.005, max_risk_per_trade_pct)
    else:
        full_kelly     = (historical_win_rate * expected_r - (1 - historical_win_rate)) / expected_r
        safe_kelly     = max(0.0, full_kelly * 0.20)
        final_risk_pct = min(safe_kelly * confidence_score, max_risk_per_trade_pct)

    # 2. Elite Liquidity Adjustment
    # If spread > 10% of the ATR, we are in a low-liquidity environment. Reduce risk.
    liquidity_penalty = 1.0
    atr_bps = (atr_pct / 100) * 10000
    if spread_bps > (atr_bps * 0.1):
        liquidity_penalty = 0.5 # Halve risk if slippage is likely high
        
    # 3. Extreme Volatility Guard
    # If ATR is > 5%, the market is "wild". Cap risk at 0.5%.
    if atr_pct > 5.0:
        final_risk_pct = min(final_risk_pct, 0.005)
        
    final_risk_pct *= liquidity_penalty

    if stop_loss_value and stop_loss_value != current_price:
        risk_amount  = free_balance * final_risk_pct
        sl_dist      = abs(current_price - stop_loss_value)
        asset_amount = risk_amount / sl_dist if sl_dist > 0 else 0.0
        notional     = asset_amount * current_price
        max_notional  = free_balance * max_position_pct
        if notional > max_notional:
            asset_amount = max_notional / current_price
            notional     = max_notional
    else:
        asset_amount = (free_balance * final_risk_pct) / current_price
        notional     = asset_amount * current_price

    if notional < min_order_quote:
        return {
            "action": "lock",
            "reason": f"Below minimum order quote: {notional:.2f} < {min_order_quote:.2f}",
            "recommended_amount": round(asset_amount, 6),
            "risk_pct": round(final_risk_pct * 100, 2),
            "notional_value": round(notional, 2),
            "sl_distance_pct": round(abs(current_price - (stop_loss_value or 0)) / current_price * 100, 2) if stop_loss_value else 0,
            "liquidity_penalty": liquidity_penalty < 1.0,
        }

    return {
        "action":           "trade",
        "recommended_amount": round(asset_amount, 6),
        "risk_pct":         round(final_risk_pct * 100, 2),
        "notional_value":   round(notional, 2),
        "sl_distance_pct":  round(abs(current_price - (stop_loss_value or 0)) / current_price * 100, 2) if stop_loss_value else 0,
        "liquidity_penalty": liquidity_penalty < 1.0
    }
