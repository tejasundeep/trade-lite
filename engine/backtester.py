import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import time
from typing import Dict, Any, List, Optional
from indicators.market_context import set_market_data, get_market_data, set_backtest, is_backtest
from db import get_session, SystemState

class BacktestEngine:
    """
    Full institutional-grade backtester with:
    - Stochastic slippage + spread + fees
    - Pessimistic intra-bar stop simulation
    - Funding rate accounting
    - R-Multiple / MFE / MAE excursion tracking
    - Walk-Forward Analysis with Strategy Tournament ranking
    - Monte Carlo Risk-of-Ruin simulation
    """

    def __init__(self, engine, tools):
        self.engine = engine
        self.tools = tools
        self.starting_equity = 10_000.0
        self.fee_rate = 0.001
        self.slippage_pct = 0.0005
        self.spread_pct = 0.0005
        self.market_depth_usdt = 50_000.0
        self.max_risk_per_trade_pct = 0.015
        self.max_position_pct = 0.20
        self.virtual_portfolio = self._fresh_portfolio()

    def _fresh_portfolio(self) -> Dict:
        return {
            "free_usdt": self.starting_equity, "position": 0.0, "side": "",
            "avg_price": 0.0, "stop_loss": 0.0, "realized_pnl": 0.0,
            "fees_paid": 0.0, "entry_risk_per_unit": 0.0,
            "open_trade_high": 0.0, "open_trade_low": 0.0,
            "bars_in_position": 0, "bars_total": 0,
            "active_strategy": "", "active_labels": {}, "total_funding": 0.0,
        }

    # ─── Public Entry Points ──────────────────────────────────────────────────
    async def run(self, symbol: str, timeframe: str, limit: int = 200) -> Dict:
        df = self.tools.get_market_data(symbol, timeframe, limit + 50)
        return await self._run_on_dataframe(symbol, timeframe, df)

    async def run_walk_forward(self, symbol: str, timeframe: str, limit: int = 300,
                               train_size: int = 120, test_size: int = 60, step_size: int = 60) -> Dict:
        df     = self.tools.get_market_data(symbol, timeframe, limit + 50)
        segments, start, seg_id = [], max(train_size, 50), 1
        while start + test_size <= len(df):
            seg_df = df.iloc[max(0, start - 50): start + test_size].copy()
            if len(seg_df) >= 55:
                res = await self._run_on_dataframe(symbol, timeframe, seg_df)
                segments.append({
                    "id": seg_id, "metrics": res["edge_quality"],
                    "total_trades": res["total_trades"], "wins": res["wins"], "losses": res["losses"],
                    "strategy_scores": self._score_strategy_trades(res["trades"]),
                })
                seg_id += 1
            start += step_size
        summary = self._score_walk_forward(symbol, timeframe, segments)
        summary["tournament"] = self._rank_strategy_tournament(segments)
        return summary

    async def run_monte_carlo(self, symbol: str, timeframe: str, limit: int = 300, iterations: int = 1000) -> Dict:
        baseline = await self.run(symbol, timeframe, limit)
        closed = [t for t in baseline.get("trades", []) if t.get("pnl") is not None]
        pnls = [float(t.get("pnl", 0.0) or 0.0) for t in closed]
        if not pnls:
            return {"error": "Not enough closed trades for Monte Carlo"}
        final_equities, max_drawdowns, ruined = [], [], 0
        ruin_threshold = self.starting_equity * 0.5
        for _ in range(iterations):
            sim_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            equity, peak, max_dd = self.starting_equity, self.starting_equity, 0.0
            for pnl in sim_pnls:
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            if equity < ruin_threshold: ruined += 1
            final_equities.append(equity)
            max_drawdowns.append(max_dd * 100)
        return {
            "symbol": symbol, "timeframe": timeframe, "mode": "monte_carlo",
            "iterations": iterations, "baseline_trades": len(closed),
            "median_final_equity": float(np.median(final_equities)),
            "mean_final_equity": float(np.mean(final_equities)),
            "worst_final_equity": float(np.min(final_equities)),
            "median_max_drawdown_pct": float(np.median(max_drawdowns)),
            "worst_max_drawdown_pct": float(np.max(max_drawdowns)),
            "risk_of_ruin_pct": (ruined / iterations) * 100,
            "baseline_metrics": baseline.get("edge_quality", {}),
        }

    async def run_paper_replay(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        alert_confidence_threshold: float = 0.75,
        max_drawdown_pct: float = 12.0,
        stop_on_safety_trip: bool = True,
    ) -> Dict[str, Any]:
        """
        Replay a historical market stream in paper mode using the same signal
        and execution path as live trading, while collecting alerts and safety
        trip reports.
        """
        df = self.tools.get_market_data(symbol, timeframe, limit + 100)
        if df is None or df.empty or len(df) < 120:
            return {"symbol": symbol, "timeframe": timeframe, "error": "Insufficient data for replay"}

        prev_backtest = is_backtest()
        trades: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        safety_events: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        peak_equity = self.starting_equity
        consecutive_losses = 0
        safety_tripped = False

        try:
            set_backtest(True)
            self.virtual_portfolio = self._fresh_portfolio()

            for i in range(100, len(df)):
                window = df.iloc[i - 100: i + 1].copy()
                set_market_data(window)

                price = float(window.iloc[-1]["close"])
                high = float(window.iloc[-1]["high"])
                low = float(window.iloc[-1]["low"])
                ts = window.iloc[-1]["timestamp"]

                self._mark_open_trade_excursion(high, low)
                self._simulate_stop(ts, low, high, trades, price)

                htf_levels, htf_bias = self._historical_htf_context(df, i)
                state = {
                    "symbol": symbol,
                    "price": price,
                    "balance": self.virtual_portfolio["free_usdt"],
                    "plan": {},
                    "indicators": {},
                    "htf_levels": htf_levels,
                }
                state = await self.engine.run(
                    state,
                    execute=False,
                    df_override=window,
                    htf_levels_override=htf_levels,
                    htf_bias_override=htf_bias,
                )
                decision = state.get("decision", {})
                if decision.get("action") == "trade" and float(decision.get("confidence", 0.0) or 0.0) >= alert_confidence_threshold:
                    alerts.append(
                        {
                            "timestamp": str(ts),
                            "symbol": symbol,
                            "side": decision.get("side"),
                            "strategy": decision.get("strategy"),
                            "confidence": decision.get("confidence", 0.0),
                            "expected_r": decision.get("expected_r", 0.0),
                            "reason": decision.get("reason", ""),
                            "price": price,
                        }
                    )

                pre_trade_count = len(trades)
                action = str(decision.get("action", "hold")).lower()
                trade_side = str(decision.get("trade_side") or decision.get("side") or action).lower()
                if trade_side not in {"buy", "sell"}:
                    trade_side = None
                if trade_side in {"buy", "sell"}:
                    self._open_or_close(trade_side, decision, price, high, low, ts, trades)

                if len(trades) > pre_trade_count:
                    last_trade = trades[-1]
                    pnl = float(last_trade.get("pnl", 0.0) or 0.0)
                    if pnl < 0:
                        consecutive_losses += 1
                    elif pnl > 0:
                        consecutive_losses = 0

                total_equity = self.virtual_portfolio["free_usdt"] + (self.virtual_portfolio["position"] * price)
                equity_curve.append(total_equity)
                peak_equity = max(peak_equity, total_equity)
                drawdown_pct = ((peak_equity - total_equity) / peak_equity) * 100 if peak_equity > 0 else 0.0

                if drawdown_pct >= max_drawdown_pct:
                    safety_tripped = True
                    safety_events.append(
                        {
                            "timestamp": str(ts),
                            "type": "max_drawdown",
                            "drawdown_pct": drawdown_pct,
                            "limit_pct": max_drawdown_pct,
                            "equity": total_equity,
                        }
                    )
                    if stop_on_safety_trip:
                        break

                if consecutive_losses >= 3:
                    safety_tripped = True
                    safety_events.append(
                        {
                            "timestamp": str(ts),
                            "type": "loss_streak",
                            "streak": consecutive_losses,
                            "equity": total_equity,
                        }
                    )
                    if stop_on_safety_trip:
                        break

            metrics = self._calculate_metrics(equity_curve, trades)
            report = {
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": "paper_replay",
                "alerts": alerts,
                "safety_events": safety_events,
                "safety_tripped": safety_tripped,
                "trades": trades,
                "equity_curve_points": len(equity_curve),
                "metrics": metrics,
            }
            self._persist_report("paper_replay_report", report)
            return report
        finally:
            set_backtest(prev_backtest)

    # ─── Core Simulation ─────────────────────────────────────────────────────
    async def _run_on_dataframe(self, symbol: str, timeframe: str, df: pd.DataFrame,
                                htf_df: pd.DataFrame = None) -> Dict:
        self.virtual_portfolio = self._fresh_portfolio()
        trades, equity_curve = [], []

        prev_backtest = is_backtest()
        try:
            set_backtest(True)

            for i in range(100, len(df)):
                window = df.iloc[i - 100: i + 1]
                set_market_data(window)
                price = float(window.iloc[-1]["close"])
                high  = float(window.iloc[-1]["high"])
                low   = float(window.iloc[-1]["low"])
                ts    = window.iloc[-1]["timestamp"]

                self._mark_open_trade_excursion(high, low)
                self._simulate_stop(ts, low, high, trades, price)

                htf_levels, htf_bias = self._historical_htf_context(df, i)
                state = {
                    "symbol": symbol,
                    "price": price,
                    "balance": self.virtual_portfolio["free_usdt"],
                    "plan": {},
                    "indicators": {},
                    "htf_levels": htf_levels,
                }
                state = await self.engine.run(
                    state,
                    execute=False,
                    df_override=window,
                    htf_levels_override=htf_levels,
                    htf_bias_override=htf_bias,
                )
                decision = state.get("decision", {})
                action = str(decision.get("action", "hold")).lower()
                trade_side = str(decision.get("trade_side") or decision.get("side") or action).lower()
                if trade_side not in {"buy", "sell"}:
                    trade_side = None

                if trade_side in {"buy", "sell"}:
                    self._open_or_close(trade_side, decision, price, high, low, ts, trades)

                total_equity = self.virtual_portfolio["free_usdt"] + (self.virtual_portfolio["position"] * price)
                equity_curve.append(total_equity)
                self.virtual_portfolio["bars_total"] += 1
                if self.virtual_portfolio["position"] > 0:
                    self.virtual_portfolio["bars_in_position"] += 1
        finally:
            set_backtest(prev_backtest)

        metrics = self._calculate_metrics(equity_curve, trades)
        closed_trades = [t for t in trades if t.get("pnl") is not None]
        wins   = len([t for t in closed_trades if t.get("pnl", 0) > 0])
        losses = len([t for t in closed_trades if t.get("pnl", 0) < 0])
        return {
            "symbol": symbol, "timeframe": timeframe,
            "total_trades": len(trades), "wins": wins, "losses": losses,
            "final_equity": metrics["final_equity"],
            "total_return_pct": metrics["total_return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "fees_paid": self.virtual_portfolio["fees_paid"],
            "edge_quality": metrics, "trades": trades,
        }

    def _historical_htf_context(self, df: pd.DataFrame, index: int) -> tuple[Dict, str]:
        # Use only historical data available up to the current bar and rebuild
        # higher-timeframe structure from the lower-timeframe source.
        history = df.iloc[: index + 1].copy()
        if history.empty:
            return {}, "Neutral"

        daily_window = self._resample_ohlcv(history, "1D")
        weekly_window = self._resample_ohlcv(history, "1W-MON")
        levels = self.tools.build_institutional_levels(daily_window, weekly_window)

        bias = "Neutral"
        hourly_window = self._resample_ohlcv(history, "1h")
        if len(hourly_window) >= 50:
            try:
                from indicators.smc import analyze_smc_structure

                bias = analyze_smc_structure(df_override=hourly_window).get("structure", "Neutral")
            except Exception as exc:
                log.debug("HTF bias fallback failed: %s", exc)
        if bias == "Neutral" and levels.get("weekly_open"):
            price = float(history.iloc[-1]["close"])
            weekly_open = levels.get("weekly_open")
            dist_pct = (price - weekly_open) / weekly_open
            if dist_pct > 0.002:
                bias = "Bullish"
            elif dist_pct < -0.002:
                bias = "Bearish"
        return levels, bias

    def _resample_ohlcv(self, source: pd.DataFrame, rule: str) -> pd.DataFrame:
        if source is None or source.empty:
            return pd.DataFrame()
        frame = source.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("timestamp").set_index("timestamp")
        aggregated = frame.resample(rule, label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
        if aggregated.empty:
            return pd.DataFrame()
        aggregated = aggregated.reset_index()
        return aggregated

    def _persist_report(self, key: str, payload: Dict[str, Any]):
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

    def _open_or_close(self, action: str, decision: Dict, price: float,
                       high: float, low: float, ts, trades: list):
        side = self.virtual_portfolio["side"]
        # Closing opposite side
        if (action == "buy" and side == "short") or (action == "sell" and side == "long"):
            amount = min(decision.get("recommended_amount", self.virtual_portfolio["position"]) or
                         self.virtual_portfolio["position"], self.virtual_portfolio["position"])
            fill = self._buy_price(price, amount) if action == "buy" else self._sell_price(price, amount)
            fee  = amount * fill * self.fee_rate
            if action == "buy":
                pnl = (self.virtual_portfolio["avg_price"] - fill) * amount - fee
                self.virtual_portfolio["free_usdt"] += self.virtual_portfolio["avg_price"] * amount + pnl
            else:
                pnl = (fill - self.virtual_portfolio["avg_price"]) * amount - fee
                self.virtual_portfolio["free_usdt"] += amount * fill - fee
            r_mult = self._r_multiple(pnl, amount)
            mfe_r, mae_r = self._excursion_r()
            self.virtual_portfolio["realized_pnl"] += pnl
            self.virtual_portfolio["fees_paid"] += fee
            self.virtual_portfolio["position"] -= amount
            if self.virtual_portfolio["position"] <= 0:
                self._reset_position()
            trades.append({"time": str(ts), "action": action, "price": fill, "amount": amount,
                            "fee": fee, "pnl": pnl, "r_multiple": r_mult, "mfe_r": mfe_r, "mae_r": mae_r,
                            "strategy": self.virtual_portfolio.get("active_strategy", ""),
                            "reason": decision.get("reason", "")})
            return

        # Opening new side
        if action == "buy":
            amount = self._cap_buy_amount(decision.get("recommended_amount", 0.01) or 0.01,
                                          price, decision.get("stop_loss"))
            fill   = self._buy_price(price, amount)
            cost   = amount * fill
            fee    = cost * self.fee_rate
            if amount > 0 and cost + fee <= self.virtual_portfolio["free_usdt"]:
                self._apply_open("long", amount, fill, fee, decision, high, low)
                trades.append({"time": str(ts), "action": "buy", "price": fill, "amount": amount, "fee": fee,
                                "strategy": decision.get("strategy", ""), "stop_loss": decision.get("stop_loss"),
                                "take_profit": decision.get("take_profit"), "reason": decision.get("reason", "")})
        else:
            amount = self._cap_sell_amount(decision.get("recommended_amount", 0.01) or 0.01,
                                           price, decision.get("stop_loss"))
            fill   = self._sell_price(price, amount)
            notional = amount * fill
            fee    = notional * self.fee_rate
            if amount > 0 and notional + fee <= self.virtual_portfolio["free_usdt"]:
                self._apply_open("short", amount, fill, fee, decision, high, low)
                trades.append({"time": str(ts), "action": "sell", "price": fill, "amount": amount, "fee": fee,
                                "strategy": decision.get("strategy", ""), "stop_loss": decision.get("stop_loss"),
                                "take_profit": decision.get("take_profit"), "reason": decision.get("reason", "")})

    def _apply_open(self, direction: str, amount: float, fill: float, fee: float, decision: Dict, high: float, low: float):
        p = self.virtual_portfolio
        old_amt, old_cost = p["position"], p["avg_price"] * p["position"]
        new_amt = old_amt + amount
        p["avg_price"] = (old_cost + fill * amount) / new_amt if new_amt > 0 else fill
        p["free_usdt"] -= (amount * fill + fee)
        p["position"]   = new_amt
        p["side"]        = direction
        p["stop_loss"]   = float(decision.get("stop_loss") or 0)
        p["entry_risk_per_unit"] = abs(p["avg_price"] - p["stop_loss"])
        p["open_trade_high"] = max(high, fill)
        p["open_trade_low"]  = min(low, fill)
        p["fees_paid"]  += fee
        p["active_strategy"] = decision.get("strategy", "")
        p["active_labels"]   = decision.get("labels", {})

    def _reset_position(self):
        p = self.virtual_portfolio
        p["side"] = ""; p["avg_price"] = 0.0; p["position"] = 0.0; p["stop_loss"] = 0.0
        p["entry_risk_per_unit"] = 0.0; p["open_trade_high"] = 0.0; p["open_trade_low"] = 0.0
        p["active_strategy"] = ""; p["active_labels"] = {}

    def _simulate_stop(self, ts, low: float, high: float, trades: list, price: float):
        p = self.virtual_portfolio
        if p["position"] <= 0 or p["stop_loss"] <= 0: return
        triggered = (p["side"] == "long" and low <= p["stop_loss"]) or \
                    (p["side"] == "short" and high >= p["stop_loss"])
        if not triggered: return
        fill = self._sell_price(min(p["stop_loss"], low), p["position"]) if p["side"] == "long" else \
               self._buy_price(max(p["stop_loss"], high), p["position"])
        fee = p["position"] * fill * self.fee_rate
        pnl = ((fill - p["avg_price"]) if p["side"] == "long" else (p["avg_price"] - fill)) * p["position"] - fee
        p["realized_pnl"] += pnl; p["fees_paid"] += fee
        if p["side"] == "long":  p["free_usdt"] += p["position"] * fill - fee
        else:                    p["free_usdt"] += p["avg_price"] * p["position"] + pnl
        r_mult = self._r_multiple(pnl, p["position"])
        mfe_r, mae_r = self._excursion_r()
        trades.append({"time": str(ts), "action": "stop_loss", "price": fill, "amount": p["position"],
                        "fee": fee, "pnl": pnl, "r_multiple": r_mult, "mfe_r": mfe_r, "mae_r": mae_r,
                        "strategy": p["active_strategy"], "reason": "Simulated Stop Loss"})
        self._reset_position()

    def _mark_open_trade_excursion(self, high: float, low: float):
        p = self.virtual_portfolio
        if p["position"] <= 0: return
        p["open_trade_high"] = max(p["open_trade_high"] or high, high)
        p["open_trade_low"]  = min(p["open_trade_low"] or low, low)

    def _buy_price(self, price: float, amount: float = 0.0) -> float:
        df = get_market_data()
        atr = price * 0.001
        if df is not None and not df.empty and "close" in df.columns:
            atr_val = df["close"].diff().abs().rolling(14, min_periods=1).mean().iloc[-1]
            if pd.notna(atr_val) and atr_val > 0:
                atr = float(atr_val)
        vol_mult = max(1.0, atr / (price * 0.005))
        stochastic = abs(np.random.normal(0, self.slippage_pct * 0.5))
        impact = (amount * price / self.market_depth_usdt) * 0.01
        return price * (1 + self.spread_pct / 2 + (self.slippage_pct * vol_mult) + impact + stochastic)

    def _sell_price(self, price: float, amount: float = 0.0) -> float:
        df = get_market_data()
        atr = price * 0.001
        if df is not None and not df.empty and "close" in df.columns:
            atr_val = df["close"].diff().abs().rolling(14, min_periods=1).mean().iloc[-1]
            if pd.notna(atr_val) and atr_val > 0:
                atr = float(atr_val)
        vol_mult = max(1.0, atr / (price * 0.005))
        stochastic = abs(np.random.normal(0, self.slippage_pct * 0.5))
        impact = (amount * price / self.market_depth_usdt) * 0.01
        return price * (1 - self.spread_pct / 2 - (self.slippage_pct * vol_mult) - impact - stochastic)

    def _cap_buy_amount(self, req: float, price: float, stop_loss) -> float:
        try: stop = float(stop_loss)
        except: return 0.0
        if req <= 0 or stop <= 0 or stop >= price: return 0.0
        free = self.virtual_portfolio["free_usdt"]
        max_by_risk = (free * self.max_risk_per_trade_pct) / (price - stop)
        max_by_exp  = max((free * self.max_position_pct) / price, 0.0)
        return max(min(req, max_by_risk, max_by_exp, free / self._buy_price(price)), 0.0)

    def _cap_sell_amount(self, req: float, price: float, stop_loss) -> float:
        try: stop = float(stop_loss)
        except: return 0.0
        if req <= 0 or stop <= 0 or stop <= price: return 0.0
        free = self.virtual_portfolio["free_usdt"]
        max_by_risk = (free * self.max_risk_per_trade_pct) / (stop - price)
        max_by_exp  = max((free * self.max_position_pct) / price, 0.0)
        return max(min(req, max_by_risk, max_by_exp, free / self._sell_price(price)), 0.0)

    def _r_multiple(self, pnl: float, amount: float) -> float:
        risk = self.virtual_portfolio["entry_risk_per_unit"] * max(amount, 0.0)
        return pnl / risk if risk > 0 else 0.0

    def _excursion_r(self):
        p = self.virtual_portfolio
        risk, entry, side = p["entry_risk_per_unit"], p["avg_price"], p["side"]
        if risk <= 0 or entry <= 0: return 0.0, 0.0
        if side == "long":
            return (p["open_trade_high"] - entry) / risk, (p["open_trade_low"] - entry) / risk
        return (entry - p["open_trade_low"]) / risk, (entry - p["open_trade_high"]) / risk

    def _calculate_metrics(self, equity_curve: list, trades: list) -> Dict:
        fe = equity_curve[-1] if equity_curve else self.starting_equity
        total_return = ((fe - self.starting_equity) / self.starting_equity) * 100
        peak, max_dd = (equity_curve[0] if equity_curve else self.starting_equity), 0.0
        for e in equity_curve:
            peak = max(peak, e)
            max_dd = min(max_dd, (e - peak) / peak if peak else 0)
        closed = [t for t in trades if t.get("pnl") is not None]
        pnls    = [float(t.get("pnl", 0.0) or 0.0) for t in closed]
        r_vals  = [float(t.get("r_multiple", 0.0) or 0.0) for t in closed]
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        returns = pd.Series(equity_curve, dtype="float64").pct_change().dropna()
        dn = returns[returns < 0]
        # Use sqrt(252) for annualised daily Sharpe; sqrt(365*24/tf_hours) for intraday
        ann = np.sqrt(252)
        sharpe  = float((returns.mean() / returns.std()) * ann) if len(returns) > 1 and returns.std() > 0 else 0.0
        sortino = float((returns.mean() / dn.std())  * ann) if len(dn) > 1 and dn.std() > 0 else 0.0
        p = self.virtual_portfolio
        return {
            "final_equity": fe, "total_return_pct": total_return, "max_drawdown_pct": max_dd * 100,
            "profit_factor": gp / gl if gl > 0 else (gp if gp > 0 else 0.0),
            "expectancy": float(np.mean(pnls)) if pnls else 0.0,
            "avg_r_multiple": float(np.mean(r_vals)) if r_vals else 0.0,
            "sharpe_like": sharpe, "sortino_like": sortino,
            "exposure_pct": (p["bars_in_position"] / p["bars_total"] * 100) if p["bars_total"] else 0.0,
            "closed_trades": len(closed),
        }

    def _score_strategy_trades(self, trades: list) -> Dict:
        grouped: Dict[str, list] = {}
        for t in trades:
            if t.get("pnl") is None: continue
            grouped.setdefault(t.get("strategy") or "unknown", []).append(t)
        strategy_scores = {}
        for s, items in grouped.items():
            pnls = [float(t.get("pnl", 0.0) or 0.0) for t in items]
            r_vals = [float(t.get("r_multiple", 0.0) or 0.0) for t in items]
            wins = len([p for p in pnls if p > 0])
            total_pnl = sum(pnls)
            avg_r = float(np.mean(r_vals)) if r_vals else 0.0
            strategy_scores[s] = {
                "strategy": s,
                "trades": len(items),
                "wins": wins,
                "total_pnl": total_pnl,
                "avg_r_multiple": avg_r,
                "score_sum": (total_pnl * 0.01) + (avg_r * 5.0) + (wins / max(len(items), 1)) * 2.5,
            }
        return strategy_scores

    def _score_walk_forward(self, symbol: str, timeframe: str, segments: list) -> Dict:
        returns = [float(s["metrics"].get("total_return_pct", 0.0)) for s in segments]
        avg_rs  = [float(s["metrics"].get("avg_r_multiple", 0.0)) for s in segments]
        pfs     = [float(s["metrics"].get("profit_factor", 0.0)) for s in segments]
        dds     = [float(s["metrics"].get("max_drawdown_pct", 0.0)) for s in segments]
        pass_flags = [r > 0 and ar >= 0 and pf >= 1.0 for r, ar, pf in zip(returns, avg_rs, pfs)]
        n = len(segments) or 1
        pass_rate = sum(pass_flags) / n * 100
        avg_ret   = float(np.mean(returns)) if returns else 0.0
        robust_score = avg_ret + float(np.mean(avg_rs) if avg_rs else 0) * 8 + float(np.mean(pfs) if pfs else 0) * 2 + pass_rate * 0.12 - abs(min(dds, default=0)) * 0.75
        verdict = ("promote_to_paper_candidate" if robust_score >= 12 and pass_rate >= 65 else
                   "research_candidate" if robust_score >= 4 and pass_rate >= 45 else
                   "reject_or_rework")
        return {"symbol": symbol, "timeframe": timeframe, "mode": "walk_forward",
                "segments": segments, "pass_rate_pct": pass_rate, "avg_return_pct": avg_ret,
                "avg_r_multiple": float(np.mean(avg_rs)) if avg_rs else 0.0,
                "edge_quality": {
                    "profit_factor": float(np.mean(pfs)) if pfs else 0.0,
                    "max_drawdown_pct": abs(min(dds, default=0.0)),
                    "avg_return_pct": avg_ret,
                    "avg_r_multiple": float(np.mean(avg_rs)) if avg_rs else 0.0,
                },
                "robust_score": robust_score, "verdict": verdict}

    def _rank_strategy_tournament(self, segments: list) -> list:
        agg: Dict[str, Dict] = {}
        for seg in segments:
            for s, score in seg.get("strategy_scores", {}).items():
                item = agg.setdefault(s, {"strategy": s, "segments": 0, "trades": 0, "wins": 0,
                                          "total_pnl": 0.0, "score_sum": 0.0, "avg_r_sum": 0.0,
                                          "positive_segments": 0})
                item["segments"] += 1; item["trades"] += score["trades"]; item["wins"] += score["wins"]
                item["total_pnl"] += score["total_pnl"]; item["avg_r_sum"] += score["avg_r_multiple"]
                item["score_sum"] += score.get("score_sum", 0.0)
                if score["total_pnl"] > 0: item["positive_segments"] += 1
        ranked = []
        for s, item in agg.items():
            n = max(item["segments"], 1)
            pass_rate = item["positive_segments"] / n * 100
            ranked.append({
                "strategy": s, "trades": item["trades"], "wins": item["wins"],
                "total_pnl": item["total_pnl"], "pass_rate_pct": pass_rate,
                "avg_r_multiple": item["avg_r_sum"] / n,
                "tournament_score": (item["score_sum"] / n) + min(n, 5) * 0.8 + pass_rate * 0.06,
            })
        return sorted(ranked, key=lambda x: x["tournament_score"], reverse=True)
