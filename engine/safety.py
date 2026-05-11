from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from db import SystemState, get_session

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingSafetyConfig:
    max_consecutive_errors: int = 3
    error_cooldown_minutes: int = 30
    max_daily_drawdown_pct: float = 3.0
    max_stale_seconds: int = 20
    max_spread_bps: float = 15.0
    min_live_balance: float = 0.0


class TradingCircuitBreaker:
    def __init__(self, config: TradingSafetyConfig, streamer=None, tools=None):
        self.config = config
        self.streamer = streamer
        self.tools = tools
        self._consecutive_errors = 0
        self._tripped_until = 0.0
        self._trip_reason = ""
        self._last_ok = time.time()

    @property
    def tripped(self) -> bool:
        return time.time() < self._tripped_until

    @property
    def trip_reason(self) -> str:
        if self.tripped:
            return self._trip_reason
        return ""

    def record_success(self):
        self._consecutive_errors = 0
        self._last_ok = time.time()

    def record_error(self, exc: Exception | str):
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.config.max_consecutive_errors:
            self.trip(f"too_many_errors: {exc}")

    def trip(self, reason: str, cooldown_minutes: Optional[int] = None):
        minutes = cooldown_minutes or self.config.error_cooldown_minutes
        self._tripped_until = time.time() + (minutes * 60)
        self._trip_reason = reason
        log.warning("Circuit breaker tripped: %s (for %s minutes)", reason, minutes)

    def reset(self):
        self._consecutive_errors = 0
        self._tripped_until = 0.0
        self._trip_reason = ""
        self._last_ok = time.time()

    def evaluate(self, symbols: List[str], balance: float, day_balance: float) -> Dict[str, Any]:
        now = time.time()
        if self.tripped:
            return {"allowed": False, "reason": self.trip_reason, "tripped": True}

        if day_balance > 0:
            dd_pct = max((day_balance - balance) / day_balance * 100.0, 0.0)
            if dd_pct >= self.config.max_daily_drawdown_pct:
                self.trip(f"daily_drawdown:{dd_pct:.2f}%")
                return {"allowed": False, "reason": self.trip_reason, "tripped": True}

        if balance <= self.config.min_live_balance:
            self.trip(f"balance_below_min:{balance:.2f}")
            return {"allowed": False, "reason": self.trip_reason, "tripped": True}

        if self.streamer is not None:
            stale_symbols = []
            for symbol in symbols:
                try:
                    age = self.streamer.market_age_seconds(symbol)
                    if age > self.config.max_stale_seconds:
                        stale_symbols.append((symbol, age))
                except Exception:
                    stale_symbols.append((symbol, float("inf")))
            if stale_symbols:
                label = ", ".join(f"{s}:{age:.1f}s" for s, age in stale_symbols[:3])
                self.trip(f"stale_market_data:{label}")
                return {"allowed": False, "reason": self.trip_reason, "tripped": True}

        return {"allowed": True, "reason": "ok", "tripped": False}


class StrategyValidationGate:
    def __init__(
        self,
        backtester,
        symbols: List[str],
        timeframe: str,
        limit: int = 400,
        train_size: int = 120,
        test_size: int = 60,
        step_size: int = 60,
        min_walk_forward_verdict: str = "promote_to_paper_candidate",
        max_risk_of_ruin_pct: float = 5.0,
        max_drawdown_pct: float = 20.0,
        min_profit_factor: float = 1.05,
    ):
        self.backtester = backtester
        self.symbols = symbols
        self.timeframe = timeframe
        self.limit = limit
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_walk_forward_verdict = min_walk_forward_verdict
        self.max_risk_of_ruin_pct = max_risk_of_ruin_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_profit_factor = min_profit_factor

    async def run(self) -> Dict[str, Any]:
        results = []
        allowed = True

        for symbol in self.symbols:
            try:
                wf = await self.backtester.run_walk_forward(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    limit=self.limit,
                    train_size=self.train_size,
                    test_size=self.test_size,
                    step_size=self.step_size,
                )
                mc = await self.backtester.run_monte_carlo(symbol, self.timeframe, limit=self.limit, iterations=400)

                verdict = wf.get("verdict", "reject_or_rework")
                profit_factor = float(wf.get("edge_quality", {}).get("profit_factor", 0.0) or 0.0)
                worst_dd = float(mc.get("worst_max_drawdown_pct", 100.0) or 100.0)
                risk_of_ruin = float(mc.get("risk_of_ruin_pct", 100.0) or 100.0)

                symbol_allowed = (
                    verdict == self.min_walk_forward_verdict
                    and risk_of_ruin <= self.max_risk_of_ruin_pct
                    and worst_dd <= self.max_drawdown_pct
                    and (profit_factor >= self.min_profit_factor or wf.get("avg_return_pct", 0.0) > 0)
                )

                results.append(
                    {
                        "symbol": symbol,
                        "walk_forward": wf,
                        "monte_carlo": mc,
                        "allowed": symbol_allowed,
                    }
                )
                if not symbol_allowed:
                    allowed = False
            except Exception as e:
                allowed = False
                results.append({"symbol": symbol, "error": str(e), "allowed": False})

        report = {
            "timestamp": time.time(),
            "timeframe": self.timeframe,
            "symbols": results,
            "allowed": allowed,
        }
        self._persist(report)
        return report

    def _persist(self, report: Dict[str, Any]):
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key="strategy_validation_report").first()
            payload = json.dumps(report, default=str)
            if row:
                row.value = payload
            else:
                session.add(SystemState(key="strategy_validation_report", value=payload))
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("Could not persist validation report: %s", e)
        finally:
            session.close()
