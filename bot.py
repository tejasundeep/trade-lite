import os
import asyncio
import argparse
import json
import logging
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict
from datetime import datetime, timezone
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box

from engine.tools import TradingTools
from engine.engine import TradingEngine
from engine.backtester import BacktestEngine
from engine.safety import TradingCircuitBreaker, TradingSafetyConfig, StrategyValidationGate
from trading.risk import MarketRiskManager, MarketRiskConfig
from db import init_db, get_session, Position

from trading.streamer import BinanceStreamer

load_dotenv()

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.FileHandler("bot.log"), RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("Invalid float for %s=%r; using %s", name, raw, default)
        return default


def _normalize_symbols(symbols: list[str], primary: str) -> list[str]:
    ordered = [s.strip() for s in symbols if s and s.strip()]
    if primary and primary not in ordered:
        ordered.insert(0, primary)
    return list(dict.fromkeys(ordered))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _HealthRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        bot = getattr(self.server, "bot", None)
        if self.path not in {"/health", "/status"}:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not_found"}).encode("utf-8"))
            return

        payload = bot.get_status_snapshot() if bot else {"ok": False, "error": "bot_unavailable"}
        if self.path == "/health":
            response = {
                "ok": True,
                "service": payload.get("service", "trade-lite"),
                "mode": payload.get("mode", "unknown"),
                "timestamp": payload.get("timestamp"),
                "alive": True,
            }
        else:
            response = payload

        data = json.dumps(response, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


class TradeXProClone:
    def __init__(self):
        key = os.getenv("BINANCE_API_KEY", "").strip()
        log.info(f"Initializing with API Key: {key[:4]}...{key[-4:] if len(key)>4 else ''}")
        trading_mode = os.getenv("TRADING_MODE", "futures").strip().lower()
        leverage = _env_int("FUTURES_LEVERAGE", 3)
        margin_type = os.getenv("MARGIN_TYPE", "ISOLATED").strip().upper()
        position_mode = os.getenv("POSITION_MODE", "ONE_WAY").strip().upper()
        
        self.tools = TradingTools(
            api_key=key,
            secret=os.getenv("BINANCE_SECRET", "").strip(),
            paper_trading=_env_bool("PAPER_TRADING", True),
            exchange_id=os.getenv("EXCHANGE_ID", "binance").strip(),
            trading_mode=trading_mode,
            leverage=leverage,
            margin_type=margin_type,
            position_mode=position_mode,
            max_daily_loss_pct=_env_float("MAX_DAILY_LOSS_PCT", 0.05),
        )
        self.risk = MarketRiskManager(MarketRiskConfig(
            max_risk_per_trade_pct=_env_float("MAX_RISK_PER_TRADE_PCT", 0.015),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.15),
            min_order_quote=_env_float("MIN_ORDER_QUOTE", 10.0),
            max_daily_loss_pct=_env_float("MAX_DAILY_LOSS_PCT", 0.05),
            cooldown_minutes=_env_int("COOLDOWN_MINUTES", 45)
        ))
        self.engine = TradingEngine(self.tools, self.risk)
        self.backtester = BacktestEngine(self.engine, self.tools)
        self.symbol = os.getenv("SYMBOL", "BTC/USDT").strip() or "BTC/USDT"
        configured_symbols = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
        self.symbols = _normalize_symbols(configured_symbols, self.symbol)
        self.paper_replay = _env_bool("PAPER_REPLAY", False)
        self.paper_replay_limit = _env_int("PAPER_REPLAY_LIMIT", 1000)
        self.paper_replay_alert_confidence = _env_float("PAPER_REPLAY_ALERT_CONFIDENCE", 0.75)
        self.paper_replay_max_drawdown_pct = _env_float("PAPER_REPLAY_MAX_DRAWDOWN_PCT", 12.0)
        self.reconcile_interval_seconds = _env_int("RECONCILE_INTERVAL_SECONDS", 60)
        self.last_reconcile_at = 0.0
        self.health_api_enabled = _env_bool("HEALTH_API_ENABLED", True)
        self.health_api_host = os.getenv("HEALTH_API_HOST", "0.0.0.0").strip()
        self.health_api_port = _env_int("HEALTH_API_PORT", 8080)
        self._health_api_server = None
        self._health_api_thread = None
        self.streamer = BinanceStreamer(self.symbols, adapter=self.tools.adapter)
        self.streamer.add_account_callback(self._handle_account_update)
        self.stats = {"cycles": 0, "last_action": "Initializing..."}
        
        # Persistent per-symbol states (caches HTF bias, etc.)
        self.symbol_states = {s: {"symbol": s, "balance": 0.0, "price": 0.0, "plan": {}, "indicators": {}} for s in self.symbols}
        self.cached_balance = 0.0
        self.last_balance_update = 0
        self.live_enabled = True
        self.validation_report = {}

        self.guard = TradingCircuitBreaker(
            TradingSafetyConfig(
                max_consecutive_errors=_env_int("MAX_CONSECUTIVE_ERRORS", 3),
                error_cooldown_minutes=_env_int("ERROR_COOLDOWN_MINUTES", 30),
                max_daily_drawdown_pct=_env_float("MAX_LIVE_DRAWDOWN_PCT", 3.0),
                max_stale_seconds=_env_int("MAX_STALE_SECONDS", 20),
                max_spread_bps=_env_float("MAX_ALLOWED_SPREAD_BPS", 15.0),
                min_live_balance=_env_float("MIN_LIVE_BALANCE", 0.0),
            ),
            streamer=self.streamer,
            tools=self.tools,
        )
        self.validation_gate = StrategyValidationGate(
            backtester=self.backtester,
            symbols=self.symbols,
            timeframe=os.getenv("TIMEFRAME", "5m"),
            limit=_env_int("VALIDATION_LIMIT", 400),
            train_size=_env_int("VALIDATION_TRAIN_SIZE", 120),
            test_size=_env_int("VALIDATION_TEST_SIZE", 60),
            step_size=_env_int("VALIDATION_STEP_SIZE", 60),
            min_walk_forward_verdict=os.getenv("VALIDATION_MIN_VERDICT", "promote_to_paper_candidate"),
            max_risk_of_ruin_pct=_env_float("VALIDATION_MAX_ROR_PCT", 5.0),
            max_drawdown_pct=_env_float("VALIDATION_MAX_DRAWDOWN_PCT", 20.0),
            min_profit_factor=_env_float("VALIDATION_MIN_PROFIT_FACTOR", 1.05),
        )
        
        self._trade_lock = asyncio.Lock()
        init_db()

        if self.tools.adapter.trading_mode == "futures" and not self.tools.adapter.paper_trading:
            for symbol in self.symbols:
                try:
                    self.tools.adapter.set_margin_type(symbol, margin_type)
                    self.tools.adapter.set_leverage(symbol, leverage)
                except Exception as e:
                    log.warning("Futures symbol bootstrap skipped for %s: %s", symbol, e)

    async def _bootstrap_validation(self):
        require_validation = _env_bool("REQUIRE_VALIDATION_ON_STARTUP", True)
        if self.tools.adapter.paper_trading or not require_validation:
            self.live_enabled = True
            return

        self.stats["last_action"] = "Running startup validation..."
        report = await self.validation_gate.run()
        self.validation_report = report
        self.live_enabled = bool(report.get("allowed", False))
        if not self.live_enabled:
            self.guard.trip("strategy_validation_failed", cooldown_minutes=24 * 365)
            log.warning("Startup validation failed. Live trading remains disabled.")
        else:
            log.info("Startup validation passed. Live trading armed.")

    async def _startup_reconcile(self):
        if self.tools.adapter.paper_trading:
            return
        try:
            report = await asyncio.to_thread(self.tools.reconcile_execution_state, self.symbols)
            self.last_reconcile_at = time.time()
            self.stats["last_action"] = f"Startup reconciled {len(report.get('symbols', []))} symbols"
            log.info(
                "Startup reconciliation complete | reconciled=%s cursor_updated=%s symbols=%d",
                report.get("reconciled", False),
                report.get("cursor_updated", False),
                len(report.get("symbols", [])),
            )
        except Exception as exc:
            log.warning("Startup reconciliation failed: %s", exc)

    def get_status_snapshot(self) -> Dict:
        adapter = getattr(self.tools, "adapter", None)
        open_positions = []
        for symbol in self.symbols:
            pos = self._safe_tool_call("get_open_position", None, symbol)
            if pos:
                open_positions.append(pos)

        return {
            "service": "trade-lite",
            "timestamp": _utc_now().isoformat(),
            "mode": os.getenv("BOT_MODE", "live"),
            "trading_mode": getattr(adapter, "trading_mode", "unknown"),
            "paper_trading": getattr(adapter, "paper_trading", False),
            "paper_replay": getattr(self, "paper_replay", False),
            "live_enabled": getattr(self, "live_enabled", False),
            "guard_tripped": getattr(self, "guard", None).tripped if getattr(self, "guard", None) else False,
            "guard_reason": getattr(self, "guard", None).trip_reason if getattr(self, "guard", None) else "",
            "cycles": self.stats.get("cycles", 0),
            "last_action": self.stats.get("last_action", ""),
            "balance": getattr(self, "cached_balance", 0.0),
            "last_balance_update": getattr(self, "last_balance_update", 0.0),
            "symbols": self.symbols,
            "open_positions_count": len(open_positions),
            "open_positions": open_positions,
            "validation_allowed": bool(self.validation_report.get("allowed", True)) if self.validation_report else None,
        }

    def start_health_api(self):
        if not self.health_api_enabled or self._health_api_server is not None:
            return

        try:
            server = ThreadingHTTPServer((self.health_api_host, self.health_api_port), _HealthRequestHandler)
        except OSError as exc:
            log.warning("Health API disabled; could not bind %s:%s: %s", self.health_api_host, self.health_api_port, exc)
            self.health_api_enabled = False
            return
        server.bot = self
        self._health_api_server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-api")
        thread.start()
        self._health_api_thread = thread
        log.info("Health API listening on http://%s:%s", self.health_api_host, self.health_api_port)

    def stop_health_api(self):
        if self._health_api_server is None:
            return
        try:
            self._health_api_server.shutdown()
            self._health_api_server.server_close()
        finally:
            self._health_api_server = None
            self._health_api_thread = None

    async def _run_paper_replay_mode(self):
        replay_symbols = [s.strip() for s in os.getenv("PAPER_REPLAY_SYMBOLS", "").split(",") if s.strip()] or [self.symbol]
        replay_timeframe = os.getenv("TIMEFRAME", "5m")
        self.stats["last_action"] = "Running paper replay..."
        reports = []
        for symbol in replay_symbols:
            report = await self.backtester.run_paper_replay(
                symbol=symbol,
                timeframe=replay_timeframe,
                limit=self.paper_replay_limit,
                alert_confidence_threshold=self.paper_replay_alert_confidence,
                max_drawdown_pct=self.paper_replay_max_drawdown_pct,
            )
            reports.append(report)
            metrics = report.get("metrics", {})
            log.info(
                "Paper replay %s | equity=%.2f return=%.2f%% max_dd=%.2f%% alerts=%s trips=%s",
                symbol,
                metrics.get("final_equity", 0.0),
                metrics.get("total_return_pct", 0.0),
                metrics.get("max_drawdown_pct", 0.0),
                len(report.get("alerts", [])),
                len(report.get("safety_events", [])),
            )

        self.validation_report["paper_replay"] = {
            "symbols": reports,
            "generated_at": _utc_now().isoformat(),
        }
        log.info("Paper replay complete for %d symbol(s).", len(reports))

    async def _run_backtest_mode(self, symbols: Optional[list[str]] = None, timeframe: Optional[str] = None, limit: Optional[int] = None):
        test_symbols = symbols or self.symbols
        test_timeframe = timeframe or os.getenv("TIMEFRAME", "5m")
        test_limit = limit or _env_int("BACKTEST_LIMIT", _env_int("VALIDATION_LIMIT", 400))
        self.stats["last_action"] = "Running backtest suite..."

        reports = []
        for symbol in test_symbols:
            walk_forward = await self.backtester.run_walk_forward(
                symbol=symbol,
                timeframe=test_timeframe,
                limit=test_limit,
                train_size=int(os.getenv("VALIDATION_TRAIN_SIZE", "120")),
                test_size=int(os.getenv("VALIDATION_TEST_SIZE", "60")),
                step_size=int(os.getenv("VALIDATION_STEP_SIZE", "60")),
            )
            monte_carlo = await self.backtester.run_monte_carlo(
                symbol=symbol,
                timeframe=test_timeframe,
                limit=test_limit,
                iterations=int(os.getenv("BACKTEST_MONTE_CARLO_ITERATIONS", "400")),
            )
            report = {
                "symbol": symbol,
                "timeframe": test_timeframe,
                "walk_forward": walk_forward,
                "monte_carlo": monte_carlo,
            }
            reports.append(report)
            log.info(
                "Backtest %s | verdict=%s robust=%.2f ror=%.2f%% worst_dd=%.2f%%",
                symbol,
                walk_forward.get("verdict", "n/a"),
                float(walk_forward.get("robust_score", 0.0) or 0.0),
                float(monte_carlo.get("risk_of_ruin_pct", 0.0) or 0.0),
                float(monte_carlo.get("worst_max_drawdown_pct", 0.0) or 0.0),
            )

        summary = {
            "mode": "backtest",
            "generated_at": _utc_now().isoformat(),
            "symbols": reports,
        }
        self.validation_report["backtest"] = summary

        from db import get_session, SystemState
        session = get_session()
        try:
            row = session.query(SystemState).filter_by(key="backtest_report").first()
            payload = json.dumps(summary, default=str)
            if row:
                row.value = payload
            else:
                session.add(SystemState(key="backtest_report", value=payload))
            session.commit()
        finally:
            session.close()

        print(json.dumps(summary, indent=2, default=str))
        return summary

    def create_layout(self) -> Layout:
        l = Layout()
        l.split_column(Layout(name="header", size=3), Layout(name="row1", size=12), Layout(name="row2", size=12), Layout(name="footer", size=3))
        l["row1"].split_row(Layout(name="market"), Layout(name="regime"), Layout(name="indicators"))
        l["row2"].split_row(Layout(name="position"), Layout(name="risk"), Layout(name="history"))
        return l

    def _color_value(self, val: float, positive_good: bool = True) -> str:
        if val > 0: return f"[{'green' if positive_good else 'red'}]{val:+.2f}[/]"
        if val < 0: return f"[{'red' if positive_good else 'green'}]{val:+.2f}[/]"
        return f"[dim]{val:.2f}[/]"

    def _safe_tool_call(self, name: str, default=None, *args, **kwargs):
        fn = getattr(self.tools, name, None)
        if not callable(fn):
            return default
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.debug("%s failed: %s", name, exc)
            return default

    def update_dashboard(self, layout: Layout, state: dict):
        inds, plan, htf = state.get("indicators", {}), state.get("plan", {}), state.get("htf_levels", {})
        smc, macro, trend, vol, vwap, of = inds.get("smc", {}), inds.get("macro", {}), inds.get("trend", {}), inds.get("vol", {}), inds.get("vwap", {}), inds.get("order_flow", {})
        price, balance, decision = state.get("price", 0), state.get("balance", 0), state.get("decision", {})
        execution_mode = os.getenv("TRADING_MODE", "futures").upper()
        mode = "[bold red]LIVE[/]" if os.getenv("PAPER_TRADING", "true").lower() == "false" else "[bold yellow]PAPER[/]"

        layout["header"].update(Panel(f"[bold cyan]TradeX Pro 2.0 (Elite)[/] | [white]{state.get('symbol')}[/] | {execution_mode} {mode} | [dim]{_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}[/] | Cycle #{self.stats['cycles']}", style="blue"))
        
        mkt = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        mkt.add_column("", style="dim"); mkt.add_column("", style="bold")
        px_col = "green" if smc.get("structure") == "Bullish" else "red" if smc.get("structure") == "Bearish" else "white"
        mkt.add_row("Price", f"[{px_col}]{price:,.2f}[/]")
        spread = self.streamer.get_spread(state.get('symbol', self.symbol))
        mkt.add_row("Spread", f"[dim]{spread:.2f}[/]")
        mkt.add_row("Balance", f"[green]{balance:,.2f} USDT[/]")
        mkt.add_row("Today PnL", self._color_value(state.get("today_pnl", 0.0)))
        mkt.add_row("Unrealized", self._color_value(state.get("unrealized_pnl", 0.0)))
        layout["market"].update(Panel(mkt, title="[bold]Real-time Market[/]", border_style="cyan"))

        reg = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        regime = plan.get("regime", {})
        if isinstance(regime, str):
            regime = {"trend": regime, "volatility": "N/A", "tradable": False}
        
        trend_str = str(regime.get('trend', 'N/A')).lower()
        tc = "green" if "bullish" in trend_str else "red" if "bearish" in trend_str else "yellow"
        reg.add_row("Trend", f"[{tc}]{regime.get('trend', 'N/A')}[/]"); reg.add_row("Volatility", regime.get("volatility", "N/A"))
        reg.add_row("Tradable", "[green]YES[/]" if regime.get("tradable") else "[red]NO[/]")
        reg.add_row("Structure", f"[bold yellow]{smc.get('structure', 'N/A')}[/]")
        reg.add_row("Iceberg", f"[bold cyan]{of.get('iceberg', 'None')}[/]")
        reg.add_row("HTF Bias", f"[bold]{plan.get('bias', 'Neutral')}[/]")
        layout["regime"].update(Panel(reg, title="[bold]Regime & OrderFlow[/]", border_style="magenta"))

        ind = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        ind.add_row("ADX", f"{trend.get('adx', 0):.1f}"); ind.add_row("ATR %", f"{vol.get('atr_pct', 0):.2f}%")
        ind.add_row("VWAP Z", f"{vwap.get('z_score', 0):.2f}")
        ind.add_row("1m Delta", f"{of.get('delta', 0):+.2f}")
        ind.add_row("CVD", f"[bold cyan]{of.get('cvd', 0):,.0f}[/]")
        layout["indicators"].update(Panel(ind, title="[bold]Signal Matrix[/]", border_style="blue"))

        pos_data, selected = state.get("position"), plan.get("selected")
        pos = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        if pos_data:
            sc = "green" if pos_data["side"] == "long" else "red"
            pos.add_row("Side", f"[{sc}]{pos_data['side'].upper()}[/]"); pos.add_row("Amount", f"{pos_data['amount']:.6f}")
            pos.add_row("SL", f"[red]{pos_data.get('stop_loss', 'N/A')}[/]"); pos.add_row("TP1 Hit", "[green]YES[/]" if pos_data.get("tp1_hit") else "No")
            if pos_data.get("trailing_stop"):
                pos.add_row("Trail SL", f"[magenta]{pos_data['trailing_stop']:,.2f}[/]")
        else: pos.add_row("Status", "[dim]No Position[/]")
        if selected:
            pos.add_row("", ""); pos.add_row("Edge", f"[bold]{selected['strategy']}[/]"); pos.add_row("Conf", f"{selected['confidence']*100:.0f}%")
        layout["position"].update(Panel(pos, title="[bold]Position & Trail[/]", border_style="yellow"))

        risk = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        metrics = state.get("performance", {})
        risk.add_row("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%"); risk.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        risk.add_row("Total Trades", f"{metrics.get('total_trades', 0)}"); risk.add_row("Total PnL", self._color_value(metrics.get("total_pnl", 0.0)))
        risk.add_row("Live Armed", "[green]YES[/]" if self.live_enabled else "[red]NO[/]")
        risk.add_row("Breaker", "[red]TRIPPED[/]" if self.guard.tripped else "[green]OK[/]")
        layout["risk"].update(Panel(risk, title="[bold]Risk & Performance[/]", border_style="red"))

        hist = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        hist.add_column("Symbol"); hist.add_column("Side"); hist.add_column("PnL"); hist.add_column("Reason")
        for t in state.get("recent_trades", []):
            hist.add_row(t["symbol"], t["side"].upper(), self._color_value(t.get("pnl", 0.0)), t.get("reason", "N/A")[:12])
        layout["history"].update(Panel(hist, title="[bold]History[/]", border_style="green"))

        layout["footer"].update(Panel(f"Status: {self.stats['last_action']}", style="dim"))

    async def _handle_account_update(self, update: dict):
        """Handle real-time balance and order updates from WebSocket."""
        event = update.get("event")
        if event in ["outboundAccountPosition", "balanceUpdate", "externalLockUpdate", "ACCOUNT_UPDATE"]:
            balance_updated = False
            for b in update.get("balances", []):
                asset = b.get("asset")
                free = b.get("free")
                if asset == "USDT" and free is not None:
                    self.cached_balance = float(free)
                    self.last_balance_update = time.time()
                    balance_updated = True
                    log.info(f"Real-time Balance Update: {self.cached_balance} USDT")
            if not balance_updated and event == "balanceUpdate" and "balances" not in update:
                self.last_balance_update = 0

            if update.get("positions"):
                session = get_session()
                try:
                    for pos in update.get("positions", []):
                        symbol = pos.get("symbol")
                        if not symbol:
                            continue
                        amount = float(pos.get("amount", 0.0) or 0.0)
                        existing = session.query(Position).filter_by(symbol=symbol).first()
                        if amount <= 0:
                            if existing:
                                session.delete(existing)
                            continue

                        side = str(pos.get("side", "long") or "long").lower()
                        stop_loss = float(existing.stop_loss if existing else 0.0)
                        take_profit = float(existing.take_profit if existing else 0.0)
                        if existing:
                            existing.amount = amount
                            existing.side = side
                            existing.avg_price = float(pos.get("entry_price", existing.avg_price) or existing.avg_price)
                            existing.updated_at = _utc_now()
                        else:
                            session.add(
                                Position(
                                    symbol=symbol,
                                    avg_price=float(pos.get("entry_price", 0.0) or 0.0),
                                    amount=amount,
                                    side=side,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    tp1_hit=False,
                                )
                            )
                    session.commit()
                except Exception as exc:
                    session.rollback()
                    log.warning("Real-time position sync failed: %s", exc)
                finally:
                    session.close()
        elif event in ["executionReport", "ORDER_TRADE_UPDATE"]:
            order = update.get("order", {})
            log.info(
                "Real-time Order Update: %s %s %s at %s",
                order.get("symbol", "N/A"),
                order.get("side", "N/A"),
                order.get("status", "N/A"),
                order.get("price", "N/A"),
            )
            try:
                await asyncio.to_thread(self.tools.reconcile_execution_state, self.symbols)
            except Exception as exc:
                log.warning("Realtime reconciliation failed: %s", exc)

    async def scan_symbol(self, symbol: str, balance: float) -> Optional[dict]:
        try:
            state = self.symbol_states[symbol]
            state["balance"] = balance
            res = await self.engine.run(state, execute=False, streamer=self.streamer)
            self.symbol_states[symbol] = res # Persist state (including HTF cache)
            
            dec = res.get("decision", {})
            if dec.get("action") == "trade" and dec.get("confidence", 0) >= 0.75: return res
        except Exception as e: log.error(f"scan {symbol}: {e}")
        return None

    def _correlation_ok(self, candidates: list) -> list:
        seen, filtered = {}, []
        for r in candidates:
            k = ("majors" if any(x in r["symbol"] for x in ("BTC", "ETH", "SOL")) else r["symbol"], r["decision"]["side"])
            if k not in seen: seen[k] = True; filtered.append(r)
        return filtered

    async def start(self):
        layout = self.create_layout()
        atr_map = {}
        # Initial state to avoid NameError if loop fails early
        state = {"symbol": self.symbol, "balance": 0.0, "price": 0.0, "indicators": {}, "plan": {}, "htf_levels": {}, "decision": {}, "position": None, "recent_trades": [], "performance": {}, "today_pnl": 0.0, "unrealized_pnl": 0.0}

        if self.paper_replay:
            self.start_health_api()
            await self._run_paper_replay_mode()
            self.stop_health_api()
            return
        
        await self._bootstrap_validation()
        self.start_health_api()
        self.streamer.start()
        log.info("Elite Streamer Started. Warming up local candle buffers...")
        await asyncio.sleep(10)

        try:
            self.cached_balance = self.tools.get_balance()
            self.last_balance_update = time.time()
        except Exception as e:
            log.warning("Initial balance sync failed: %s", e)

        await self._startup_reconcile()

        try:
            with Live(layout, refresh_per_second=4, screen=False):
                while True:
                    t0 = time.monotonic()
                    cycle_ok = False
                    try:
                        self.stats["cycles"] += 1
                        # 1. Real-time Balance is handled via callbacks.
                        # We only poll as a fallback if the stream hasn't updated in 5 minutes.
                        if time.time() - self.last_balance_update > 300:
                            try:
                                self.cached_balance = self.tools.get_balance()
                                self.last_balance_update = time.time()
                                log.info(f"Balance updated via REST: {self.cached_balance} USDT")
                                log.debug("Fallback Balance Fetch")
                            except Exception as e:
                                log.warning(f"Balance fetch failed: {e}")

                        balance = self.cached_balance
                        price_map = self.streamer.prices

                        day_balance = self.tools.get_day_balance_snapshot()
                        safety = self.guard.evaluate(self.symbols, balance, day_balance)
                        can_trade = self.live_enabled and safety.get("allowed", True)
                        if not can_trade:
                            self.stats["last_action"] = f"Trading paused: {safety.get('reason', 'disabled')}"

                        if can_trade:
                            # 2. Position Management
                            async with self._trade_lock:
                                self.stats["last_action"] = "Managing positions & Trail..."
                                now = time.time()
                                if now - self.last_reconcile_at >= self.reconcile_interval_seconds:
                                    reconcile_report = self.tools.reconcile_execution_state(self.symbols)
                                    self.last_reconcile_at = now
                                    self.stats["last_action"] = f"Reconciled {len(reconcile_report.get('symbols', []))} symbols"
                                # Safe extraction of SMC indicators
                                smc_map = {s: self.symbol_states.get(s, {}).get("indicators", {}).get("smc", {}) for s in self.symbols}
                                self.tools.manage_open_positions(price_map, atr_map, smc_map)

                            # 3. Scanning Symbols
                            self.stats["last_action"] = f"Scanning {len(self.symbols)} symbols..."
                            results = await asyncio.gather(*[self.scan_symbol(s, balance) for s in self.symbols])
                            candidates = self._correlation_ok([r for r in results if r is not None])

                            # Update ATR Map
                            for r in results:
                                if r:
                                    atr_map[r["symbol"]] = r["indicators"].get("vol", {}).get("atr", 0)

                            # 4. Execution Logic
                            state = self.symbol_states.get(self.symbol, state)
                            if candidates:
                                best = max(candidates, key=lambda r: r["decision"]["confidence"] * r["decision"]["expected_r"])
                                async with self._trade_lock:
                                    self.stats["last_action"] = f"Executing trade on {best['symbol']}..."
                                    final_res = await self.engine.run(best, execute=True, streamer=self.streamer)
                                    self.symbol_states[best['symbol']] = final_res
                                    state = final_res
                            else:
                                state = await self.engine.run(self.symbol_states[self.symbol], execute=False, streamer=self.streamer)
                                self.symbol_states[self.symbol] = state
                                atr_map[state["symbol"]] = state["indicators"].get("vol", {}).get("atr", 0)
                        else:
                            state = self.symbol_states.get(self.symbol, state)
                            state["decision"] = state.get("decision", {"action": "hold", "reason": safety.get("reason", "Trading paused")})
                            state["price"] = price_map.get(state["symbol"], state.get("price", 0))
                            state["balance"] = balance
                            state["position"] = self._safe_tool_call("get_open_position", None, state["symbol"])
                            state["recent_trades"] = self._safe_tool_call("get_recent_trades", [], 5)
                            state["performance"] = self._safe_tool_call("get_performance_metrics", {})
                            state["today_pnl"] = self._safe_tool_call("get_todays_realized_pnl", 0.0)
                            state["unrealized_pnl"] = self._safe_tool_call("get_unrealized_pnl", 0.0, state["symbol"], state["price"])

                        # 5. Enrichment for Dashboard
                        state["price"] = price_map.get(state["symbol"], state.get("price", 0))
                        state["balance"] = balance
                        state["position"] = self._safe_tool_call("get_open_position", None, state["symbol"])
                        state["recent_trades"] = self._safe_tool_call("get_recent_trades", [], 5)
                        state["performance"] = self._safe_tool_call("get_performance_metrics", {})
                        state["today_pnl"] = self._safe_tool_call("get_todays_realized_pnl", 0.0)
                        state["unrealized_pnl"] = self._safe_tool_call("get_unrealized_pnl", 0.0, state["symbol"], state["price"])

                        self.stats["last_action"] = "Elite Engine Idle"
                        cycle_ok = True
                    except Exception as e:
                        self.stats["last_action"] = f"Error: {str(e)[:40]}"
                        log.exception("Loop error")
                        self.guard.record_error(e)
                    
                    # Ensure dashboard updates even if logic fails
                    self.update_dashboard(layout, state)
                    if cycle_ok:
                        self.guard.record_success()
                    dt = time.monotonic() - t0
                    await asyncio.sleep(max(0.5, 3 - dt))
        finally:
            self.stop_health_api()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeX Pro bot runner")
    parser.add_argument("--mode", choices=["live", "replay", "backtest"], default=os.getenv("BOT_MODE", "live").lower())
    parser.add_argument("--symbols", help="Comma-separated symbols to override SYMBOLS")
    parser.add_argument("--symbol", help="Single symbol override for live/replay/backtest")
    parser.add_argument("--timeframe", help="Override timeframe")
    parser.add_argument("--limit", type=int, help="Override data limit for backtest/replay")
    args = parser.parse_args()

    if args.symbol:
        os.environ["SYMBOL"] = args.symbol
        os.environ["SYMBOLS"] = args.symbol
    elif args.symbols:
        os.environ["SYMBOLS"] = args.symbols

    if args.timeframe:
        os.environ["TIMEFRAME"] = args.timeframe

    bot = TradeXProClone()
    if args.mode == "replay":
        bot.paper_replay = True
        if args.limit:
            bot.paper_replay_limit = args.limit
        asyncio.run(bot.start())
    elif args.mode == "backtest":
        asyncio.run(bot._run_backtest_mode(
            symbols=[s.strip() for s in (args.symbols or os.getenv("SYMBOLS", bot.symbol)).split(",") if s.strip()],
            timeframe=args.timeframe or os.getenv("TIMEFRAME", "5m"),
            limit=args.limit,
        ))
    else:
        asyncio.run(bot.start())
