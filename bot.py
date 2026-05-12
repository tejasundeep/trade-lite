import os
import asyncio
import argparse
import json
import logging
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict, Any
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
from engine.grid import GridConfig, GridManager
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
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    logging.getLogger(__name__).warning("Invalid boolean for %s=%r; using %s", name, raw, default)
    return default


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
        self.bot_mode = os.getenv("BOT_MODE", "live").strip().lower()
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
        self.grid = GridManager(
            self.tools,
            GridConfig(
                enabled=_env_bool("GRID_ENABLED", True),
                levels=_env_int("GRID_LEVELS", 8),
                range_pct=_env_float("GRID_RANGE_PCT", 0.08),
                total_quote_pct=_env_float("GRID_TOTAL_QUOTE_PCT", 0.20),
                min_order_quote=_env_float("GRID_MIN_ORDER_QUOTE", _env_float("MIN_ORDER_QUOTE", 10.0)),
                recenter_threshold_pct=_env_float("GRID_RECENTER_THRESHOLD_PCT", 0.035),
                refresh_seconds=_env_int("GRID_REFRESH_SECONDS", 3),
                trade_lookback=_env_int("GRID_TRADE_LOOKBACK", 500),
                max_inventory_pct=_env_float("GRID_MAX_INVENTORY_PCT", _env_float("MAX_POSITION_PCT", 0.15)),
            ),
        )
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
        self.runtime_notice = ""
        
        # Persistent per-symbol states (caches HTF bias, etc.)
        self.symbol_states = {s: {"symbol": s, "balance": 0.0, "price": 0.0, "plan": {}, "indicators": {}} for s in self.symbols}
        self.cached_balance = 0.0
        self.last_balance_update = 0
        self.live_enabled = True
        self.validation_report = {}
        self.validation_status = {"active": False, "symbol": "", "index": 0, "total": 0, "stage": ""}
        self.validation_started_at = 0.0

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
            progress_callback=self._set_validation_progress,
        )
        
        self._trade_lock = asyncio.Lock()
        init_db()
        self.grid_states: Dict[str, dict] = {}

        if self.tools.adapter.trading_mode == "futures" and not self.tools.adapter.paper_trading:
            if _env_bool("FUTURES_BOOTSTRAP_ON_STARTUP", False):
                for symbol in self.symbols:
                    try:
                        self.tools.adapter.set_margin_type(symbol, margin_type)
                        self.tools.adapter.set_leverage(symbol, leverage)
                    except Exception as e:
                        log.warning("Futures symbol bootstrap skipped for %s: %s", symbol, e)
            else:
                log.info("Futures margin/leverage bootstrap is disabled on startup. Set FUTURES_BOOTSTRAP_ON_STARTUP=true to enable it.")

    async def _bootstrap_validation(self):
        require_validation = _env_bool("REQUIRE_VALIDATION_ON_STARTUP", True)
        if self.bot_mode == "grid" or self.tools.adapter.paper_trading or not require_validation:
            self.live_enabled = True
            self.validation_status = {"active": False, "symbol": "", "index": 0, "total": 0, "stage": "skipped"}
            return

        self.validation_status = {"active": True, "symbol": "", "index": 0, "total": len(self.symbols), "stage": "starting"}
        self.validation_started_at = time.monotonic()
        self.stats["last_action"] = "Running startup validation..."
        report = await asyncio.to_thread(self._run_validation_blocking)
        self.validation_report = report
        self.live_enabled = bool(report.get("allowed", False))
        self.validation_status = {"active": False, "symbol": "", "index": len(report.get("symbols", [])), "total": len(self.symbols), "stage": "done"}
        if not self.live_enabled:
            self.guard.trip("strategy_validation_failed", cooldown_minutes=24 * 365)
            log.warning("Startup validation failed. Live trading remains disabled.")
        else:
            log.info("Startup validation passed. Live trading armed.")

    def _run_validation_blocking(self) -> Dict[str, Any]:
        return asyncio.run(self.validation_gate.run())

    def _set_validation_progress(self, payload: Dict[str, object]):
        current = dict(self.validation_status)
        current.update(payload or {})
        self.validation_status = current

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
            "bot_mode": self.bot_mode,
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
            "grid": {s: self.grid.get_state(s) for s in self.symbols} if self.bot_mode == "grid" else {},
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
        try:
            if val is None:
                return "[dim]N/A[/]"
            num = float(val)
        except (TypeError, ValueError):
            return f"[dim]{val}[/]"

        if num > 0:
            return f"[{'green' if positive_good else 'red'}]{num:+.2f}[/]"
        if num < 0:
            return f"[{'red' if positive_good else 'green'}]{num:+.2f}[/]"
        return f"[dim]{num:.2f}[/]"

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
        adapter = getattr(self.tools, "adapter", None)
        execution_mode = getattr(adapter, "trading_mode", os.getenv("TRADING_MODE", "futures")).upper()
        bot_mode = self.bot_mode.upper()
        mode = "[bold yellow]PAPER[/]" if getattr(adapter, "paper_trading", False) else "[bold red]LIVE[/]"
        notice = f" | [yellow]{self.runtime_notice}[/]" if self.runtime_notice else ""

        layout["header"].update(Panel(f"[bold cyan]TradeX Pro 2.0 (Elite)[/] | [white]{state.get('symbol')}[/] | {bot_mode} | {execution_mode} {mode} | [dim]{_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}[/] | Cycle #{self.stats['cycles']}{notice}", style="blue"))
        
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
        ind.add_row("1m Delta", f"[bold {'green' if of.get('delta', 0) > 0 else 'red'}]{of.get('delta', 0):+,.2f}[/]")
        ind.add_row("CVD", f"[bold cyan]{of.get('cvd', 0):,.0f}[/]")
        
        rsi_val = inds.get("rsi", 50.0)
        rsi_color = "red" if rsi_val > 70 else "green" if rsi_val < 30 else "white"
        ind.add_row("RSI (14)", f"[{rsi_color}]{rsi_val:.1f}[/]")
        
        macd = inds.get("macd", {})
        macd_color = "green" if macd.get("hist", 0) > 0 else "red"
        ind.add_row("MACD Hist", f"[{macd_color}]{macd.get('hist', 0):.4f}[/]")
        
        bb = inds.get("bollinger", {})
        price = state.get("price", 0.0)
        bb_pos = "Upper" if price > bb.get("upper", 0) else "Lower" if price < bb.get("lower", 0) else "Mid"
        ind.add_row("BB Zone", f"[cyan]{bb_pos}[/]")
        
        layout["indicators"].update(Panel(ind, title="[bold]Signal Matrix[/]", border_style="blue"))

        pos_data, selected = state.get("position"), plan.get("selected")
        grid_info = state.get("grid", {}) or {}
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
        if grid_info:
            pos.add_row("", "")
            pos.add_row("Grid", f"{grid_info.get('fills_applied', 0)} fills / {grid_info.get('levels', 0)} levels")
            pos.add_row("Anchor", f"{grid_info.get('anchor_price', 0.0):,.2f}")
            if grid_info.get("grid_step_pct"):
                pos.add_row("Step", f"{float(grid_info.get('grid_step_pct', 0.0)) * 100:.2f}%")
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
            symbol = str(t.get("symbol", "N/A"))
            side = str(t.get("side", "N/A")).upper()
            reason = str(t.get("reason", "N/A"))[:12]
            hist.add_row(symbol, side, self._color_value(t.get("pnl", 0.0)), reason)
        layout["history"].update(Panel(hist, title="[bold]History[/]", border_style="green"))

        validation_note = ""
        if self.validation_status.get("active"):
            symbol_name = self.validation_status.get("symbol") or "..."
            index = int(self.validation_status.get("index", 0) or 0)
            total = int(self.validation_status.get("total", 0) or 0)
            stage = str(self.validation_status.get("stage", "") or "")
            validation_note = f" | Validation: {index}/{total} {symbol_name} {stage}".rstrip()
        elif self.validation_status.get("stage") == "done" and self.validation_report:
            validation_note = " | Validation: done"

        layout["footer"].update(Panel(f"Status: {self.stats['last_action']}{validation_note}{notice}", style="dim"))

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

    def _grid_dashboard_state(self, symbol: str, grid_result: dict, can_trade: bool, balance: float) -> dict:
        price = float(grid_result.get("price", 0.0) or 0.0)
        position = grid_result.get("open_position") or self._safe_tool_call("get_open_position", None, symbol)
        return {
            "symbol": symbol,
            "balance": balance,
            "price": price,
            "indicators": {
                "smc": {"structure": "Grid"},
                "macro": {},
                "trend": {"adx": 0.0},
                "vol": {"atr_pct": 0.0},
                "vwap": {"z_score": 0.0},
                "order_flow": {"delta": 0.0, "cvd": 0.0, "iceberg": "None"},
            },
            "plan": {
                "regime": {"trend": "grid", "volatility": "range", "tradable": can_trade},
                "bias": "Neutral",
                "selected": {
                    "strategy": "grid_long",
                    "confidence": 1.0,
                },
            },
            "decision": {
                "action": "hold",
                "reason": f"Grid active | fills={grid_result.get('fills_applied', 0)}",
            },
            "position": position,
            "recent_trades": self._safe_tool_call("get_recent_trades", [], 5),
            "performance": self._safe_tool_call("get_performance_metrics", {}),
            "today_pnl": self._safe_tool_call("get_todays_realized_pnl", 0.0),
            "unrealized_pnl": self._safe_tool_call("get_unrealized_pnl", 0.0, symbol, price),
            "grid": grid_result,
        }

    async def _run_grid_mode(self):
        self.stats["last_action"] = "Running grid mode..."
        await self._bootstrap_validation()
        self.start_health_api()
        self.streamer.start()
        log.info("Grid streamer started. Warming up local candle buffers...")
        await asyncio.sleep(2)

        try:
            self.cached_balance = self.tools.get_balance()
            self.last_balance_update = time.time()
        except Exception as e:
            log.warning("Initial balance sync failed: %s", e)

        await self._startup_reconcile()

        layout = self.create_layout()
        state = {
            "symbol": self.symbol,
            "balance": 0.0,
            "price": 0.0,
            "indicators": {},
            "plan": {},
            "htf_levels": {},
            "decision": {},
            "position": None,
            "recent_trades": [],
            "performance": {},
            "today_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "grid": {},
        }

        try:
            with Live(layout, refresh_per_second=4, screen=False):
                while True:
                    t0 = time.monotonic()
                    cycle_ok = False
                    try:
                        self.stats["cycles"] += 1
                        if time.time() - self.last_balance_update > 300:
                            try:
                                self.cached_balance = self.tools.get_balance()
                                self.last_balance_update = time.time()
                            except Exception as e:
                                log.warning("Balance fetch failed: %s", e)

                        balance = self.cached_balance
                        symbol_count = max(len(self.symbols), 1)
                        per_symbol_balance = balance / symbol_count
                        day_balance = self.tools.get_day_balance_snapshot()
                        safety = self.guard.evaluate(self.symbols, balance, day_balance)
                        can_trade = self.live_enabled and safety.get("allowed", True)
                        if not can_trade:
                            reason = safety.get("reason", "disabled")
                            self.stats["last_action"] = f"Grid paused: {reason}"
                            if not self.tools.adapter.paper_trading:
                                for symbol in self.symbols:
                                    try:
                                        self.tools.adapter.cancel_all_open_orders(symbol)
                                    except Exception as exc:
                                        log.warning("Could not cancel grid orders for %s: %s", symbol, exc)
                            primary_state = self._grid_dashboard_state(
                                self.symbol,
                                self.grid_states.get(self.symbol, {"price": self.streamer.prices.get(self.symbol, 0.0)}),
                                False,
                                balance,
                            )
                        else:
                            primary_state = None
                            for symbol in self.symbols:
                                price = self.streamer.prices.get(symbol, 0.0)
                                if price <= 0:
                                    ticker = self._safe_tool_call("get_ticker", {}, symbol) or {}
                                    price = float(ticker.get("price", 0.0) or 0.0)
                                if price <= 0:
                                    continue
                                result = self.grid.sync(symbol, price, per_symbol_balance, allow_new_orders=True)
                                self.grid_states[symbol] = result
                                if symbol == self.symbol:
                                    primary_state = self._grid_dashboard_state(symbol, result, True, balance)

                            if primary_state is None:
                                primary_state = self._grid_dashboard_state(
                                    self.symbol,
                                    self.grid_states.get(self.symbol, {"price": self.streamer.prices.get(self.symbol, 0.0)}),
                                    True,
                                    balance,
                                )

                            self.stats["last_action"] = f"Grid synced {len(self.symbols)} symbols"

                        state = primary_state
                        cycle_ok = True
                    except Exception as e:
                        self.stats["last_action"] = f"Error: {str(e)[:40]}"
                        log.exception("Grid loop error")
                        self.guard.record_error(e)

                    self.update_dashboard(layout, state)
                    if cycle_ok:
                        self.guard.record_success()
                    dt = time.monotonic() - t0
                    await asyncio.sleep(max(0.5, self.grid.config.refresh_seconds - dt))
        finally:
            self.stop_health_api()

    async def start(self):
        layout = self.create_layout()
        atr_map = {}
        # Initial state to avoid NameError if loop fails early
        state = {"symbol": self.symbol, "balance": 0.0, "price": 0.0, "indicators": {}, "plan": {}, "htf_levels": {}, "decision": {}, "position": None, "recent_trades": [], "performance": {}, "today_pnl": 0.0, "unrealized_pnl": 0.0}
        validation_task = None
        balance_task = None
        reconcile_task = None

        if self.bot_mode == "grid":
            await self._run_grid_mode()
            return

        if self.paper_replay:
            self.start_health_api()
            await self._run_paper_replay_mode()
            self.stop_health_api()
            return

        if not self.tools.adapter.paper_trading and _env_bool("REQUIRE_VALIDATION_ON_STARTUP", True):
            self.live_enabled = False
            self.stats["last_action"] = "Running startup validation..."
            validation_task = asyncio.create_task(self._bootstrap_validation())
        else:
            await self._bootstrap_validation()

        self.start_health_api()
        self.streamer.start()
        log.info("Elite Streamer Started. Warming up local candle buffers...")
        await asyncio.sleep(2)
        self.stats["last_action"] = "Syncing initial balance..."
        self.last_balance_update = time.time()
        balance_task = asyncio.create_task(asyncio.to_thread(self.tools.get_balance))
        print(
            "\nTradeLite starting\n"
            f"Mode: {self.bot_mode} | Trading: {getattr(self.tools.adapter, 'trading_mode', 'unknown')} | "
            f"Paper: {getattr(self.tools.adapter, 'paper_trading', False)}\n"
            f"Symbol: {self.symbol}\n"
            "Status: Syncing initial balance...\n",
            flush=True,
        )
        self.update_dashboard(layout, state)

        startup_wait_deadline = time.monotonic() + 8
        while balance_task is not None and not balance_task.done() and time.monotonic() < startup_wait_deadline:
            log.info("Waiting for initial balance sync to complete...")
            print("Waiting for initial balance sync to complete...", flush=True)
            await asyncio.sleep(1)

        if balance_task is not None and balance_task.done():
            try:
                self.cached_balance = float(balance_task.result() or 0.0)
                self.last_balance_update = time.time()
                log.info("Initial balance sync complete: %.2f", self.cached_balance)
                if self.tools.adapter.paper_trading and not _env_bool("PAPER_TRADING", True):
                    self.runtime_notice = "Paper fallback activated after balance auth failure"
                    self.stats["last_action"] = self.runtime_notice
            except Exception as exc:
                log.warning("Initial balance sync failed: %s", exc)
            finally:
                balance_task = None
        
        # Start reconciliation only after balance sync settled
        reconcile_task = asyncio.create_task(self._startup_reconcile())

        try:
            with Live(layout, refresh_per_second=4, screen=False):
                waiting_logged = False
                validation_grace_seconds = _env_int("VALIDATION_STARTUP_GRACE_SECONDS", 15)
                while True:
                    t0 = time.monotonic()
                    cycle_ok = False
                    try:
                        self.stats["cycles"] += 1
                        if balance_task is not None and balance_task.done():
                            try:
                                self.cached_balance = float(balance_task.result() or 0.0)
                                self.last_balance_update = time.time()
                                log.info("Initial balance sync complete: %.2f", self.cached_balance)
                                if self.tools.adapter.paper_trading and not _env_bool("PAPER_TRADING", True):
                                    self.runtime_notice = "Paper fallback activated after balance auth failure"
                                    self.stats["last_action"] = self.runtime_notice
                            except Exception as exc:
                                log.warning("Initial balance sync failed: %s", exc)
                            finally:
                                balance_task = None

                        if reconcile_task is not None and reconcile_task.done():
                            try:
                                await reconcile_task
                            except Exception as exc:
                                log.warning("Startup reconciliation failed: %s", exc)
                            finally:
                                reconcile_task = None

                        if validation_task is not None and not self.live_enabled:
                            elapsed = time.monotonic() - self.validation_started_at
                            if elapsed >= validation_grace_seconds:
                                self.live_enabled = True
                                self.runtime_notice = "Startup validation still running; live mode armed after grace period"
                                self.stats["last_action"] = self.runtime_notice
                                log.warning(self.runtime_notice)

                        if balance_task is not None:
                            if not waiting_logged:
                                log.info("Waiting for initial balance sync to complete...")
                                print("Waiting for initial balance sync to complete...", flush=True)
                                waiting_logged = True
                            state["symbol"] = self.symbol
                            state["balance"] = self.cached_balance
                            state["price"] = self.streamer.prices.get(self.symbol, 0.0)
                            state["decision"] = {"action": "hold", "reason": "Waiting for initial balance sync"}
                            state["position"] = self._safe_tool_call("get_open_position", None, self.symbol)
                            state["recent_trades"] = self._safe_tool_call("get_recent_trades", [], 5)
                            state["performance"] = self._safe_tool_call("get_performance_metrics", {})
                            state["today_pnl"] = self._safe_tool_call("get_todays_realized_pnl", 0.0)
                            state["unrealized_pnl"] = self._safe_tool_call("get_unrealized_pnl", 0.0, self.symbol, state["price"])
                            self.stats["last_action"] = "Waiting for initial balance sync..."
                            self.update_dashboard(layout, state)
                            dt = time.monotonic() - t0
                            await asyncio.sleep(max(0.5, 3 - dt))
                            continue

                        # 1. Real-time Balance is handled via callbacks.
                        # We only poll as a fallback if the stream hasn't updated in 5 minutes.
                        if self.cached_balance > 0 and time.time() - self.last_balance_update > 300:
                            try:
                                self.cached_balance = self.tools.get_balance()
                                self.last_balance_update = time.time()
                                log.info(f"Balance updated via REST: {self.cached_balance} USDT")
                                log.debug("Fallback Balance Fetch")
                            except Exception as e:
                                log.warning(f"Balance fetch failed: {e}")

                        balance = self.cached_balance
                        if balance <= 0:
                            self.stats["last_action"] = "Waiting for balance data..."
                            state["symbol"] = self.symbol
                            state["balance"] = balance
                            state["price"] = self.streamer.prices.get(self.symbol, 0.0)
                            state["decision"] = {"action": "hold", "reason": "Balance unavailable"}
                            self.update_dashboard(layout, state)
                            dt = time.monotonic() - t0
                            await asyncio.sleep(max(0.5, 3 - dt))
                            continue

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
                            open_count = len(self._safe_tool_call("get_open_positions", []))
                            max_pos = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

                            if candidates and open_count < max_pos:
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

                    if validation_task is not None and validation_task.done():
                        try:
                            await validation_task
                        except Exception as exc:
                            log.warning("Startup validation failed: %s", exc)
                            self.live_enabled = False
                            self.runtime_notice = f"Startup validation failed: {exc}"
                            self.stats["last_action"] = self.runtime_notice
                        finally:
                            validation_task = None
                    
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
    parser.add_argument("--mode", choices=["live", "grid", "replay", "backtest"], default=os.getenv("BOT_MODE", "live").lower())
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
    elif args.mode == "grid":
        asyncio.run(bot.start())
    elif args.mode == "backtest":
        asyncio.run(bot._run_backtest_mode(
            symbols=[s.strip() for s in (args.symbols or os.getenv("SYMBOLS", bot.symbol)).split(",") if s.strip()],
            timeframe=args.timeframe or os.getenv("TIMEFRAME", "5m"),
            limit=args.limit,
        ))
    else:
        asyncio.run(bot.start())
