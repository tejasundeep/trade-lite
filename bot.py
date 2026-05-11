import os
import asyncio
import logging
import time
from typing import Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box

from engine.tools import TradingTools
from engine.engine import TradingEngine
from trading.risk import MarketRiskManager, MarketRiskConfig
from db import init_db

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

class TradeXProClone:
    def __init__(self):
        key = os.getenv("BINANCE_API_KEY", "").strip()
        log.info(f"Initializing with API Key: {key[:4]}...{key[-4:] if len(key)>4 else ''}")
        
        self.tools = TradingTools(
            api_key=key,
            secret=os.getenv("BINANCE_SECRET", "").strip(),
            paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
            exchange_id=os.getenv("EXCHANGE_ID", "binance").strip()
        )
        self.risk = MarketRiskManager(MarketRiskConfig(
            max_risk_per_trade_pct=float(os.getenv("MAX_RISK_PER_TRADE_PCT", 0.015)),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", 0.05)),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", 45))
        ))
        self.engine = TradingEngine(self.tools, self.risk)
        self.symbol = os.getenv("SYMBOL", "BTC/USDT")
        self.symbols = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]
        self.streamer = BinanceStreamer(self.symbols, adapter=self.tools.adapter)
        self.streamer.add_account_callback(self._handle_account_update)
        self.stats = {"cycles": 0, "last_action": "Initializing..."}
        
        # Persistent per-symbol states (caches HTF bias, etc.)
        self.symbol_states = {s: {"symbol": s, "balance": 0.0, "price": 0.0, "plan": {}, "indicators": {}} for s in self.symbols}
        self.cached_balance = 0.0
        self.last_balance_update = 0
        
        self._trade_lock = asyncio.Lock()
        init_db()

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

    def update_dashboard(self, layout: Layout, state: dict):
        inds, plan, htf = state.get("indicators", {}), state.get("plan", {}), state.get("htf_levels", {})
        smc, macro, trend, vol, vwap, of = inds.get("smc", {}), inds.get("macro", {}), inds.get("trend", {}), inds.get("vol", {}), inds.get("vwap", {}), inds.get("order_flow", {})
        price, balance, decision = state.get("price", 0), state.get("balance", 0), state.get("decision", {})
        mode = "[bold red]LIVE[/]" if os.getenv("PAPER_TRADING", "true").lower() == "false" else "[bold yellow]PAPER[/]"

        layout["header"].update(Panel(f"[bold cyan]TradeX Pro 2.0 (Elite)[/] | [white]{state.get('symbol')}[/] | {mode} | [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/] | Cycle #{self.stats['cycles']}", style="blue"))
        
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
        if event in ["ACCOUNT_UPDATE", "outboundAccountPosition"]:
            for b in update.get("balances", []):
                if b["asset"] == "USDT":
                    self.cached_balance = b["free"]
                    self.last_balance_update = time.time()
                    log.info(f"Real-time Balance Update: {self.cached_balance} USDT")
        elif event in ["ORDER_TRADE_UPDATE", "executionReport"]:
            order = update.get("order", {})
            log.info(f"Real-time Order Update: {order['symbol']} {order['side']} {order['status']} at {order['price']}")
            # In a full production system, we'd sync the local DB state here.
            # For now, we log it and let the next cycle pick up changes if any.

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
        
        self.streamer.start()
        log.info("Elite Streamer Started. Warming up local candle buffers...")
        await asyncio.sleep(10)

        with Live(layout, refresh_per_second=4, screen=False):
            while True:
                t0 = time.monotonic()
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
                    
                    # 2. Position Management
                    async with self._trade_lock:
                        self.stats["last_action"] = "Managing positions & Trail..."
                        # Safe extraction of SMC indicators
                        smc_map = {s: self.symbol_states.get(s, {}).get("indicators", {}).get("smc", {}) for s in self.symbols}
                        self.tools.manage_open_positions(price_map, atr_map, smc_map) 
                        
                    # 3. Scanning Symbols
                    self.stats["last_action"] = f"Scanning {len(self.symbols)} symbols..."
                    results = await asyncio.gather(*[self.scan_symbol(s, balance) for s in self.symbols])
                    candidates = self._correlation_ok([r for r in results if r is not None])

                    # Update ATR Map
                    for r in results:
                        if r: atr_map[r["symbol"]] = r["indicators"].get("vol", {}).get("atr", 0)

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

                    # 5. Enrichment for Dashboard
                    state["price"] = price_map.get(state["symbol"], state.get("price", 0))
                    state["balance"] = balance
                    state["position"] = self.tools.get_open_position(state["symbol"])
                    state["recent_trades"] = self.tools.get_recent_trades(5)
                    state["performance"] = self.tools.get_performance_metrics()
                    state["today_pnl"] = self.tools.get_todays_realized_pnl()
                    state["unrealized_pnl"] = self.tools.get_unrealized_pnl(state["symbol"], state["price"])
                    
                    self.stats["last_action"] = "Elite Engine Idle"
                except Exception as e:
                    self.stats["last_action"] = f"Error: {str(e)[:40]}"
                    log.exception("Loop error")
                
                # Ensure dashboard updates even if logic fails
                self.update_dashboard(layout, state)
                dt = time.monotonic() - t0
                await asyncio.sleep(max(0.5, 3 - dt))

if __name__ == "__main__":
    asyncio.run(TradeXProClone().start())
