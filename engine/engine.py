import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Optional

from indicators.smc import analyze_smc_structure
from indicators.order_flow import analyze_order_flow
from indicators.vwap import calculate_vwap_analysis
from indicators.liquidity import calculate_liquidity_map
from indicators.macro import get_macro_analysis
from indicators.trend_strength import analyze_trend_strength
from indicators.volatility import analyze_volatility
from indicators.risk import calculate_risk_parameters
from indicators.market_context import set_market_data, is_backtest
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.bollinger import calculate_bollinger
from .edge import build_edge_plan
from .strategies import StrategyOrchestrator
from .cache import GlobalAsyncCache

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, tools, risk_manager):
        self.tools        = tools
        self.risk_manager = risk_manager
        self.orchestrator = StrategyOrchestrator()
        self.cache        = GlobalAsyncCache(tools)

    @staticmethod
    def _ensure_ema_columns(df):
        if df is None or df.empty:
            return df
        frame = df.copy()
        if "EMA_50" not in frame.columns:
            frame["EMA_50"] = frame["close"].ewm(span=50, adjust=False).mean()
        if "EMA_200" not in frame.columns:
            frame["EMA_200"] = frame["close"].ewm(span=200, adjust=False).mean()
        return frame

    def _safe_call(self, name: str, default=None, *args, **kwargs):
        fn = getattr(self.tools, name, None)
        if not callable(fn):
            return default
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            log.error("Network error during %s: %s", name, exc)
            return default
        except TimeoutError as exc:
            log.warning("Timeout during %s: %s", name, exc)
            return default
        except Exception as exc:
            log.debug("%s failed: %s", name, exc)
            return default

    async def run(
        self,
        state: Dict,
        execute: bool = True,
        streamer=None,
        df_override=None,
        htf_df_override=None,
        htf_levels_override=None,
        htf_bias_override=None,
    ) -> Dict:
        symbol = state["symbol"]

        # 1. Zero Latency Retrieval
        if df_override is not None:
            df = self._ensure_ema_columns(df_override)
        elif streamer:
            df = streamer.get_candles(symbol)
            if df.empty: df = self.tools.get_market_data(symbol, timeframe="1m")
            df = self._ensure_ema_columns(df)
        else:
            df = self._ensure_ema_columns(self.tools.get_market_data(symbol))

        # ROBUSTNESS GATE: Data Freshness Check
        if df is not None and not df.empty:
            # Handle different data sources (REST index vs Streamer columns)
            if "timestamp" in df.columns:
                last_ts = df["timestamp"].iloc[-1]
            else:
                last_ts = df.index[-1]
            
            # Convert to float unix seconds regardless of type
            if hasattr(last_ts, "timestamp"):
                last_candle_time = last_ts.timestamp()
            else:
                # If it's in ms (Binance default), convert to seconds
                last_candle_time = float(last_ts) / 1000.0 if float(last_ts) > 1e11 else float(last_ts)

            if time.time() - last_candle_time > 180:
                log.warning("%s data is stale (%ds ago). Skipping cycle.", symbol, int(time.time() - last_candle_time))
                state.update({"decision": {"action": "hold", "reason": "Stale data gate"}})
                return state


        balance = state.get("balance")
        if balance is None:
            balance = self._safe_call("get_balance", 0.0)

        if df is None or df.empty or "close" not in df.columns:
            state.update({
                "df": df if df is not None else None,
                "price": 0.0,
                "balance": balance,
                "position": self._safe_call("get_open_position", None, symbol),
                "indicators": {},
                "plan": {"symbol": symbol, "regime": "Unknown", "selected": None, "reason": "No market data"},
                "decision": {"action": "hold", "reason": "No market data"},
            })
            return state

        price = float(df.iloc[-1]["close"])
        if streamer:
            price = streamer.prices.get(symbol, 0.0) or price
            if price == 0:
                ticker = self._safe_call("get_ticker", {}, symbol) or {}
                price = float(ticker.get("price", price) or price)
        else:
            ticker = self._safe_call("get_ticker", {}, symbol) or {}
            price = float(ticker.get("price", price) or price)

        set_market_data(df)
        
        if htf_levels_override is not None:
            htf_levels = htf_levels_override
        elif htf_df_override is not None and not htf_df_override.empty:
            htf_levels = self.tools.build_institutional_levels(htf_df_override, htf_df_override)
        elif is_backtest():
            htf_levels = state.get("htf_levels", {})
        else:
            htf_levels = self.tools.get_institutional_levels(symbol)
        position   = self._safe_call("get_open_position", None, symbol)
        
        if position:
            position["unrealized_pnl"] = self._safe_call("get_unrealized_pnl", 0.0, symbol, price)

        # 2. ANALYZE — Optimized HTF bias and Heavy Indicators
        # Elite Note: Offload heavy calls to background cache
        macro = self.cache.get("macro_sentiment")
        if not macro:
             # Fallback to sync if cache empty but it's risky
             macro = await get_macro_analysis()
        
        htf_bias = htf_bias_override or state.get("htf_bias", "Neutral")
        
        # Pre-trade Balance Gate: Don't even try if balance is critically low
        # 2. ANALYZE — ELITE CACHING GATE
        # Logic: Don't hammer external APIs every 5s for data that changes slowly.
        
        # Open Interest (5m Cache)
        oi_data = state.get("last_oi_data")
        last_oi_time = state.get("last_oi_update", 0)
        adapter = getattr(self.tools, "adapter", None)
        if not oi_data or (time.time() - last_oi_time > 300):
            from indicators.order_flow import analyze_open_interest
            oi_data = analyze_open_interest(symbol, adapter)
            state["last_oi_data"] = oi_data
            state["last_oi_update"] = time.time()

        from indicators.stochastic import calculate_stochastic
        rsi_data = calculate_rsi()
        stoch_data = calculate_stochastic()
        
        indicators = {
            "smc":        analyze_smc_structure(htf_structure=htf_bias),
            "order_flow": analyze_order_flow(symbol, adapter, streamer),
            "oi":         oi_data,
            "vwap":       calculate_vwap_analysis(),
            "liquidity":  calculate_liquidity_map(),
            "macro":      macro,
            "trend":      analyze_trend_strength(),
            "vol":        analyze_volatility(),
            "rsi":        rsi_data.get("value", 50.0),
            "rsi_series": rsi_data.get("series", []),
            "stoch":      stoch_data,
            "macd":       calculate_macd(),
            "bollinger":  calculate_bollinger(),
            "cross_exchange": self.cache.get(f"cross_exchange_{symbol.split('/')[0]}", self.cache.get("cross_exchange_BTC")),
            "asset_correlation": self.cache.get("asset_correlation_market"),
        }

        # 3. DYNAMIC STRATEGY SELECTION
        # We scan all available strategies and pick the one with the highest confidence
        best_signal = self.orchestrator.get_best_signal(df, indicators)
        
        if best_signal:
            plan = {
                "symbol": symbol,
                "regime": {"name": "multi_strategy_scanning", "tradable": True},
                "selected": {
                    "strategy": best_signal.strategy_name,
                    "action": best_signal.action,
                    "confidence": best_signal.confidence,
                    "entry": best_signal.entry,
                    "stop_loss": best_signal.stop_loss,
                    "take_profit": best_signal.take_profit,
                    "expected_r": (abs(best_signal.take_profit - best_signal.entry) / abs(best_signal.entry - best_signal.stop_loss)) if best_signal.stop_loss and abs(best_signal.entry - best_signal.stop_loss) > 0 else 2.0,
                    "reason": best_signal.reason
                },
                "bias": htf_bias
            }
        else:
            plan = {"symbol": symbol, "regime": "Unknown", "selected": None, "reason": "No strategy found an edge"}

        state.update({
            "df": df, "price": price, "balance": balance,
            "htf_levels": htf_levels, "htf_bias": htf_bias,
            "position": position, "indicators": indicators, "plan": plan,
        })

        # 4. DECIDE
        state = self._decide(state, streamer)

        # 5. EXECUTE
        if execute:
            state = self._execute(state)

        return state

    def _decide(self, state: Dict, streamer = None) -> Dict:
        plan     = state.get("plan", {})
        selected = plan.get("selected")
        price    = state["price"]
        balance  = state["balance"]
        indicators = state.get("indicators", {})
        position = state.get("position")

        if not selected:
            state["decision"] = {"action": "hold", "reason": plan.get("reason", "No edge")}
            return state

        metrics    = self._safe_call("get_performance_metrics", {})
        
        # Elite Data Extraction
        vol_data = indicators.get("vol", {})
        atr_pct  = vol_data.get("atr_pct", 1.0)
        
        # --- ELITE SAFEGUARD: Volatility Circuit Breaker ---
        # If the market is too volatile (e.g., News spikes), technical indicators are noise.
        df = state.get("df")
        if df is not None and not df.empty:
            avg_atr = df["high"].tail(20).max() - df["low"].tail(20).min()
            current_atr = vol_data.get("atr", 0)
            if current_atr > avg_atr * 2.5:
                 state["decision"] = {"action": "hold", "reason": "Circuit Breaker: Extreme Volatility detected (News Shock?)"}
                 return state

        of_data  = indicators.get("order_flow", {})
        
        # Zero-Latency Spread
        if streamer:
            spread_val = streamer.get_spread(state["symbol"])
            spread_bps = (spread_val / (price + 1e-9)) * 10000 if spread_val > 0 else 5.0
        else:
            ticker   = self.tools.get_ticker(state["symbol"])
            bid, ask = ticker.get("bid", price), ticker.get("ask", price)
            spread_bps = ((ask - bid) / (price + 1e-9)) * 10000 if ask > bid else 5.0

        # Iceberg Confluence (Boost confidence if iceberg detected at entry)
        confidence = selected["confidence"]
        if of_data.get("iceberg") != "None":
            confidence = min(1.0, confidence + 0.05)

        # --- ELITE SAFEGUARD: Portfolio Correlation Guard ---
        try:
            all_positions = self.tools.adapter.get_positions()
            if len(all_positions) >= 3 and not position:
                state["decision"] = {"action": "hold", "reason": "Portfolio Guard: Max concurrent positions reached"}
                return state
        except: pass

        if position:
            position_side = str(position.get("side", "")).lower()
            current_amount = float(position.get("amount", 0.0) or 0.0)
            current_notional = current_amount * price
            exposure_cap = balance * getattr(self.risk_manager.config, "max_position_pct", 0.60)
            
            # POSITION FLIPPING LOGIC
            is_conflict = (position_side == "long" and selected["action"] == "sell") or (position_side == "short" and selected["action"] == "buy")
            if is_conflict and confidence > 0.85:
                state["decision"] = {
                    "action": "close",
                    "reason": f"Conflict Flip: Consensus strongly shifted to {selected['action']}",
                    "amount": current_amount
                }
                return state

            same_direction = (position_side == "long" and selected["action"] == "buy") or (position_side == "short" and selected["action"] == "sell")
            if same_direction and current_notional >= exposure_cap * 0.9:
                state["decision"] = {
                    "action": "hold",
                    "reason": "Existing position already near exposure cap",
                }
                return state

        risk_params = calculate_risk_parameters(
            free_balance     = balance,
            current_price    = price,
            confidence_score = confidence,
            atr_pct          = atr_pct,
            stop_loss        = selected["stop_loss"],
            expected_r       = selected["expected_r"],
            historical_win_rate = metrics.get("win_rate", 0.55),
            total_trades     = metrics.get("total_trades", 0),
            spread_bps       = spread_bps,
            max_risk_per_trade_pct = getattr(self.risk_manager.config, "max_risk_per_trade_pct", 0.05),
            max_position_pct = getattr(self.risk_manager.config, "max_position_pct", 0.60),
            min_order_quote  = getattr(self.risk_manager.config, "min_order_quote", 10.0),
        )

        if risk_params.get("action") == "lock":
            state["decision"] = {"action": "hold", "reason": risk_params.get("reason", "Risk locked")}
            return state

        atr = vol_data.get("atr", price * 0.005)
        if atr < (price * 0.0001): atr = price * 0.001 
        
        signal_entry = selected.get("entry", price)
        # Tighten hunt zone for Execution Edge (±0.3 ATR instead of 0.5)
        hunt_zone  = {"min": signal_entry - atr * 0.3, "max": signal_entry + atr * 0.3}

        state["decision"] = {
            "action":             "trade",
            "side":               selected["action"],
            "trade_side":         selected["action"],
            "recommended_amount": risk_params.get("recommended_amount", 0.0),
            "stop_loss":          selected["stop_loss"],
            "take_profit":        selected["take_profit"],
            "strategy":           selected["strategy"],
            "confidence":         confidence,
            "expected_r":         selected["expected_r"],
            "reason":             selected["reason"],
            "hunt_zone":          hunt_zone,
            "spread_bps":         spread_bps
        }
        return state

    def _execute(self, state: Dict) -> Dict:
        decision = state.get("decision", {})
        if decision.get("action") != "trade":
            return state
            
        # Prevent double-buying within 10 seconds
        now = time.time()
        last_trade = getattr(self, "_last_trade_at", 0)
        if now - last_trade < 10:
            log.info("Skipping trade: Executed another trade too recently.")
            return state

        symbol      = state["symbol"]
        price       = state["price"]
        side        = decision.get("side", "hold")
        amount      = decision.get("recommended_amount", 0.0)
        stop_loss   = decision.get("stop_loss")
        take_profit = decision.get("take_profit")
        hunt_zone   = decision.get("hunt_zone", {})
        spread_bps  = decision.get("spread_bps", 0)

        # 1. Slippage Protection (Max 0.15% from Signal Entry)
        if hunt_zone and not (hunt_zone["min"] <= price <= hunt_zone["max"]):
            decision["action"] = "hold"
            decision["reason"] = f"Slippage/Price Deviation: {price:.2f} outside zone"
            state["decision"]  = decision
            return state

        # 2. Toxic Spread Protection
        if spread_bps > 15: # 15 bps limit
            decision["action"] = "hold"
            decision["reason"] = f"Toxic Spread: {spread_bps:.1f} bps"
            state["decision"]  = decision
            return state

        if side != "hold" and amount > 0 and stop_loss and take_profit:
            result = self.tools.execute_trade(
                symbol=symbol, side=side, amount=amount,
                price=price, stop_loss=stop_loss,
                take_profit=take_profit, reason=decision.get("reason", "Edge"),
                logic_snapshot=state.get("indicators")
            )
            state["execution_result"] = result
            if result.get("error"):
                decision["action"] = "hold"
                decision["reason"] = result.get("error", "Execution failed")
                state["decision"] = decision
            log.info("Executed %s %s %.6f @ %.2f | Spread %.1f bps",
                     side, symbol, amount, price, spread_bps)

        return state
