import sys, os
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
from indicators.market_context import set_market_data
from .edge import build_edge_plan

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, tools, risk_manager):
        self.tools        = tools
        self.risk_manager = risk_manager

    async def run(self, state: Dict, execute: bool = True, streamer = None, df_override = None, htf_df_override = None) -> Dict:
        symbol = state["symbol"]

        # 1. RETRIEVE - Zero Latency Priority
        if df_override is not None:
            df = df_override
            price = float(df.iloc[-1]["close"])
        elif streamer:
            df = streamer.get_candles(symbol)
            if df.empty: df = self.tools.get_market_data(symbol)
            price = streamer.prices.get(symbol, 0.0)
            if price == 0: 
                ticker = self.tools.get_ticker(symbol)
                price = ticker["price"]
        else:
            df = self.tools.get_market_data(symbol)
            ticker = self.tools.get_ticker(symbol)
            price = ticker["price"]

        set_market_data(df)
        
        balance = state.get("balance") or self.tools.get_balance()
        htf_levels = self.tools.get_institutional_levels(symbol)
        position   = self.tools.get_open_position(symbol)
        
        if position:
            position["unrealized_pnl"] = self.tools.get_unrealized_pnl(symbol, price)

        # 2. ANALYZE — Optimized HTF bias
        # Elite Note: HTF structure (1h) doesn't change every 5 seconds. 
        # We cache it in state to avoid REST call spam.
        htf_bias = state.get("htf_bias", "Neutral")
        last_htf_update = state.get("last_htf_update", 0)
        
        # Only update HTF bias every 15 minutes or if missing
        import time
        if htf_bias == "Neutral" or (time.time() - last_htf_update > 900) or htf_df_override is not None:
            try:
                if htf_df_override is not None:
                    df_htf = htf_df_override
                else:
                    df_htf = self.tools.get_market_data(symbol, timeframe="1h", limit=200)
                
                set_market_data(df_htf)
                htf_smc  = analyze_smc_structure()
                htf_bias = htf_smc.get("structure", "Neutral")
                state["last_htf_update"] = time.time()
                set_market_data(df)   # restore LTF
            except Exception as e:
                log.warning("HTF fetch failed: %s", e)

        macro = await get_macro_analysis()

        indicators = {
            "smc":        analyze_smc_structure(htf_structure=htf_bias),
            "order_flow": analyze_order_flow(symbol, self.tools.adapter, streamer),
            "vwap":       calculate_vwap_analysis(),
            "liquidity":  calculate_liquidity_map(),
            "macro":      macro,
            "trend":      analyze_trend_strength(),
            "vol":        analyze_volatility(),
        }

        # 3. EDGE PLAN
        plan = build_edge_plan(df, indicators, symbol, htf_levels)

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

        if not selected:
            state["decision"] = {"action": "hold", "reason": plan.get("reason", "No edge")}
            return state

        metrics    = self.tools.get_performance_metrics()
        
        # Elite Data Extraction
        vol_data = indicators.get("vol", {})
        atr_pct  = vol_data.get("atr_pct", 1.0)
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

        risk_params = calculate_risk_parameters(
            free_balance     = balance,
            current_price    = price,
            confidence_score = confidence,
            atr_pct          = atr_pct,
            stop_loss        = selected["stop_loss"],
            expected_r       = selected["expected_r"],
            historical_win_rate = metrics.get("win_rate", 0.55),
            total_trades     = metrics.get("total_trades", 0),
            spread_bps       = spread_bps
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
            )
            state["execution_result"] = result
            log.info("Executed %s %s %.6f @ %.2f | Spread %.1f bps",
                     side, symbol, amount, price, spread_bps)

        return state
