import os
import json
import logging
from typing import List, Dict
from sqlalchemy.orm import Session
from db import Trade, Position, Session as DBSession
from openai import OpenAI

log = logging.getLogger(__name__)

class TradingChatbot:
    def __init__(self, tools=None, streamer=None, bot=None):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.tools = tools
        self.streamer = streamer
        self.bot = bot
        self.messages = [] # History for memory
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=20.0)

    def get_context(self) -> str:
        session = DBSession()
        try:
            # 1. Live Balance
            live_balance = 0.0
            if self.bot and hasattr(self.bot, "cached_balance"):
                live_balance = self.bot.cached_balance
            elif self.tools:
                try: live_balance = self.tools.get_balance()
                except: pass
            
            # 2. Market Prices
            prices = self.streamer.prices if self.streamer else {}

            # 3. System Health & Safety
            guard_status = "OK"
            guard_reason = ""
            if self.bot and self.bot.guard:
                if self.bot.guard.tripped:
                    guard_status = "TRIPPED (Trading Halted)"
                    guard_reason = self.bot.guard.trip_reason

            # 4. Active Analysis (All Strategies + Consensus)
            analysis_summary = ""
            if self.bot and hasattr(self.bot, "symbol_states"):
                analysis_summary = "\nCURRENT MARKET ANALYSIS (Multi-Strategy Consensus):\n"
                for s, state in self.bot.symbol_states.items():
                    inds = state.get("indicators", {})
                    smc = inds.get("smc", {})
                    of = inds.get("order_flow", {})
                    vol = inds.get("vol", {})
                    trend = inds.get("trend", {})
                    vwap = inds.get("vwap", {})
                    bb = inds.get("bollinger", {})
                    plan = state.get("plan", {})
                    decision = state.get("decision", {})
                    consensus = plan.get("consensus", {})
                    selected = plan.get("selected", {})
                    
                    analysis_summary += f"- {s}: Price {state.get('price')}.\n"
                    analysis_summary += f"  Indicators: RSI={inds.get('rsi', 'N/A')}, ADX={trend.get('adx', 'N/A')}, ATR%={vol.get('atr_pct', 'N/A')}, VWAP Z={vwap.get('z_score', 'N/A')}\n"
                    analysis_summary += f"  SMC: Structure={smc.get('structure', 'N/A')}, Zone={smc.get('zone', 'N/A')}\n"
                    analysis_summary += f"  Order Flow: Delta={of.get('delta', 'N/A')}, CVD={of.get('cvd', 'N/A')}, Absorption={of.get('absorption', 'N/A')}\n"
                    analysis_summary += f"  Bollinger: Upper={bb.get('upper', 'N/A')}, Lower={bb.get('lower', 'N/A')}\n"
                    
                    if consensus:
                        analysis_summary += f"  CONSENSUS: Buy Score={consensus.get('buy_score', 0)}, Sell Score={consensus.get('sell_score', 0)}, Agreement={consensus.get('agreement_ratio', 0):.0%}\n"
                        analysis_summary += f"  Agreeing: {consensus.get('agreeing_strategies', [])}\n"
                        analysis_summary += f"  Opposing: {consensus.get('opposing_strategies', [])}\n"
                    
                    if selected:
                        analysis_summary += f"  Selected Strategy: {selected.get('strategy', 'None')} (conf={selected.get('confidence', 0):.2%})\n"
                    
                    if decision:
                        analysis_summary += f"  Decision: {decision.get('action', 'hold')} - {decision.get('reason', 'N/A')}\n"

            # 5. DB Positions & Trades
            log.debug("Fetching positions from DB...")
            positions = session.query(Position).all()
            
            log.debug("Fetching recent trades from DB...")
            trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(5).all()
            
            context = f"SYSTEM STATUS:\n"
            context += f"- Guard Status: {guard_status} {guard_reason}\n"
            context += f"- Live Balance: {live_balance:.2f} USDT\n"
            
            total_invested = 0.0
            context += "\nACTIVE POSITIONS:\n"
            if not positions:
                context += "- No open positions.\n"
            for pos in positions:
                cur_price = prices.get(pos.symbol, pos.avg_price)
                notional = pos.amount * cur_price
                total_invested += notional
                upnl = (cur_price - pos.avg_price) * pos.amount * (1 if pos.side == "long" else -1)
                context += f"- {pos.side.upper()} {pos.symbol}: Qty {pos.amount} @ {pos.avg_price}. CurPrice: {cur_price}. Notional: {notional:.2f} USDT. UnPnL: {upnl:.2f} USDT.\n"
            
            context += f"\nTOTAL EXPOSURE: {total_invested:.2f} USDT\n"
            context += analysis_summary
            
            context += "\nRECENT TRADES:\n"
            for t in trades:
                context += f"- {t.timestamp}: {t.side.upper()} {t.symbol} at {t.price}. Result: {t.pnl or 'N/A'}. Reason: {t.reason}\n"
            
            log.debug("Context generation complete.")
            return context
        finally:
            session.close()

    def ask(self, question: str) -> str:
        if not self.client:
            return "DeepSeek API Key not found in .env."
        
        log.info(f"AI Assistant: Processing query: {question}")

        # --- REAL-TIME SCENARIOS & EDGE CASES (USER COMMANDS) ---
        q = question.lower()
        
        # 1. Panic / Emergency Stop
        if "emergency stop" in q or "pause trading" in q or "halt" in q:
            if self.bot and self.bot.guard:
                self.bot.guard.trip("User requested emergency stop", cooldown_minutes=1440)
                return "🚨 **EMERGENCY STOP EXECUTED.** I have tripped the circuit breaker and halted all autonomous trading. I will not enter any new trades until you tell me to 'resume'."

        # 2. Resuming / Overriding Safety
        if "resume" in q or "reset safety" in q or "start trading" in q:
            if self.bot and self.bot.guard:
                self.bot.guard.reset()
                return "✅ **TRADING RESUMED.** I have reset the circuit breakers. The autonomous engine is now re-scanning the markets using all strategies (consensus scoring). Let's get back to work."

        # 3. Manual Liquidation (Two-step verification)
        if "close all" in q and ("position" in q or "trade" in q):
             if "confirm" in q:
                 if self.tools:
                     res = self.tools.close_all_positions("User confirmed AI Close All")
                     return f"🔥 **LIQUIDATION COMPLETE.** {res['message']} I have exited all market exposure as requested."
             return "⚠️ **CONFIRMATION REQUIRED.** Do you really want to close ALL active positions at market price? Type '**confirm close all**' to proceed."

        # 4. Data/Spread Check Edge Case
        if "spread" in q or "slippage" in q:
            context = self.get_context()
            return f"Current market spreads are monitored. If the spread exceeds 15bps, I automatically block entries to protect your capital. {context.split('SYSTEM STATUS:')[1].split('ACTIVE POSITIONS:')[0]}"

        context = self.get_context()
        
        system_prompt = f"""
        You are the 'TradeX Pro' AI Trading Assistant.
        Your mission is to be the 'Perfect Assistant' between the User and the Autonomous Multi-Strategy Consensus Engine.
        
        Current System Context:
        {context}
        
        CRITICAL: This bot uses a MULTI-STRATEGY CONSENSUS ENGINE with 15 strategies.
        Each strategy has a weight, and the bot trades based on the COMBINED weighted score of ALL strategies.
        SMC is just ONE of 15 strategies. Others include: Momentum Breakout, VWAP Mean Reversion,
        Range Mean Reversion (Bollinger/RSI), Trend Scalper (EMA), Tape Reading (Order Flow),
        CVD Absorption, Liquidation Hunter, AMD (Power of Three), Coinbase Premium, Leader-Laggard,
        Institutional Climax, Elite Structural Breakout, and Silver Bullet.
        
        When explaining trades or decisions:
        - Reference the ACTUAL strategies that fired (shown in consensus data), not just SMC.
        - Explain the consensus scoring: buy score vs sell score, agreement ratio, conflict penalties.
        - If multiple strategies agree, highlight the CONFLUENCE.
        - If the bot is holding, explain WHICH strategies disagreed or why consensus was weak.
        - Reference actual indicator values (RSI, ADX, VWAP Z-score, Bollinger position, etc.)
        
        AI MISSION:
        1. Explain which strategies contributed to each trade decision and why.
        2. Handle 'Why no trades?' by showing the consensus scorecard (weak agreement, conflicting strategies, etc.)
        3. Be the user's hands: If they want to stop, suggest the 'emergency stop' command.
        4. Maintain a professional, data-driven, analytical tone.
        5. NEVER attribute all decisions to SMC alone. Always reference the full multi-strategy picture.
        """
        
        # Maintain history
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": system_prompt})
        else:
            self.messages[0] = {"role": "system", "content": system_prompt} 

        self.messages.append({"role": "user", "content": question})
        if len(self.messages) > 15: # Slightly larger memory
            self.messages = [self.messages[0]] + self.messages[-14:]

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=self.messages,
                stream=False
            )
            answer = response.choices[0].message.content
            log.info(f"AI Assistant: Response received. Length: {len(answer) if answer else 0}")
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            log.error(f"DeepSeek API Error: {e}")
            return f"Error communicating with DeepSeek: {str(e)}"
