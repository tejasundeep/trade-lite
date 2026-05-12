import time
import os
import logging
from datetime import datetime
import pytz
from .smc import analyze_smc_structure
from .order_flow import analyze_order_flow

log = logging.getLogger(__name__)

class MasterInstitutionalStrategy:
    def __init__(self, streamer=None):
        self.streamer = streamer

    def is_silver_bullet_window(self) -> bool:
        """Checks if current time is within ICT Silver Bullet windows (NY Time)."""
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        hour = now_ny.hour
        
        # 03:00 - 04:00 (London Open)
        # 10:00 - 11:00 (NY AM)
        # 14:00 - 15:00 (NY PM)
        return hour in [3, 10, 14]

    def get_decision(self, symbol: str, df) -> dict:
        """
        Combines SMC, Order Flow, and Time to produce a high-probability trade decision.
        """
        # 1. SMC Analysis (Sweeps, FVGs, Structure)
        smc = analyze_smc_structure(df_override=df)
        if "error" in smc:
            return {"action": "wait", "reason": f"SMC Error: {smc['error']}"}

        # 2. Order Flow Analysis (CVD, Delta, Absorption)
        of = analyze_order_flow(symbol=symbol, streamer=self.streamer)
        
        # 3. Time Context
        is_silver_bullet = self.is_silver_bullet_window()
        
        # --- STRATEGY 1: INSTITUTIONAL TRAP (Sweep + FVG) ---
        sweep = smc.get("liquidity_sweep", {})
        fvgs = smc.get("fvgs", [])
        
        # Bullish Trap: Low Sweep + Bullish FVG
        if sweep.get("low") and any(f["type"] == "bullish" for f in fvgs):
            confidence = 0.85 if is_silver_bullet else 0.70
            return {
                "action": "buy",
                "reason": "Institutional Trap: Low Sweep + FVG",
                "confidence": confidence,
                "indicators": {"smc": smc, "of": of},
                "setup": "trap"
            }
            
        # Bearish Trap: High Sweep + Bearish FVG
        if sweep.get("high") and any(f["type"] == "bearish" for f in fvgs):
            confidence = 0.85 if is_silver_bullet else 0.70
            return {
                "action": "sell",
                "reason": "Institutional Trap: High Sweep + FVG",
                "confidence": confidence,
                "indicators": {"smc": smc, "of": of},
                "setup": "trap"
            }

        # --- STRATEGY 2: CVD ABSORPTION ---
        abs_bias = of.get("absorption", "None")
        if abs_bias == "Bullish Absorption" and smc["zone"] == "Discount":
            return {
                "action": "buy",
                "reason": "Order Flow: Bullish Absorption in Discount Zone",
                "confidence": 0.80,
                "indicators": {"smc": smc, "of": of},
                "setup": "absorption"
            }
        elif abs_bias == "Bearish Absorption" and smc["zone"] == "Premium":
            return {
                "action": "sell",
                "reason": "Order Flow: Bearish Absorption in Premium Zone",
                "confidence": 0.80,
                "indicators": {"smc": smc, "of": of},
                "setup": "absorption"
            }

        # --- STRATEGY 3: SILVER BULLET MOMENTUM ---
        if is_silver_bullet and smc["structure"] != "Neutral":
            # Just follow the structure in the bullet window
            action = "buy" if smc["structure"] == "Bullish" else "sell"
            return {
                "action": action,
                "reason": f"Silver Bullet: {smc['structure']} momentum window",
                "confidence": 0.75,
                "indicators": {"smc": smc, "of": of},
                "setup": "silver_bullet"
            }

        return {"action": "wait", "reason": "No high-probability setup found"}
