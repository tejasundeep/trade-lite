import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketRegime:
    name:       str
    trend:      str
    volatility: str
    tradable:   bool
    reasons:    List[str]
    metrics:    Dict[str, float]


@dataclass(frozen=True)
class StrategySignal:
    strategy:    str
    action:      str
    confidence:  float
    entry:       float
    stop_loss:   Optional[float]
    take_profit: Optional[float]
    expected_r:  float
    reason:      str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _structural_sl(df: pd.DataFrame, action: str, smc: dict, atr: float) -> float:
    """SL below/above nearest structural pivot with a 0.5 ATR buffer."""
    pd_arr = smc.get("pd_array", {})
    if action == "buy":
        level = pd_arr.get("low", float(df["low"].tail(20).min()))
        return level - atr * 0.5
    level = pd_arr.get("high", float(df["high"].tail(20).max()))
    return level + atr * 0.5


def _compute_confidence(base: float, action: str, indicators: Dict) -> float:
    """Dynamic confidence built from real indicator alignment."""
    score = base
    smc        = indicators.get("smc", {})
    order_flow = indicators.get("order_flow", {})
    trend      = indicators.get("trend", {})
    macro      = indicators.get("macro", {})
    vol        = indicators.get("vol", {})

    # HTF structure alignment
    if smc.get("mtf_aligned"):
        score += 0.05

    # Order-flow absorption confirmation
    abs_str = order_flow.get("absorption", "None")
    if (action == "buy" and "Bullish" in abs_str) or (action == "sell" and "Bearish" in abs_str):
        score += 0.05

    # Trend strength
    if trend.get("adx", 0) > 25:
        score += 0.03

    # Contrarian macro sentiment (Fear < 30 → buy edge; Greed > 70 → sell edge)
    s_score = macro.get("sentiment", {}).get("score", 50)
    if (action == "buy" and s_score < 30) or (action == "sell" and s_score > 70):
        score += 0.03

    # Volatility penalty
    if vol.get("atr_pct", 2.0) > 4.0:
        score -= 0.05

    return round(min(max(score, 0.0), 1.0), 3)


def _safe_r(tp: float, entry: float, sl: float) -> float:
    risk = abs(entry - sl)
    return abs(tp - entry) / risk if risk > 1e-9 else 0.0


# ─── Regime Detection ────────────────────────────────────────────────────────

def detect_market_regime(df: pd.DataFrame, indicators: Dict) -> MarketRegime:
    latest   = df.iloc[-1]
    price    = float(latest["close"])
    ema50    = float(latest.get("EMA_50",  0))
    ema200   = float(latest.get("EMA_200", 0))
    atr_pct  = indicators.get("vol", {}).get("atr_pct", 0) / 100
    adx      = indicators.get("trend", {}).get("adx", 0)

    # Elite Regime: Volatility-Adjusted Trend
    # Expansion requires ADX > 25, otherwise it's just "Drifting"
    if price > ema50 > ema200:
        trend = "bullish_expansion" if adx > 25 else "bullish_drift"
    elif price < ema50 < ema200:
        trend = "bearish_expansion" if adx > 25 else "bearish_drift"
    else:
        trend = "chop"

    volatility = "extreme" if atr_pct > 0.04 else "compression" if atr_pct < 0.01 else "normal"
    
    # Non-tradable if in extreme volatility (risk too high) or chop with zero strength
    tradable   = not (volatility == "extreme" or (adx < 15 and trend == "chop"))

    return MarketRegime(f"{trend}_{volatility}", trend, volatility, tradable, [], {"adx": adx, "atr_pct": atr_pct})


# ─── Signal Generators ───────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, indicators: Dict, regime: MarketRegime, htf: Dict) -> List[StrategySignal]:
    signals  = []
    smc      = indicators.get("smc", {})
    of       = indicators.get("order_flow", {})
    vol      = indicators.get("vol", {})
    atr      = vol.get("atr", float(df["close"].iloc[-1]) * 0.005)
    entry    = float(df["close"].iloc[-1])
    
    # ── 1. Institutional OB Mitigation (High Confidence) ───────────────
    # Price returns to a fresh Order Block with Imbalance confirmation
    obs = smc.get("order_blocks", [])
    for ob in obs:
        if ob["type"] == "bullish" and ob["bottom"] <= entry <= ob["top"]:
            if "Bullish" in of.get("imbalance", "") or "Bullish" in of.get("absorption", ""):
                sl = ob["bottom"] - atr * 0.2
                tp = smc.get("pd_array", {}).get("high", entry + atr * 4)
                r  = _safe_r(tp, entry, sl)
                if r >= 2.0:
                    conf = _compute_confidence(0.82, "buy", indicators)
                    signals.append(StrategySignal("ob_mitigation_elite", "buy", conf, entry, sl, tp, r, "OB Mitigation + Imbalance"))
        elif ob["type"] == "bearish" and ob["bottom"] <= entry <= ob["top"]:
            if "Bearish" in of.get("imbalance", "") or "Bearish" in of.get("absorption", ""):
                sl = ob["top"] + atr * 0.2
                tp = smc.get("pd_array", {}).get("low", entry - atr * 4)
                r  = _safe_r(tp, entry, sl)
                if r >= 2.0:
                    conf = _compute_confidence(0.82, "sell", indicators)
                    signals.append(StrategySignal("ob_mitigation_elite", "sell", conf, entry, sl, tp, r, "OB Mitigation + Imbalance"))

    # ── 2. FVG Imbalance Sniper ───────────────
    fvgs = smc.get("fvgs", [])
    for fvg in fvgs:
        if fvg["type"] == "bullish" and fvg["bottom"] <= entry <= fvg["top"]:
            if of.get("delta", 0) > 0:
                sl = fvg["bottom"] - atr * 0.3
                tp = entry + atr * 3
                r  = _safe_r(tp, entry, sl)
                if r >= 1.5:
                    signals.append(StrategySignal("fvg_imbalance_sniper", "buy", 0.75, entry, sl, tp, r, "Bullish FVG Entry"))

    # ── 3. Enhanced SFP Sniper ───────────────
    pd_arr = smc.get("pd_array", {})
    if smc.get("liquidity_sweep", {}).get("low") and smc.get("zone") == "Discount":
        if of.get("delta", 0) > 0 or "Bullish" in of.get("absorption", ""):
            sl = entry - atr * 0.8
            tp = pd_arr.get("high", entry + atr * 4)
            r  = _safe_r(tp, entry, sl)
            if r >= 1.8:
                signals.append(StrategySignal("sfp_sniper_v2", "buy", 0.78, entry, sl, tp, r, "SFP + OF Delta Flip"))

    return signals

    return signals


# ─── Edge Plan Builder ───────────────────────────────────────────────────────

def build_edge_plan(df: pd.DataFrame, indicators: Dict, symbol: str, htf_levels: Dict = None) -> Dict:
    if df is None or len(df) < 50:
        return {"symbol": symbol, "regime": "Unknown", "selected": None, "reason": "Insufficient data"}

    regime = detect_market_regime(df, indicators)
    htf    = htf_levels or {}
    price  = float(df.iloc[-1]["close"])

    # Institutional bias from weekly open
    bias = "Neutral"
    if htf.get("weekly_open"):
        dist_pct = (price - htf["weekly_open"]) / htf["weekly_open"]
        if   dist_pct >  0.002: bias = "Bullish"
        elif dist_pct < -0.002: bias = "Bearish"

    candidates = generate_signals(df, indicators, regime, htf)
    eligible   = [c for c in candidates if c.action != "hold"]

    # Filter by institutional bias (Elite Discipline)
    if bias == "Bullish": eligible = [c for c in eligible if c.action == "buy"]
    elif bias == "Bearish": eligible = [c for c in eligible if c.action == "sell"]

    if not regime.tradable or not eligible:
        reason = "Non-tradable Regime" if not regime.tradable else f"No edge found (Bias: {bias})"
        return {"symbol": symbol, "regime": asdict(regime), "selected": None,
                "bias": bias, "reason": reason}

    # Rank by confidence × R:R composite
    best = max(eligible, key=lambda x: x.confidence * min(x.expected_r, 5.0))
    return {"symbol": symbol, "regime": asdict(regime), "selected": asdict(best), "bias": bias}
