from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class StrategySignal:
    strategy_name: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str

class BaseStrategy:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        raise NotImplementedError

class EliteSMCStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Elite SMC", weight=2.0)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        # This will wrap the existing logic from edge.py but in a modular way
        from engine.edge import generate_signals, detect_market_regime
        regime = detect_market_regime(df, indicators)
        signals = generate_signals(df, indicators, regime, {})
        
        if not signals:
            return None
            
        # Pick the best signal from the SMC group
        best = max(signals, key=lambda x: x.confidence)
        return StrategySignal(
            strategy_name=f"{self.name} ({best.strategy})",
            action=best.action,
            confidence=best.confidence,
            entry=best.entry,
            stop_loss=best.stop_loss,
            take_profit=best.take_profit,
            reason=best.reason
        )

class InstitutionalTrapStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Institutional Trap", weight=1.6)
    
    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        smc = indicators.get("smc", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        sweep = smc.get("liquidity_sweep", {})
        fvgs = smc.get("fvgs", [])

        if sweep.get("low") and any(f["type"] == "bullish" for f in fvgs):
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.85,
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=price + atr * 4,
                reason="Liquidity Sweep + Bullish FVG detected"
            )
        
        if sweep.get("high") and any(f["type"] == "bearish" for f in fvgs):
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.85,
                entry=price,
                stop_loss=price + atr * 2,
                take_profit=price - atr * 4,
                reason="Liquidity Sweep + Bearish FVG detected"
            )
        return None

class CVDAbsorptionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("CVD Absorption", weight=1.3)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        smc = indicators.get("smc", {})
        of = indicators.get("order_flow", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        abs_bias = of.get("absorption", "None")

        if abs_bias == "Bullish Absorption" and smc.get("zone") == "Discount":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.82,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=price + atr * 3.5,
                reason="Big money absorbing sells in Discount zone"
            )
        elif abs_bias == "Bearish Absorption" and smc.get("zone") == "Premium":
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.82,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=price - atr * 3.5,
                reason="Big money absorbing buys in Premium zone"
            )
        return None

class SilverBulletStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("ICT Silver Bullet", weight=1.1)

    def is_window(self) -> bool:
        from datetime import datetime
        import pytz
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        return now_ny.hour in [3, 10, 14]

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        if not self.is_window():
            return None
            
        smc = indicators.get("smc", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        if smc.get("structure") == "Bullish":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.80,
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=price + atr * 4,
                reason="NY Momentum Window + Bullish Structure"
            )
        elif smc.get("structure") == "Bearish":
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.80,
                entry=price,
                stop_loss=price + atr * 2,
                take_profit=price - atr * 4,
                reason="NY Momentum Window + Bearish Structure"
            )
        return None

class CoinbasePremiumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Coinbase Premium", weight=1.8)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        from indicators.correlation import analyze_cross_exchange_correlation
        symbol = indicators.get("symbol", "BTC/USDT")
        
        # This is a HEAVY call, we only do it if we are looking for high-conviction
        analysis = analyze_cross_exchange_correlation(symbol)
        if "error" in analysis:
            return None
            
        bias = analysis.get("lead_lag_bias", "Neutral")
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)

        if bias == "Bullish (Spot Leading)":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.90, # Institutional spot buying is a massive signal
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=price + atr * 5,
                reason="Institutions are buying on Coinbase (Spot Leading)"
            )
        elif bias == "Bearish (Perp Leading)":
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.90,
                entry=price,
                stop_loss=price + atr * 2,
                take_profit=price - atr * 5,
                reason="Retail is over-leveraged on Binance (Perp Leading)"
            )
        return None

class LeaderLaggardStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Leader-Laggard", weight=1.2)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        from indicators.correlation import analyze_asset_correlation
        symbol = indicators.get("symbol", "LINK/USDT")
        
        # Check if BTC or SOL are pumping while we are flat
        corr_data = analyze_asset_correlation(symbol, competitors=["BTC/USDT", "SOL/USDT"])
        if "error" in corr_data:
            return None
            
        # If correlation is high but price returns are diverging
        correlations = corr_data.get("correlations", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)

        # Implementation of "The Shadow" move
        # (This assumes the engine passes recent returns in indicators)
        # For now, we use the average correlation cluster as a proxy for high-prob catch-up
        if corr_data.get("is_systemic_cluster") and indicators.get("smc", {}).get("structure") == "Bullish":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.85,
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=price + atr * 4,
                reason="Systemic Cluster pump detected: Leader is moving, Laggard entry."
            )
        return None

class LiquidationHunterStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Liquidation Hunter", weight=1.5)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        smc = indicators.get("smc", {})
        of = indicators.get("order_flow", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        sweep = smc.get("liquidity_sweep", {})
        delta = of.get("delta", 0.0)
        
        if sweep.get("low") and delta > 0:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.88,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=price + atr * 4.5,
                reason="Liquidation Hunt: Retail stops swept + Big money absorption"
            )
        
        if sweep.get("high") and delta < 0:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.88,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=price - atr * 4.5,
                reason="Liquidation Hunt: Retail stops swept + Big money distribution"
            )
        return None

class AMDStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Power of Three (AMD)", weight=1.3)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        smc = indicators.get("smc", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        structure = smc.get("structure", "Neutral")
        pd_array = smc.get("pd_array", {})
        eq = pd_array.get("eq", price)

        if structure == "Bullish" and price < eq:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.84,
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=eq + (eq - price),
                reason="AMD: Manipulation phase over, entering Distribution"
            )
        elif structure == "Bearish" and price > eq:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.84,
                entry=price,
                stop_loss=price + atr * 2,
                take_profit=eq - (price - eq),
                reason="AMD: Manipulation phase over, entering Distribution"
            )
        return None

class VWAPMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("VWAP Mean Reversion", weight=1.1)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        vwap_analysis = indicators.get("vwap", {})
        z_score = vwap_analysis.get("z_score", 0.0)
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        if z_score < -2.5:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.82,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=vwap_analysis.get("vwap", price),
                reason=f"VWAP Extreme: Z-Score {z_score:.2f} (Snap-back to fair value)"
            )
        elif z_score > 2.5:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.82,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=vwap_analysis.get("vwap", price),
                reason=f"VWAP Extreme: Z-Score {z_score:.2f} (Snap-back to fair value)"
            )
        return None

class TrendScalperStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Trend Scalper")

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        price = float(df.iloc[-1]["close"])
        ema_50 = df["EMA_50"].iloc[-1] if "EMA_50" in df.columns else None
        ema_200 = df["EMA_200"].iloc[-1] if "EMA_200" in df.columns else None
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        rsi = indicators.get("rsi", 50)
        
        if not ema_50 or not ema_200:
            return None

        # Bullish: Price > EMA 50 > EMA 200 and RSI is not overbought
        if price > ema_50 > ema_200 and rsi < 70:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.78,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=price + atr * 3.0,
                reason="Trend following: Price > EMA50 > EMA200"
            )
        # Bearish: Price < EMA 50 < EMA 200 and RSI is not oversold
        elif price < ema_50 < ema_200 and rsi > 30:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.78,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=price - atr * 3.0,
                reason="Trend following: Price < EMA50 < EMA200"
            )
        return None

class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Momentum Breakout", weight=1.0)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        price = float(df.iloc[-1]["close"])
        vol = indicators.get("vol", {})
        rsi = indicators.get("rsi", 50)
        trend = indicators.get("trend", {})
        ema_200 = df["EMA_200"].iloc[-1] if "EMA_200" in df.columns else None
        atr = vol.get("atr", price * 0.01)
        
        # Standard: Volume Spike (3x) + ADX Rising (>25)
        vol_spike = df["volume"].iloc[-1] > df["volume"].tail(20).mean() * 3.0
        adx = trend.get("adx", 0)
        
        if vol_spike and adx > 25:
            # Bullish Breakout with Trend Filter
            if rsi > 60 and price > ema_200 and df["close"].iloc[-1] > df["high"].iloc[-2]:
                return StrategySignal(
                    strategy_name=self.name,
                    action="buy",
                    confidence=0.88,
                    entry=price,
                    stop_loss=price - atr * 2.5,
                    take_profit=price + atr * 6,
                    reason="Optimal Momentum: Vol spike + ADX trend + EMA alignment"
                )
            # Bearish Breakout with Trend Filter
            elif rsi < 40 and price < ema_200 and df["close"].iloc[-1] < df["low"].iloc[-2]:
                return StrategySignal(
                    strategy_name=self.name,
                    action="sell",
                    confidence=0.88,
                    entry=price,
                    stop_loss=price + atr * 2.5,
                    take_profit=price - atr * 6,
                    reason="Optimal Momentum: Vol spike + ADX trend + EMA alignment"
                )
        return None

class RangeMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Range Mean Reversion", weight=1.0)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        bb = indicators.get("bollinger", {})
        rsi = indicators.get("rsi", 50)
        trend = indicators.get("trend", {})
        of = indicators.get("order_flow", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        # Standard: Low ADX (<20) indicates ranging market
        adx = trend.get("adx", 0)
        if adx > 20: return None 
        
        if not bb or "upper" not in bb: return None
            
        # Confluence: BB outer + RSI extreme + Absorption
        if price <= bb["lower"] and rsi < 30 and of.get("absorption") == "Bullish Absorption":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.85,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=bb.get("mid", price + atr * 2),
                reason="Optimal Range: Low ADX + BB/RSI Extreme + Absorption"
            )
        elif price >= bb["upper"] and rsi > 70 and of.get("absorption") == "Bearish Absorption":
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.85,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=bb.get("mid", price - atr * 2),
                reason="Optimal Range: Low ADX + BB/RSI Extreme + Absorption"
            )
        return None

class TapeReadingImbalanceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Tape Reading (Order Flow)", weight=1.4)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        of = indicators.get("order_flow", {})
        smc = indicators.get("smc", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        imbalance = of.get("imbalance", "Neutral")
        absorption = of.get("absorption", "None")
        zone = smc.get("zone", "Neutral")
        
        # Confluence: Don't buy bullish imbalance in Premium zone (likely trap)
        if (imbalance == "Extreme Bullish Imbalance" or absorption == "Bullish Absorption") and zone == "Discount":
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.89,
                entry=price,
                stop_loss=price - atr * 1.5,
                take_profit=price + atr * 4.5,
                reason=f"Standard Tape: {imbalance} in {zone} zone"
            )
        elif (imbalance == "Extreme Bearish Imbalance" or absorption == "Bearish Absorption") and zone == "Premium":
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.89,
                entry=price,
                stop_loss=price + atr * 1.5,
                take_profit=price - atr * 4.5,
                reason=f"Standard Tape: {imbalance} in {zone} zone"
            )
        return None

class EliteStructuralBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Elite Structural Breakout", weight=1.9)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        of = indicators.get("order_flow", {})
        oi = indicators.get("oi", {})
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        vah, val = of.get("vah", 0), of.get("val", 0)
        imbalance = of.get("imbalance", "Neutral")
        
        if not vah or not val: return None
        
        # Elite: Breakout + Tape Imbalance + OI Aggressive bias
        oi_bias = oi.get("oi_bias", "Neutral")
            
        if price > vah and df["close"].iloc[-2] <= vah:
            if imbalance == "Extreme Bullish Imbalance" and oi_bias == "Aggressive Bullish":
                return StrategySignal(
                    strategy_name=self.name,
                    action="buy",
                    confidence=0.94,
                    entry=price,
                    stop_loss=vah,
                    take_profit=price + atr * 6,
                    reason="Elite: VAH escape + Bullish Tape + Aggressive OI confirmation"
                )
        elif price < val and df["close"].iloc[-2] >= val:
            if imbalance == "Extreme Bearish Imbalance" and oi_bias == "Aggressive Bearish":
                return StrategySignal(
                    strategy_name=self.name,
                    action="sell",
                    confidence=0.94,
                    entry=price,
                    stop_loss=val,
                    take_profit=price - atr * 6,
                    reason="Elite: VAL escape + Bearish Tape + Aggressive OI confirmation"
                )
        return None

class InstitutionalMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Institutional Reload", weight=1.7)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        smc = indicators.get("smc", {})
        rsi_series = indicators.get("rsi_series", [])
        if len(df) < 10 or len(rsi_series) < 10: return None
        
        price_now, price_prev = df["close"].iloc[-1], df["low"].tail(10).iloc[0] # Looking back further for structural low
        rsi_now, rsi_prev = rsi_series[-1], min(rsi_series[-10:-1])
        atr = indicators.get("vol", {}).get("atr", price_now * 0.01)
        
        # Elite: Hidden Divergence (Trend Continuation)
        # Bullish Hidden: Price makes Higher Low, RSI makes Lower Low
        # This signals that the trend is strong and being reloaded.
        
        structure = smc.get("structure", "Neutral")
        fvgs = smc.get("fvgs", [])
        obs = smc.get("order_blocks", [])
        
        is_in_fvg = any(f["type"] == "bullish" and f["bottom"] <= price_now <= f["top"] for f in fvgs)
        is_in_ob = any(o["type"] == "bullish" and o["bottom"] <= price_now <= o["top"] for o in obs)
        
        # Bullish Hidden Divergence at POI
        if structure == "Bullish" and price_now > price_prev and rsi_now < rsi_prev:
            if is_in_fvg or is_in_ob:
                return StrategySignal(
                    strategy_name=self.name,
                    action="buy",
                    confidence=0.95,
                    entry=price_now,
                    stop_loss=price_now - atr * 2,
                    take_profit=price_now + atr * 6,
                    reason="Elite: Hidden Bullish Divergence at FVG/OB (Institutional Reload)"
                )
                
        # Bearish Hidden: Price makes Lower High, RSI makes Higher High
        price_high_now, price_high_prev = df["close"].iloc[-1], df["high"].tail(10).iloc[0]
        rsi_high_now, rsi_high_prev = rsi_series[-1], max(rsi_series[-10:-1])
        
        is_in_bear_fvg = any(f["type"] == "bearish" and f["bottom"] <= price_now <= f["top"] for f in fvgs)
        
        if structure == "Bearish" and price_high_now < price_high_prev and rsi_high_now > rsi_high_prev:
            if is_in_bear_fvg:
                return StrategySignal(
                    strategy_name=self.name,
                    action="sell",
                    confidence=0.95,
                    entry=price_now,
                    stop_loss=price_now + atr * 2,
                    take_profit=price_now - atr * 6,
                    reason="Elite: Hidden Bearish Divergence at FVG (Institutional Reload)"
                )
        return None

class ClimaxReversalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Institutional Climax", weight=1.2)

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        price = float(df.iloc[-1]["close"])
        vol = df["volume"].iloc[-1]
        avg_vol = df["volume"].tail(20).mean()
        of = indicators.get("order_flow", {})
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        # Elite: Massive volume + Tape Absorption confirmation
        if vol > avg_vol * 4.5:
            abs_bias = of.get("absorption", "None")
            if abs_bias == "Bullish Absorption" and indicators.get("rsi", 50) < 30:
                return StrategySignal(
                    strategy_name=self.name,
                    action="buy",
                    confidence=0.90,
                    entry=price,
                    stop_loss=price - atr * 2,
                    take_profit=price + atr * 6,
                    reason="Elite: Volume Climax + Bullish Absorption reversal"
                )
            elif abs_bias == "Bearish Absorption" and indicators.get("rsi", 50) > 70:
                return StrategySignal(
                    strategy_name=self.name,
                    action="sell",
                    confidence=0.90,
                    entry=price,
                    stop_loss=price + atr * 2,
                    take_profit=price - atr * 6,
                    reason="Elite: Volume Climax + Bearish Absorption reversal"
                )
        return None

class StrategyOrchestrator:
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
            InstitutionalMomentumStrategy(),
            ClimaxReversalStrategy(),
            EliteStructuralBreakoutStrategy(),
            MomentumBreakoutStrategy(),
            RangeMeanReversionStrategy(),
            TapeReadingImbalanceStrategy(),
            CoinbasePremiumStrategy(),
            LeaderLaggardStrategy(),
            LiquidationHunterStrategy(),
            AMDStrategy(),
            VWAPMeanReversionStrategy(),
            InstitutionalTrapStrategy(),
            CVDAbsorptionStrategy(),
            SilverBulletStrategy(),
            EliteSMCStrategy()
        ]

    def get_best_signal(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        active_signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.evaluate(df, indicators)
                if signal and signal.action in ["buy", "sell"]:
                    # Attach the strategy's weight to the signal for the aggregator
                    signal.strategy_weight = strategy.weight
                    active_signals.append(signal)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Strategy {strategy.name} failed: {e}")

        if not active_signals:
            return None

        # --- EXPERT WEIGHTED AGGREGATOR ---
        buy_sigs = [s for s in active_signals if s.action == "buy"]
        sell_sigs = [s for s in active_signals if s.action == "sell"]
        
        buy_weight_sum = sum(s.strategy_weight for s in buy_sigs)
        sell_weight_sum = sum(s.strategy_weight for s in sell_sigs)
        
        # Decide direction based on Weight Dominance
        if buy_weight_sum > sell_weight_sum:
            winning_sigs = buy_sigs
            winning_weight = buy_weight_sum
            action = "buy"
        else:
            winning_sigs = sell_sigs
            winning_weight = sell_weight_sum
            action = "sell"

        # Calculate Mathematically Weighted Average Confidence
        # Formula: (Sum of Confidence * Weight) / Sum of Weights
        weighted_conf_sum = sum(s.confidence * s.strategy_weight for s in winning_sigs)
        avg_confidence = weighted_conf_sum / winning_weight
        
        # Confluence Multiplier: 
        # Boost starts at 2 agreeing strategies for higher frequency
        if len(winning_sigs) >= 2:
            avg_confidence = min(0.99, avg_confidence * 1.15)

        # Pick the most authoritative strategy as the primary metadata source
        best_signal = max(winning_sigs, key=lambda x: x.strategy_weight)
        best_signal.confidence = avg_confidence
        
        # --- EXECUTION FILTERS (Optimized for 5m Frequency) ---
        price = float(df.iloc[-1]["close"])
        vol = indicators.get("vol", {})
        spread_bps = indicators.get("spread_bps", 3.0) 
        
        cost_impact = (spread_bps / 10000) * price
        expected_profit = abs(best_signal.take_profit - best_signal.entry)
        
        if cost_impact > expected_profit * 0.20:
            return None # Relaxed cost threshold to 20%

        # Macro Sentiment Filter (Relaxed to 0.85x)
        macro = indicators.get("macro", {})
        sentiment = macro.get("sentiment", {}).get("score", 50)
        liquidity = indicators.get("liquidity", {})
        
        if (sentiment < 20 and action == "sell") or (sentiment > 80 and action == "buy"):
            best_signal.confidence *= 0.85
            
        # Dynamic Liquidity
        if action == "buy":
            targets = liquidity.get("buy_side_liquidity_targets", [])
            if targets:
                valid_targets = [t for t in targets if t > best_signal.entry]
                if valid_targets: best_signal.take_profit = min(valid_targets)
        elif action == "sell":
            targets = liquidity.get("sell_side_liquidity_targets", [])
            if targets:
                valid_targets = [t for t in targets if t < best_signal.entry]
                if valid_targets: best_signal.take_profit = max(valid_targets)
        
        # Final Expert Gate (Adjusted to 0.75 for optimal balance)
        if best_signal.confidence < 0.75:
            return None

        best_signal.reason = f"WEIGHTED CONSENSUS ({len(winning_sigs)} models) | " + best_signal.reason
        return best_signal
