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
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        raise NotImplementedError

class EliteSMCStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Elite SMC")

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
        super().__init__("Institutional Trap")
    
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
        super().__init__("CVD Absorption")

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
        super().__init__("ICT Silver Bullet")

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
        super().__init__("Coinbase Premium")

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
        super().__init__("Leader-Laggard")

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
        super().__init__("Liquidation Hunter")

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
        super().__init__("Power of Three (AMD)")

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
        super().__init__("VWAP Mean Reversion")

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

class StrategyOrchestrator:
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
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
        all_signals = []
        for strat in self.strategies:
            try:
                sig = strat.evaluate(df, indicators)
                if sig:
                    all_signals.append(sig)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Strategy {strat.name} failed: {e}")
        
        if not all_signals:
            return None
            
        return max(all_signals, key=lambda x: x.confidence)
