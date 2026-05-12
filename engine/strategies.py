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

class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("RSI Mean Reversion")

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        rsi = indicators.get("rsi", 50.0)
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        if rsi < 30:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.7,
                entry=price,
                stop_loss=price - atr * 2,
                take_profit=price + atr * 3,
                reason=f"Oversold RSI ({rsi:.1f})"
            )
        elif rsi > 70:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.7,
                entry=price,
                stop_loss=price + atr * 2,
                take_profit=price - atr * 3,
                reason=f"Overbought RSI ({rsi:.1f})"
            )
        return None

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("EMA Crossover")

    def evaluate(self, df: pd.DataFrame, indicators: Dict) -> Optional[StrategySignal]:
        # Assume df has EMA_50 and EMA_200
        if "EMA_50" not in df.columns or "EMA_200" not in df.columns:
            return None
            
        ema_short = df["EMA_50"].iloc[-1]
        ema_long = df["EMA_200"].iloc[-1]
        prev_short = df["EMA_50"].iloc[-2]
        prev_long = df["EMA_200"].iloc[-2]
        
        price = float(df.iloc[-1]["close"])
        atr = indicators.get("vol", {}).get("atr", price * 0.01)
        
        # Golden Cross
        if prev_short <= prev_long and ema_short > ema_long:
            return StrategySignal(
                strategy_name=self.name,
                action="buy",
                confidence=0.75,
                entry=price,
                stop_loss=ema_long,
                take_profit=price + atr * 5,
                reason="Golden Cross (EMA 50/200)"
            )
        # Death Cross
        elif prev_short >= prev_long and ema_short < ema_long:
            return StrategySignal(
                strategy_name=self.name,
                action="sell",
                confidence=0.75,
                entry=price,
                stop_loss=ema_long,
                take_profit=price - atr * 5,
                reason="Death Cross (EMA 50/200)"
            )
        return None

class StrategyOrchestrator:
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
            EliteSMCStrategy(),
            RSIMeanReversionStrategy(),
            EMACrossoverStrategy()
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
            
        # Prioritize by confidence
        return max(all_signals, key=lambda x: x.confidence)
