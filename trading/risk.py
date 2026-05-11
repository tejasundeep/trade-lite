from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRiskConfig:
    max_risk_per_trade_pct: float = 0.015
    max_position_pct:       float = 0.15
    min_order_quote:        float = 10.0
    max_daily_loss_pct:     float = 0.05
    cooldown_minutes:       int   = 45
    liquidity_depth_factor: float = 0.10


class MarketRiskManager:
    """Thin config holder — cooldown persistence lives in indicators/risk.py (DB-backed)."""
    def __init__(self, config: MarketRiskConfig = MarketRiskConfig()):
        self.config = config
