import asyncio
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd

log = logging.getLogger(__name__)

class GlobalAsyncCache:
    """
    Global Async Cache to offload heavy API calls and computations 
    from the main trading cycle to prevent stalling.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalAsyncCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, tools=None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.tools = tools
        self.cache: Dict[str, Any] = {}
        self.last_updated: Dict[str, float] = {}
        self.refresh_intervals: Dict[str, int] = {
            "cross_exchange": 300,  # 5 mins
            "asset_correlation": 600, # 10 mins
            "macro_sentiment": 1800, # 30 mins
            "funding_rates": 3600,   # 1 hour
        }
        self.tasks = {}
        self.is_running = False
        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default)

    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.last_updated[key] = time.time()

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        log.info("Starting Global Async Cache background workers...")
        self.tasks["background_refresh"] = asyncio.create_task(self._refresh_loop())

    async def stop(self):
        self.is_running = False
        for name, task in self.tasks.items():
            task.cancel()
        log.info("Global Async Cache workers stopped.")

    async def _refresh_loop(self):
        while self.is_running:
            try:
                # 1. Macro Sentiment Refresh
                if self._should_refresh("macro_sentiment"):
                    await self._refresh_macro()

                # 2. Cross Exchange Correlation
                # We do this for the primary symbols
                if self._should_refresh("cross_exchange"):
                    await self._refresh_cross_exchange()

                # 3. Asset Correlation
                if self._should_refresh("asset_correlation"):
                    await self._refresh_asset_correlation()

            except Exception as e:
                log.error(f"Cache refresh loop error: {e}")
            
            await asyncio.sleep(10) # check every 10s

    def _should_refresh(self, key: str) -> bool:
        last = self.last_updated.get(key, 0)
        interval = self.refresh_intervals.get(key, 300)
        return (time.time() - last) > interval

    async def _refresh_macro(self):
        log.debug("Refreshing Macro Sentiment Cache...")
        try:
            from indicators.macro import get_macro_analysis
            res = await get_macro_analysis()
            self.set("macro_sentiment", res)
        except Exception as e:
            log.warning(f"Macro refresh failed: {e}")

    async def _refresh_cross_exchange(self):
        log.debug("Refreshing Cross Exchange Correlation Cache...")
        try:
            from indicators.correlation import analyze_cross_exchange_correlation
            # We fetch for BTC by default as a market proxy
            res = await asyncio.to_thread(analyze_cross_exchange_correlation, "BTC/USDT")
            self.set("cross_exchange_BTC", res)
        except Exception as e:
            log.warning(f"Cross exchange refresh failed: {e}")

    async def _refresh_asset_correlation(self):
        log.debug("Refreshing Asset Correlation Cache...")
        try:
            from indicators.correlation import analyze_asset_correlation
            # Proxy check for BTC vs Major alts
            res = await asyncio.to_thread(analyze_asset_correlation, "BTC/USDT", ["ETH/USDT", "SOL/USDT"])
            self.set("asset_correlation_market", res)
        except Exception as e:
            log.warning(f"Asset correlation refresh failed: {e}")
