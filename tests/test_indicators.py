import unittest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.smc import analyze_smc_structure
from indicators.order_flow import analyze_order_flow
from indicators.market_context import set_market_data

class TestTradingIndicators(unittest.TestCase):
    def setUp(self):
        # Create a mock bullish trending dataframe with swings
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        close = np.linspace(100, 200, 100)
        # Add a swing high at index 20, swing low at index 40, etc.
        close[20] = 130 # Swing High
        close[19] = 125
        close[21] = 125
        
        close[40] = 110 # Swing Low
        close[39] = 115
        close[41] = 115
        
        close[60] = 180 # Higher High
        close[59] = 175
        close[61] = 175
        
        close[80] = 160 # Higher Low
        close[79] = 165
        close[81] = 165

        data = {
            "timestamp": dates,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.random.randint(100, 1000, 100)
        }
        self.bullish_df = pd.DataFrame(data)
        
        # Create a mock range dataframe with small oscillations
        range_close = [150 + (2 if i % 10 == 0 else -2 if i % 10 == 5 else 0) for i in range(100)]
        range_data = {
            "timestamp": dates,
            "open": [150] * 100,
            "high": [155] * 100,
            "low": [145] * 100,
            "close": range_close,
            "volume": [500] * 100
        }
        self.range_df = pd.DataFrame(range_data)

    def test_smc_bullish_structure(self):
        set_market_data(self.bullish_df)
        res = analyze_smc_structure()
        self.assertEqual(res["structure"], "Bullish")
        self.assertIn("zone", res)
        self.assertIn("pd_array", res)

    def test_smc_range_structure(self):
        # Create a clearer range with multiple swings
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        close = [150] * 100
        # Highs at 20, 40, 60, 80
        for i in [20, 40, 60, 80]:
            close[i] = 160
            close[i-1] = 155
            close[i+1] = 155
        # Lows at 30, 50, 70, 90
        for i in [30, 50, 70, 90]:
            close[i] = 140
            close[i-1] = 145
            close[i+1] = 145
            
        range_df = pd.DataFrame({
            "timestamp": dates,
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [500] * 100
        })
        set_market_data(range_df)
        res = analyze_smc_structure()
        self.assertEqual(res["structure"], "Ranging")

    def test_fvg_detection(self):
        # Inject a clear bullish FVG
        df = self.bullish_df.copy()
        # Ensure we have a gap: High[i-2] < Low[i]
        df.loc[90, "high"] = 150 
        df.loc[91, "open"] = 155
        df.loc[91, "close"] = 165
        df.loc[92, "low"] = 170 
        
        set_market_data(df)
        res = analyze_smc_structure()
        fvgs = res.get("fvgs", [])
        self.assertTrue(any(f["type"] == "bullish" for f in fvgs))

    def test_liquidity_sweep_detection(self):
        df = self.bullish_df.copy()
        # Find the last swing high price
        res_initial = analyze_smc_structure(df_override=df)
        last_sh = res_initial["pd_array"]["high"]
        
        # Create a sweep in the last 2 candles
        # Candle 98: Sweep High
        df.loc[98, "high"] = last_sh + 10
        df.loc[98, "close"] = last_sh - 2
        
        set_market_data(df)
        res = analyze_smc_structure()
        self.assertTrue(res["liquidity_sweep"]["high"])

if __name__ == "__main__":
    unittest.main()
