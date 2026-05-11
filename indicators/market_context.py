from contextvars import ContextVar
import pandas as pd
from typing import Optional

_market_data: ContextVar[Optional[pd.DataFrame]] = ContextVar("market_data", default=None)
_is_backtest: ContextVar[bool] = ContextVar("is_backtest", default=False)

def set_market_data(df: pd.DataFrame):
    _market_data.set(df)

def get_market_data() -> Optional[pd.DataFrame]:
    return _market_data.get()

def set_backtest(val: bool):
    _is_backtest.set(val)

def is_backtest() -> bool:
    return _is_backtest.get()
