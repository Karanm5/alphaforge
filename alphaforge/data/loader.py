"""
AlphaForge Data Layer
Unified interface for market data (yfinance) and macro data (FRED).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

# Optional FRED import
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# ─────────────────────────────────────────────
#  Market Data
# ─────────────────────────────────────────────

def fetch_prices(
    tickers: List[str],
    start: str = "2018-01-01",
    end: Optional[str] = None,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame with tickers as columns.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        # multi-ticker download
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.iloc[:, :len(tickers)]
            prices.columns = tickers
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    return prices


def fetch_returns(
    tickers: List[str],
    start: str = "2018-01-01",
    end: Optional[str] = None,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Compute log or simple returns from price data."""
    prices = fetch_prices(tickers, start, end)
    if log_returns:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


def fetch_ohlcv(ticker: str, start: str = "2018-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Return full OHLCV for a single ticker."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return data


# ─────────────────────────────────────────────
#  FRED Macro Data
# ─────────────────────────────────────────────

FRED_SERIES = {
    "VIX":        "VIXCLS",
    "FFR":        "DFF",          # Fed Funds Rate
    "US10Y":      "DGS10",        # 10-year Treasury
    "US2Y":       "DGS2",         # 2-year Treasury
    "SPREAD_10_2":"T10Y2Y",       # 10Y-2Y spread
    "CPI":        "CPIAUCSL",     # CPI
    "UNRATE":     "UNRATE",       # Unemployment
    "GDP":        "GDP",          # GDP
    "CREDIT":     "BAMLH0A0HYM2", # HY OAS spread
}


def fetch_fred(
    series_ids: List[str],
    api_key: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch FRED series by their FRED IDs.
    series_ids can be friendly names from FRED_SERIES dict or raw FRED IDs.
    """
    if not FRED_AVAILABLE:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    fred = Fred(api_key=api_key)
    frames = {}

    for sid in series_ids:
        fred_id = FRED_SERIES.get(sid, sid)
        try:
            s = fred.get_series(fred_id, observation_start=start, observation_end=end)
            s.name = sid
            frames[sid] = s
        except Exception as e:
            print(f"[FRED] Could not fetch {sid} ({fred_id}): {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def fetch_macro_dashboard(api_key: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Convenience: fetch all standard macro series."""
    return fetch_fred(list(FRED_SERIES.keys()), api_key, start)


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def align_series(*dfs: pd.DataFrame, fill_method: str = "ffill") -> List[pd.DataFrame]:
    """Align multiple DataFrames on their common date index."""
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
    aligned = []
    for df in dfs:
        a = df.loc[common_idx]
        if fill_method == "ffill":
            a = a.ffill()
        aligned.append(a)
    return aligned


def compute_rolling_stats(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling annualised stats for a single return series."""
    ann = 252
    stats = pd.DataFrame(index=returns.index)
    stats["rolling_ret"]  = returns.rolling(window).mean() * ann
    stats["rolling_vol"]  = returns.rolling(window).std() * np.sqrt(ann)
    stats["rolling_sr"]   = stats["rolling_ret"] / stats["rolling_vol"]
    stats["rolling_dd"]   = (
        (1 + returns).cumprod() / (1 + returns).cumprod().expanding().max() - 1
    )
    return stats


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from return series."""
    cumret = (1 + returns).cumprod()
    peak   = cumret.expanding().max()
    dd     = (cumret - peak) / peak
    return dd


def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(returns).min()


def calmar_ratio(returns: pd.Series, ann: int = 252) -> float:
    annual_ret = returns.mean() * ann
    mdd = abs(max_drawdown(returns))
    return annual_ret / mdd if mdd > 0 else np.nan
