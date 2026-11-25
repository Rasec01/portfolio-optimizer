"""Price data for several assets: download with yfinance or read a wide CSV (Date + one column per asset)."""
from typing import Optional, Sequence

import pandas as pd


def load_prices(tickers: Sequence[str], start: str = "2010-01-01", csv: Optional[str] = None) -> pd.DataFrame:
    if csv:
        prices = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf

        prices = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="any").sort_index()
    print(f"[data] {prices.shape[1]} assets, {len(prices)} days, {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices
