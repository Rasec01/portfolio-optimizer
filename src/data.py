"""Prices for several assets, and the expected returns and covariance estimated from them."""
from typing import Optional, Sequence

import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def load_prices(tickers: Sequence[str], start: str = "2010-01-01", csv: Optional[str] = None) -> pd.DataFrame:
    if csv:
        prices = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf

        prices = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="any").sort_index()
    print(f"[data] {prices.shape[1]} assets, {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def estimate(prices: pd.DataFrame, shrink: bool = True):
    """
    Return (mu, cov) as annualised pandas objects.

    Sample covariance is noisy when assets are many and history is short, and the
    optimiser leans hard on that noise. Ledoit-Wolf shrinkage pulls it towards a
    structured target and usually gives steadier weights.
    """
    rets = prices.pct_change().dropna()
    mu = rets.mean() * TRADING_DAYS
    if shrink:
        cov = LedoitWolf().fit(rets.to_numpy()).covariance_
        cov = pd.DataFrame(cov, index=rets.columns, columns=rets.columns)
    else:
        cov = rets.cov()
    return mu, cov * TRADING_DAYS
