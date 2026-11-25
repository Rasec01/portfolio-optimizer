"""Expected returns and covariance, annualised from daily prices."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


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
