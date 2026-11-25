"""
Mean-variance optimisation (Markowitz, 1952) with long-only weights that sum to one.

All functions take annualised expected returns `mu` and covariance `cov`
(pandas objects indexed by asset) and return weights as a pandas Series.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def stats(w, mu, cov, rf: float = 0.0) -> dict:
    w = np.asarray(w)
    ret = float(w @ np.asarray(mu))
    vol = float(np.sqrt(w @ np.asarray(cov) @ w))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan}


def _solve(objective, n, max_weight, extra=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra]
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(mu, cov, max_weight: float = 1.0) -> pd.Series:
    S = np.asarray(cov)
    w = _solve(lambda w: w @ S @ w, len(mu), max_weight)
    return pd.Series(w, index=mu.index, name="min_variance")


def max_sharpe(mu, cov, rf: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    m, S = np.asarray(mu), np.asarray(cov)
    w = _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w), len(mu), max_weight)
    return pd.Series(w, index=mu.index, name="max_sharpe")


def target_return(mu, cov, target: float, max_weight: float = 1.0) -> pd.Series:
    m, S = np.asarray(mu), np.asarray(cov)
    con = {"type": "eq", "fun": lambda w: w @ m - target}
    w = _solve(lambda w: w @ S @ w, len(mu), max_weight, extra=(con,))
    return pd.Series(w, index=mu.index, name=f"target_{target:.2%}")


def efficient_frontier(mu, cov, points: int = 40, max_weight: float = 1.0) -> pd.DataFrame:
    """Volatility of the cheapest portfolio for each attainable target return."""
    lo = stats(min_variance(mu, cov, max_weight), mu, cov)["return"]
    hi = _max_attainable_return(mu, max_weight)
    rows = []
    for r in np.linspace(lo, hi, points):
        try:
            w = target_return(mu, cov, r, max_weight)
        except RuntimeError:
            continue
        rows.append({"return": r, "volatility": stats(w, mu, cov)["volatility"]})
    return pd.DataFrame(rows)


def _max_attainable_return(mu, max_weight):
    """Highest return reachable with long-only weights capped at max_weight."""
    left, total = 1.0, 0.0
    for r in np.sort(np.asarray(mu))[::-1]:
        take = min(max_weight, left)
        total += take * r
        left -= take
        if left <= 1e-12:
            break
    return total - 1e-6
