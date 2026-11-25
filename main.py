"""
Mean-variance portfolio optimiser.

    python main.py
    python main.py --tickers SPY EFA EEM AGG GLD --max-weight 0.4 --chart outputs/frontier.png
    python main.py --holdout-start 2021-01-01     # estimate before, evaluate after
    python main.py --csv data/prices.csv          # Date + one column per asset
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TRADING_DAYS, estimate, load_prices
from src.optimize import efficient_frontier, max_sharpe, min_variance, stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"])
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--csv", default=None)
    p.add_argument("--rf", type=float, default=0.02, help="annual risk-free rate for the Sharpe ratio")
    p.add_argument("--max-weight", type=float, default=1.0, help="cap on any single asset")
    p.add_argument("--no-shrink", action="store_true", help="use the sample covariance")
    p.add_argument("--holdout-start", default=None, help="estimate before this date, evaluate from it")
    p.add_argument("--chart", default=None, help="save the efficient frontier to this file")
    return p.parse_args()


def realised(prices: pd.DataFrame, w: pd.Series, rf: float) -> dict:
    """Performance of fixed weights, rebalanced daily, over the given prices."""
    daily = prices.pct_change().dropna() @ w
    ann_ret = (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1
    ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
    return {"return": ann_ret, "volatility": ann_vol, "sharpe": (ann_ret - rf) / ann_vol}


def main():
    a = parse_args()
    prices = load_prices(a.tickers, a.start, a.csv)
    est = prices if a.holdout_start is None else prices.loc[: a.holdout_start].iloc[:-1]
    mu, cov = estimate(est, shrink=not a.no_shrink)

    ports = {
        "Max Sharpe": max_sharpe(mu, cov, a.rf, a.max_weight),
        "Min variance": min_variance(mu, cov, a.max_weight),
        "Equal weight": pd.Series(1 / len(mu), index=mu.index),
    }
    print("\nWeights:")
    print(pd.DataFrame(ports).round(3).to_string())

    print("\nIn-sample (estimated) statistics:")
    print(pd.DataFrame({k: stats(w, mu, cov, a.rf) for k, w in ports.items()}).round(4).to_string())

    if a.holdout_start:
        hold = prices.loc[a.holdout_start:]
        print(f"\nOut of sample, {hold.index[0].date()} to {hold.index[-1].date()} (weights fixed):")
        print(pd.DataFrame({k: realised(hold, w, a.rf) for k, w in ports.items()}).round(4).to_string())

    if a.chart:
        import matplotlib.pyplot as plt

        fr = efficient_frontier(mu, cov, max_weight=a.max_weight)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fr["volatility"], fr["return"], label="Efficient frontier")
        for k, w in ports.items():
            s = stats(w, mu, cov, a.rf)
            ax.scatter(s["volatility"], s["return"], zorder=3, label=k)
        vols = np.sqrt(np.diag(cov))
        ax.scatter(vols, mu, marker="x", color="grey", label="Single assets")
        for t, x, y in zip(mu.index, vols, mu):
            ax.annotate(t, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Annual volatility")
        ax.set_ylabel("Annual expected return")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(a.chart).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.chart, dpi=150)
        print(f"\nChart saved to {a.chart}")


if __name__ == "__main__":
    main()
