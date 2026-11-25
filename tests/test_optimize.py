import numpy as np
import pandas as pd
import pytest

from src.data import estimate
from src.optimize import efficient_frontier, max_sharpe, min_variance, stats, target_return


@pytest.fixture
def two_assets():
    mu = pd.Series([0.05, 0.10], index=["bond", "stock"])
    s1, s2, rho = 0.05, 0.20, 0.2
    cov = pd.DataFrame([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]], index=mu.index, columns=mu.index)
    return mu, cov


@pytest.fixture
def five_assets():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rets = rng.multivariate_normal([0.0002, 0.0003, 0.0004, 0.0001, 0.0005],
                                   np.diag([1, 2, 3, 0.5, 4]) * 1e-4, size=len(idx))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCDE"))
    return estimate(prices)


def test_min_variance_matches_the_two_asset_formula(two_assets):
    mu, cov = two_assets
    v1, v2, c = cov.iloc[0, 0], cov.iloc[1, 1], cov.iloc[0, 1]
    w_bond = (v2 - c) / (v1 + v2 - 2 * c)
    assert min_variance(mu, cov)["bond"] == pytest.approx(w_bond, abs=1e-5)


def test_weights_are_long_only_sum_to_one_and_respect_the_cap(five_assets):
    mu, cov = five_assets
    for w in (min_variance(mu, cov, 0.4), max_sharpe(mu, cov, 0.0, 0.4)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all() and (w <= 0.4 + 1e-6).all()


def test_max_sharpe_beats_every_frontier_point(five_assets):
    mu, cov = five_assets
    best = stats(max_sharpe(mu, cov, 0.01), mu, cov, 0.01)["sharpe"]
    fr = efficient_frontier(mu, cov, points=20)
    assert ((fr["return"] - 0.01) / fr["volatility"]).max() <= best + 1e-4


def test_frontier_risk_rises_with_return(five_assets):
    mu, cov = five_assets
    fr = efficient_frontier(mu, cov, points=15)
    assert (fr["volatility"].diff().dropna() >= -1e-6).all()


def test_target_return_is_hit(five_assets):
    mu, cov = five_assets
    r = float(mu.mean())
    assert stats(target_return(mu, cov, r), mu, cov)["return"] == pytest.approx(r, abs=1e-6)
