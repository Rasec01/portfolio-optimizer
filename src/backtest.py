from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def backtest(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Simulate a long/short strategy based on the 'signal' column.

    The signal is lagged by one period to avoid look-ahead bias:
    today's signal is based on today's prediction, but the trade
    is executed at tomorrow's open (approximated as today's close).

    Args:
        df: DataFrame with 'signal' and 'return' columns.

    Returns:
        (cumulative_strategy, cumulative_market) as pd.Series.
    """
    df = df.copy()
    df["strategy_return"] = df["signal"].shift(1) * df["return"]

    cumulative_strategy = (1 + df["strategy_return"]).cumprod()
    cumulative_market = (1 + df["return"]).cumprod()

    _print_stats(df, cumulative_strategy, cumulative_market)
    return cumulative_strategy, cumulative_market


def _print_stats(
    df: pd.DataFrame,
    strat: pd.Series,
    market: pd.Series,
) -> None:
    """Print a summary of backtest performance statistics."""
    trading_days = 252

    strat_ret = df["strategy_return"].dropna()
    mkt_ret = df["return"].dropna()

    strat_ann = strat_ret.mean() * trading_days
    mkt_ann = mkt_ret.mean() * trading_days
    strat_vol = strat_ret.std() * (trading_days ** 0.5)
    mkt_vol = mkt_ret.std() * (trading_days ** 0.5)
    strat_sharpe = strat_ann / strat_vol if strat_vol else float("nan")
    mkt_sharpe = mkt_ann / mkt_vol if mkt_vol else float("nan")

    strat_dd = (strat / strat.cummax() - 1).min()
    mkt_dd = (market / market.cummax() - 1).min()

    print("\n" + "=" * 50)
    print(f"{'Metric':<25} {'Strategy':>10} {'Market':>10}")
    print("-" * 50)
    print(f"{'Total Return':<25} {strat.iloc[-1] - 1:>9.1%} {market.iloc[-1] - 1:>9.1%}")
    print(f"{'Ann. Return':<25} {strat_ann:>9.1%} {mkt_ann:>9.1%}")
    print(f"{'Ann. Volatility':<25} {strat_vol:>9.1%} {mkt_vol:>9.1%}")
    print(f"{'Sharpe Ratio':<25} {strat_sharpe:>10.2f} {mkt_sharpe:>10.2f}")
    print(f"{'Max Drawdown':<25} {strat_dd:>9.1%} {mkt_dd:>9.1%}")
    print("=" * 50 + "\n")


def plot_results(
    cumulative_strategy: pd.Series,
    cumulative_market: pd.Series,
    title: str = "Strategy vs Buy & Hold",
    save_path: str | None = None,
) -> None:
    """
    Plot cumulative returns for strategy vs market.

    Args:
        cumulative_strategy: Output from backtest().
        cumulative_market:   Output from backtest().
        title:               Chart title.
        save_path:           If provided, save the figure to this path.
    """
    sns.set_theme(style="darkgrid", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(cumulative_strategy.index, cumulative_strategy.values,
            label="ML Strategy", linewidth=2, color="#2196F3")
    ax.plot(cumulative_market.index, cumulative_market.values,
            label="Buy & Hold", linewidth=1.5, linestyle="--", color="#FF9800", alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Return (×)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[backtest] Chart saved → {save_path}")

    plt.show()
