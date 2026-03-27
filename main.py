"""
Financial Market Prediction & Backtesting System
=================================================
Entry point — runs the full pipeline end-to-end.

Usage
-----
    python main.py
    python main.py --ticker QQQ --start 2018-01-01 --plot
"""

import argparse
import os

from src.data_loader import download_data
from src.features import create_features
from src.model import train_model
from src.signals import generate_signals
from src.backtest import backtest, plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market Prediction & Backtesting System")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-return horizon in trading days")
    parser.add_argument("--threshold", type=float, default=0.0, help="Signal threshold")
    parser.add_argument("--plot", action="store_true", help="Show cumulative-return chart")
    parser.add_argument("--save-chart", default=None, help="Save chart to this file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n========================================")
    print(" Market Prediction & Backtesting System ")
    print("========================================\n")

    # 1. Data
    df = download_data(ticker=args.ticker, start=args.start)

    # 2. Features
    df = create_features(df, horizon=args.horizon)

    # 3. Model
    model, metrics = train_model(df)
    print(f"\nModel R²  : {metrics['r2']}")
    print(f"Model RMSE: {metrics['rmse']}")

    # 4. Signals
    df = generate_signals(df, model, threshold=args.threshold)

    # 5. Backtest
    strategy, market = backtest(df)

    # 6. Optional chart
    if args.plot or args.save_chart:
        save_path = args.save_chart or os.path.join("data", f"{args.ticker}_backtest.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_results(
            strategy,
            market,
            title=f"{args.ticker} — ML Strategy vs Buy & Hold",
            save_path=save_path if args.save_chart else None,
        )
        if args.plot and not args.save_chart:
            plot_results(strategy, market, title=f"{args.ticker} — ML Strategy vs Buy & Hold")

    print("\nPipeline complete ✓")


if __name__ == "__main__":
    main()
