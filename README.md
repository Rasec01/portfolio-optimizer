# Financial Market Prediction & Backtesting System

A modular Python pipeline that downloads historical price data, engineers technical features, trains a Random Forest regressor to predict forward returns, converts predictions into long/short signals, and runs a vectorised backtest.

---

## Quickstart

```bash
# 1. Clone / unzip the project
cd market-prediction-system

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (SPY, 2015-present)
python main.py

# 5. Optional flags
python main.py --ticker QQQ --start 2018-01-01 --plot
python main.py --ticker AAPL --plot --save-chart data/aapl_backtest.png
```

---

## CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | Ticker symbol to download |
| `--start` | `2015-01-01` | History start date (YYYY-MM-DD) |
| `--horizon` | `5` | Forward-return target in trading days |
| `--threshold` | `0.0` | Minimum predicted return to go long |
| `--plot` | off | Show cumulative-return chart |
| `--save-chart` | `None` | Save chart to given path |

---

## Project Structure

```
market-prediction-system/
│
├── data/                      # Output directory (charts, cached data)
├── notebooks/
│   └── exploration.ipynb      # Interactive walkthrough
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # yfinance download + validation
│   ├── features.py            # Technical indicator engineering
│   ├── model.py               # Random Forest training + metrics
│   ├── signals.py             # Convert predictions → long/short signals
│   └── backtest.py            # Vectorised P&L + performance stats + chart
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
download_data()
      │
      ▼
create_features()   ←  returns, MAs, MA ratio, volatility, target
      │
      ▼
train_model()       ←  Random Forest, time-series train/test split
      │
      ▼
generate_signals()  ←  prediction > threshold → long, else short
      │
      ▼
backtest()          ←  lagged signal × daily return, cumulative P&L
```

### Features

| Column | Description |
|---|---|
| `return` | Daily percentage change |
| `ma10` | 10-day simple moving average |
| `ma30` | 30-day simple moving average |
| `ma_ratio` | `ma10 / ma30` — momentum proxy |
| `volatility` | 10-day rolling std of daily returns |
| `target` | Forward return over `--horizon` days (label) |

### Backtest assumptions

- Signals are **lagged by one day** to prevent look-ahead bias.
- Costs (slippage, commission) are **not modelled** — results are indicative only.
- Short signals represent a short position, not staying flat.

---

## Notebook

Open `notebooks/exploration.ipynb` for an interactive walkthrough including correlation heatmaps and feature-importance charts:

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Extending the System

- **More features**: add RSI, Bollinger Bands, volume ratios in `src/features.py`
- **Different models**: swap `RandomForestRegressor` in `src/model.py` for XGBoost, LSTM, etc.
- **Transaction costs**: subtract a per-trade cost in `src/backtest.py`
- **Position sizing**: replace binary ±1 signals with Kelly-sized positions

---

## Disclaimer

This project is for **educational purposes only**. Past backtest performance does not guarantee future results. Do not use this system for live trading without independent validation and risk management.
