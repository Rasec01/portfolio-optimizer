import yfinance as yf
import pandas as pd


def download_data(ticker: str = "SPY", start: str = "2015-01-01") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.

    Args:
        ticker: Stock/ETF ticker symbol (default: SPY)
        start:  Start date string in YYYY-MM-DD format (default: 2015-01-01)

    Returns:
        DataFrame with a single 'Close' column, date-indexed, no NaNs.
    """
    print(f"[data_loader] Downloading {ticker} from {start} ...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    df = df[["Close"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    print(f"[data_loader] Downloaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df
