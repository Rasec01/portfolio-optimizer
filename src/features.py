import pandas as pd


def create_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Engineer features from raw Close-price data.

    Features added
    --------------
    return      – daily percentage change
    ma10        – 10-day simple moving average
    ma30        – 30-day simple moving average
    ma_ratio    – ma10 / ma30  (momentum proxy)
    volatility  – 10-day rolling std of daily returns
    target      – forward return over `horizon` trading days (regression label)

    Args:
        df:      DataFrame with a 'Close' column.
        horizon: Number of trading days to look forward for the target label.

    Returns:
        Feature-enriched DataFrame with all NaN rows removed.
    """
    df = df.copy()

    # Price-based features
    df["return"] = df["Close"].pct_change()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma30"] = df["Close"].rolling(30).mean()
    df["ma_ratio"] = df["ma10"] / df["ma30"]

    # Volatility
    df["volatility"] = df["return"].rolling(10).std()

    # Forward-return label (regression target)
    df["target"] = df["return"].shift(-horizon)

    df.dropna(inplace=True)
    print(f"[features] Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df
