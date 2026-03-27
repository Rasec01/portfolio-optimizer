from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.model import FEATURE_COLS


def generate_signals(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Use the trained model to produce directional trading signals.

    Signal encoding
    ---------------
     1 → predicted return > threshold  → go long
    -1 → predicted return ≤ threshold  → go short (or stay flat)

    Args:
        df:        Feature DataFrame (must contain FEATURE_COLS).
        model:     Fitted RandomForestRegressor from train_model().
        threshold: Minimum predicted return to trigger a long signal.

    Returns:
        DataFrame with two new columns: 'prediction' and 'signal'.
    """
    df = df.copy()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]

    df["prediction"] = model.predict(X)
    df["signal"] = np.where(df["prediction"] > threshold, 1, -1)

    long_pct = (df["signal"] == 1).mean() * 100
    print(f"[signals] Long signals: {long_pct:.1f}%  |  Short signals: {100 - long_pct:.1f}%")
    return df
