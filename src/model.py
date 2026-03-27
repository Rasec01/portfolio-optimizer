from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

FEATURE_COLS = ["ma10", "ma30", "ma_ratio", "volatility"]


def train_model(
    df: pd.DataFrame,
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict]:
    """
    Train a Random Forest regressor to predict forward returns.

    Args:
        df:           Feature DataFrame produced by create_features().
        n_estimators: Number of trees in the forest.
        test_size:    Fraction of data reserved for out-of-sample evaluation.
        random_state: Seed for reproducibility.

    Returns:
        (model, metrics) where metrics is a dict with R², RMSE, and split info.
    """
    # Use only columns that exist in the dataframe
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df["target"]

    # Time-series split — no shuffling to avoid look-ahead bias
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics = {
        "r2": round(r2, 4),
        "rmse": round(rmse, 6),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": feature_cols,
    }

    print(f"[model] R²={metrics['r2']}  RMSE={metrics['rmse']}")
    print(f"[model] Train: {metrics['train_rows']} rows | Test: {metrics['test_rows']} rows")
    return model, metrics
