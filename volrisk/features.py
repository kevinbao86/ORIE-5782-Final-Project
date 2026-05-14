from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute asset returns from price data.

    method can be 'log' or 'simple'.
    """
    if prices.empty:
        raise ValueError("prices must be non-empty")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method must be either 'log' or 'simple'")

    return returns.dropna(how="all")


def realized_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute rolling realized volatility."""
    if window <= 1:
        raise ValueError("window must be greater than 1")
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def make_vol_forecast_dataset(
    returns: pd.DataFrame,
    asset: str,
    target_window: int = 21,
    lags: tuple[int, ...] = (1, 5, 21, 63),
) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised learning data for next-period volatility forecasting.

    The target is forward realized volatility over the next `target_window`
    trading days. Features are based only on information available at the
    current date, so the resulting dataset can be used in walk-forward testing.
    """
    if asset not in returns.columns:
        raise ValueError(f"{asset!r} not found in returns columns")

    r = returns[asset].dropna()
    X = pd.DataFrame(index=r.index)
    X["ret_1d"] = r
    X["abs_ret_1d"] = r.abs()

    for lag in lags:
        X[f"abs_ret_mean_{lag}d"] = r.abs().rolling(lag).mean()
        X[f"return_mean_{lag}d"] = r.rolling(lag).mean()

        if lag == 1:
            X[f"realized_vol_{lag}d"] = r.abs() * np.sqrt(TRADING_DAYS)
        else:
            X[f"realized_vol_{lag}d"] = r.rolling(lag).std() * np.sqrt(TRADING_DAYS)

    # Forward-looking target. At date t this measures realized volatility over
    # the next target_window returns. It is never used as a feature.
    y = r.rolling(target_window).std().shift(-target_window) * np.sqrt(TRADING_DAYS)
    y.name = "target_vol"

    data = X.join(y).dropna(subset=["target_vol"])
    return data.drop(columns=["target_vol"]), data["target_vol"]


def make_multi_asset_vol_dataset(
    returns: pd.DataFrame,
    target_window: int = 21,
    lags: tuple[int, ...] = (1, 5, 21, 63),
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack per-asset volatility datasets into one panel dataset.

    The target is attached before concatenation to avoid pandas alignment
    issues from duplicate date indices across assets.
    """
    parts = []

    for asset in returns.columns:
        X_asset, y_asset = make_vol_forecast_dataset(
            returns, asset=asset, target_window=target_window, lags=lags
        )
        asset_data = X_asset.copy()
        asset_data["asset"] = asset
        asset_data["target_vol"] = y_asset.to_numpy()
        parts.append(asset_data)

    panel = pd.concat(parts).sort_index()
    X = panel.drop(columns=["target_vol"])
    y = panel["target_vol"]
    return X, y
