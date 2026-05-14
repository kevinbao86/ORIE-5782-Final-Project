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
    drop_missing_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised learning data for next-period volatility forecasting.

    The target is forward realized volatility over the next `target_window`
    trading days. Features are based only on information available at the
    current date, so the resulting dataset can be used in walk-forward testing.
    """
    if asset not in returns.columns:
        raise ValueError(f"{asset!r} not found in returns columns")

    r = returns[asset].dropna()
    abs_r = r.abs()
    wealth = (1 + r).cumprod()

    X = pd.DataFrame(index=r.index)
    X["ret_1d"] = r
    X["abs_ret_1d"] = abs_r

    for lag in lags:
        abs_mean = abs_r.rolling(lag).mean()
        abs_std = abs_r.rolling(lag).std()
        rolling_vol = r.rolling(lag).std() * np.sqrt(TRADING_DAYS)

        X[f"abs_ret_mean_{lag}d"] = abs_mean
        X[f"return_mean_{lag}d"] = r.rolling(lag).mean()
        X[f"abs_ret_max_{lag}d"] = abs_r.rolling(lag).max()
        X[f"abs_ret_rank_{lag}d"] = abs_r.rolling(lag).apply(
            lambda values: float(np.mean(values <= values[-1])),
            raw=True,
        )
        X[f"abs_ret_shock_{lag}d"] = (
            abs_r >= abs_r.rolling(lag).quantile(0.95)
        ).astype(float)
        X[f"drawdown_{lag}d"] = wealth / wealth.rolling(lag).max() - 1

        if lag == 1:
            X[f"abs_ret_zscore_{lag}d"] = 0.0
            X[f"realized_vol_{lag}d"] = r.abs() * np.sqrt(TRADING_DAYS)
            X[f"downside_vol_{lag}d"] = r.clip(upper=0).abs() * np.sqrt(TRADING_DAYS)
        else:
            X[f"abs_ret_zscore_{lag}d"] = (abs_r - abs_mean) / abs_std
            X[f"realized_vol_{lag}d"] = rolling_vol
            X[f"downside_vol_{lag}d"] = r.clip(upper=0).rolling(lag).std() * np.sqrt(
                TRADING_DAYS
            )

    if "realized_vol_5d" in X.columns and "realized_vol_21d" in X.columns:
        X["realized_vol_ratio_5_21d"] = X["realized_vol_5d"] / X["realized_vol_21d"]
    if "realized_vol_21d" in X.columns and "realized_vol_63d" in X.columns:
        X["realized_vol_ratio_21_63d"] = X["realized_vol_21d"] / X["realized_vol_63d"]
    if "realized_vol_1d" in X.columns and "realized_vol_21d" in X.columns:
        X["realized_vol_ratio_1_21d"] = X["realized_vol_1d"] / X["realized_vol_21d"]

    X = X.replace([np.inf, -np.inf], np.nan)

    # Forward-looking target. At date t this measures realized volatility over
    # the next target_window returns. It is never used as a feature.
    y = r.rolling(target_window).std().shift(-target_window) * np.sqrt(TRADING_DAYS)
    y.name = "target_vol"

    data = X.join(y)
    if drop_missing_target:
        data = data.dropna(subset=["target_vol"])
    return data.drop(columns=["target_vol"]), data["target_vol"]


def make_multi_asset_vol_dataset(
    returns: pd.DataFrame,
    target_window: int = 21,
    lags: tuple[int, ...] = (1, 5, 21, 63),
    drop_missing_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack per-asset volatility datasets into one panel dataset.

    The target is attached before concatenation to avoid pandas alignment
    issues from duplicate date indices across assets.
    """
    parts = []

    for asset in returns.columns:
        X_asset, y_asset = make_vol_forecast_dataset(
            returns,
            asset=asset,
            target_window=target_window,
            lags=lags,
            drop_missing_target=drop_missing_target,
        )
        asset_data = X_asset.copy()
        asset_data["asset"] = asset
        asset_data["target_vol"] = y_asset.to_numpy()
        parts.append(asset_data)

    panel = pd.concat(parts).sort_index()
    X = panel.drop(columns=["target_vol"])
    y = panel["target_vol"]
    return X, y
