from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


@dataclass
class VolForecastResult:
    predictions: pd.DataFrame
    model_name: str


def historical_vol_forecast(
    returns: pd.DataFrame,
    lookback: int = 21,
) -> pd.DataFrame:
    """Use recent realized volatility as the next-period volatility forecast."""
    return returns.rolling(lookback).std() * np.sqrt(252)


def ewma_vol_forecast(
    returns: pd.DataFrame,
    span: int = 21,
) -> pd.DataFrame:
    """Exponentially weighted volatility forecast."""
    return returns.ewm(span=span, adjust=False).std() * np.sqrt(252)


def build_random_forest_model(
    random_state: int = 42,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    """Build a Random Forest volatility forecasting pipeline."""
    if numeric_features is None:
        numeric_features = [
            "ret_1d",
            "abs_ret_1d",
            "abs_ret_mean_1d",
            "realized_vol_1d",
            "return_mean_1d",
            "abs_ret_mean_5d",
            "realized_vol_5d",
            "return_mean_5d",
            "abs_ret_mean_21d",
            "realized_vol_21d",
            "return_mean_21d",
            "abs_ret_mean_63d",
            "realized_vol_63d",
            "return_mean_63d",
        ]
    categorical_features = ["asset"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def walk_forward_rf_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_days: int | None = None,
    refit_every: int = 21,
    random_state: int = 42,
    *,
    initial_train_size: int | None = None,
) -> pd.DataFrame:
    """Generate walk-forward Random Forest volatility forecasts.

    The model is trained only on dates strictly before the forecast date. On
    each forecast date, it predicts volatility for every asset with available
    features, then returns a wide Date x Asset forecast matrix.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if (
        initial_train_days is not None
        and initial_train_size is not None
        and initial_train_days != initial_train_size
    ):
        raise ValueError("initial_train_days and initial_train_size must match when both are set")

    if initial_train_days is None:
        initial_train_days = initial_train_size if initial_train_size is not None else 756

    if initial_train_days <= 0:
        raise ValueError("initial_train_days must be positive")
    if refit_every <= 0:
        raise ValueError("refit_every must be positive")
    if "asset" not in X.columns:
        raise ValueError("X must contain an 'asset' column")

    data = X.copy()
    data["_target"] = y.to_numpy()
    data = data.dropna(subset=["_target"]).sort_index()

    unique_dates = pd.Index(sorted(data.index.unique()))
    if len(unique_dates) <= initial_train_days:
        raise ValueError(
            "Not enough unique dates for the requested initial_train_days. "
            f"Got {len(unique_dates)}, requested {initial_train_days}."
        )

    pred_frames = []
    fitted_model: Pipeline | None = None
    forecast_dates = unique_dates[initial_train_days:]

    for step, date in enumerate(forecast_dates):
        if fitted_model is None or step % refit_every == 0:
            train_dates = unique_dates[: initial_train_days + step]
            train = data.loc[data.index.isin(train_dates)]
            numeric_features = [
                column for column in train.columns if column not in {"asset", "_target"}
            ]
            fitted_model = build_random_forest_model(
                random_state=random_state,
                numeric_features=numeric_features,
            )
            fitted_model.fit(train.drop(columns=["_target"]), train["_target"])

        test = data.loc[[date]].drop(columns=["_target"])
        pred_values = fitted_model.predict(test)

        pred_frames.append(
            pd.DataFrame(
                {
                    "Date": [date] * len(test),
                    "asset": test["asset"].to_numpy(),
                    "pred_vol": pred_values,
                }
            )
        )

    pred_long = pd.concat(pred_frames, ignore_index=True)
    pred_wide = pred_long.pivot(index="Date", columns="asset", values="pred_vol")
    pred_wide.index = pd.to_datetime(pred_wide.index)
    pred_wide = pred_wide.sort_index()
    return pred_wide.clip(lower=1e-4)
