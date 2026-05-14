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
            "abs_ret_max_1d",
            "abs_ret_rank_1d",
            "abs_ret_shock_1d",
            "abs_ret_zscore_1d",
            "drawdown_1d",
            "downside_vol_1d",
            "abs_ret_mean_5d",
            "realized_vol_5d",
            "return_mean_5d",
            "abs_ret_max_5d",
            "abs_ret_rank_5d",
            "abs_ret_shock_5d",
            "abs_ret_zscore_5d",
            "drawdown_5d",
            "downside_vol_5d",
            "abs_ret_mean_21d",
            "realized_vol_21d",
            "return_mean_21d",
            "abs_ret_max_21d",
            "abs_ret_rank_21d",
            "abs_ret_shock_21d",
            "abs_ret_zscore_21d",
            "drawdown_21d",
            "downside_vol_21d",
            "abs_ret_mean_63d",
            "realized_vol_63d",
            "return_mean_63d",
            "abs_ret_max_63d",
            "abs_ret_rank_63d",
            "abs_ret_shock_63d",
            "abs_ret_zscore_63d",
            "drawdown_63d",
            "downside_vol_63d",
            "realized_vol_ratio_1_21d",
            "realized_vol_ratio_5_21d",
            "realized_vol_ratio_21_63d",
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
    target_transform: str = "log",
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
    if target_transform not in {"log", "raw"}:
        raise ValueError("target_transform must be either 'log' or 'raw'")

    data = X.copy()
    data["_target"] = y.to_numpy()
    data = data.sort_index()

    unique_dates = pd.Index(sorted(data.index.unique()))
    known_target_dates = pd.Index(sorted(data.loc[data["_target"].notna()].index.unique()))
    if len(known_target_dates) < initial_train_days:
        raise ValueError(
            "Not enough unique dates for the requested initial_train_days. "
            f"Got {len(known_target_dates)}, requested {initial_train_days}."
        )

    pred_frames = []
    fitted_model: Pipeline | None = None
    forecast_dates = [
        date for date in unique_dates if (known_target_dates < date).sum() >= initial_train_days
    ]
    if not forecast_dates:
        raise ValueError("No forecast dates available after the requested initial_train_days")

    for step, date in enumerate(forecast_dates):
        if fitted_model is None or step % refit_every == 0:
            train = data.loc[(data.index < date) & data["_target"].notna()]
            numeric_features = [
                column
                for column in train.columns
                if column not in {"asset", "_target"}
                and pd.api.types.is_numeric_dtype(train[column])
            ]
            fitted_model = build_random_forest_model(
                random_state=random_state,
                numeric_features=numeric_features,
            )
            train_target = train["_target"].clip(lower=1e-4)
            if target_transform == "log":
                train_target = np.log(train_target)
            fitted_model.fit(train.drop(columns=["_target"]), train_target)

        test = data.loc[[date]].drop(columns=["_target"])
        pred_values = fitted_model.predict(test)
        if target_transform == "log":
            pred_values = np.exp(pred_values)

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
