import numpy as np
import pandas as pd
import pytest

from volrisk.features import make_multi_asset_vol_dataset
from volrisk.models import walk_forward_rf_forecast


def _sample_returns(periods: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "A": np.random.default_rng(1).normal(0, 0.01, len(idx)),
            "B": np.random.default_rng(2).normal(0, 0.02, len(idx)),
        },
        index=idx,
    )


def test_make_multi_asset_vol_dataset_has_asset_and_target_alignment():
    returns = _sample_returns(periods=120)

    X, y = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))

    assert len(X) == len(y)
    assert not X.empty
    assert "asset" in X.columns
    assert set(X["asset"].unique()) == {"A", "B"}
    assert y.notna().all()


def test_make_multi_asset_vol_dataset_can_keep_missing_future_targets():
    returns = _sample_returns(periods=120)

    X, y = make_multi_asset_vol_dataset(
        returns,
        target_window=5,
        lags=(1, 5, 21),
        drop_missing_target=False,
    )

    assert len(X) == len(y)
    assert X.index.max() == returns.index.max()
    assert y.isna().sum() == 10
    assert y.loc[returns.index[-1]].isna().all()


def test_realized_vol_1d_uses_abs_return_proxy():
    returns = _sample_returns(periods=80)

    X, _ = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))

    assert "realized_vol_1d" in X.columns
    assert X["realized_vol_1d"].notna().all()
    assert (X["realized_vol_1d"] >= 0).all()


def test_spike_features_exist_and_are_finite_after_enough_history():
    returns = _sample_returns(periods=120)

    X, _ = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))
    spike_columns = [
        "abs_ret_max_5d",
        "downside_vol_5d",
        "drawdown_21d",
        "realized_vol_ratio_5_21d",
        "abs_ret_shock_21d",
        "abs_ret_rank_21d",
    ]

    assert set(spike_columns).issubset(X.columns)
    assert not X[spike_columns].dropna().empty


def test_walk_forward_rf_forecast_returns_wide_positive_matrix():
    returns = _sample_returns(periods=160)
    X, y = make_multi_asset_vol_dataset(
        returns,
        target_window=5,
        lags=(1, 5, 21),
        drop_missing_target=False,
    )

    pred = walk_forward_rf_forecast(X, y, initial_train_days=60, refit_every=20, random_state=42)

    assert isinstance(pred, pd.DataFrame)
    assert set(pred.columns) == {"A", "B"}
    assert (pred > 0).all().all()
    assert len(pred) > 0
    assert pred.index.max() == returns.index.max()


def test_walk_forward_rf_forecast_accepts_legacy_initial_train_size():
    returns = _sample_returns(periods=160)
    X, y = make_multi_asset_vol_dataset(
        returns,
        target_window=5,
        lags=(1, 5, 21),
        drop_missing_target=False,
    )

    pred = walk_forward_rf_forecast(
        X,
        y,
        initial_train_size=60,
        refit_every=20,
        random_state=42,
        target_transform="raw",
    )

    assert isinstance(pred, pd.DataFrame)
    assert set(pred.columns) == {"A", "B"}
    assert (pred > 0).all().all()


def test_walk_forward_rf_forecast_rejects_unknown_target_transform():
    returns = _sample_returns(periods=120)
    X, y = make_multi_asset_vol_dataset(
        returns,
        target_window=5,
        lags=(1, 5, 21),
        drop_missing_target=False,
    )

    with pytest.raises(ValueError, match="target_transform"):
        walk_forward_rf_forecast(
            X,
            y,
            initial_train_days=40,
            refit_every=20,
            target_transform="sqrt",
        )
