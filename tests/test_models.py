import numpy as np
import pandas as pd

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


def test_realized_vol_1d_uses_abs_return_proxy():
    returns = _sample_returns(periods=80)

    X, _ = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))

    assert "realized_vol_1d" in X.columns
    assert X["realized_vol_1d"].notna().all()
    assert (X["realized_vol_1d"] >= 0).all()


def test_walk_forward_rf_forecast_returns_wide_positive_matrix():
    returns = _sample_returns(periods=160)
    X, y = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))

    pred = walk_forward_rf_forecast(X, y, initial_train_days=60, refit_every=20, random_state=42)

    assert isinstance(pred, pd.DataFrame)
    assert set(pred.columns) == {"A", "B"}
    assert (pred > 0).all().all()
    assert len(pred) > 0


def test_walk_forward_rf_forecast_accepts_legacy_initial_train_size():
    returns = _sample_returns(periods=160)
    X, y = make_multi_asset_vol_dataset(returns, target_window=5, lags=(1, 5, 21))

    pred = walk_forward_rf_forecast(X, y, initial_train_size=60, refit_every=20, random_state=42)

    assert isinstance(pred, pd.DataFrame)
    assert set(pred.columns) == {"A", "B"}
    assert (pred > 0).all().all()
