import numpy as np
import pandas as pd
import pytest

from volrisk.features import compute_returns, realized_volatility


def test_compute_simple_returns():
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.date_range("2020-01-01", periods=3))
    returns = compute_returns(prices, method="simple")
    assert np.isclose(returns["A"].iloc[0], 0.10)
    assert np.isclose(returns["A"].iloc[1], 0.10)


def test_compute_log_returns():
    prices = pd.DataFrame({"A": [100, 110]}, index=pd.date_range("2020-01-01", periods=2))
    returns = compute_returns(prices, method="log")
    assert np.isclose(returns["A"].iloc[0], np.log(1.1))


def test_compute_returns_rejects_bad_method():
    prices = pd.DataFrame({"A": [100, 101]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(ValueError):
        compute_returns(prices, method="bad")


def test_realized_volatility_shape():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.00]}, index=pd.date_range("2020-01-01", periods=4))
    vol = realized_volatility(returns, window=2)
    assert vol.shape == returns.shape
    assert vol["A"].notna().sum() == 3
