import numpy as np
import pandas as pd
import pytest

from volrisk.portfolio import equal_weight, inverse_vol_weights, normalize_weights


def test_equal_weight_rows_sum_to_one():
    idx = pd.date_range("2020-01-01", periods=3)
    w = equal_weight(idx, ["A", "B", "C"])
    assert np.allclose(w.sum(axis=1), 1.0)


def test_inverse_vol_weights_more_weight_to_lower_vol():
    vol = pd.DataFrame(
        {"LowVol": [0.1], "HighVol": [0.3]},
        index=pd.date_range("2020-01-01", periods=1),
    )
    w = inverse_vol_weights(vol)
    assert w["LowVol"].iloc[0] > w["HighVol"].iloc[0]
    assert np.isclose(w.sum(axis=1).iloc[0], 1.0)


def test_normalize_weights_rejects_negative():
    raw = pd.DataFrame({"A": [1.0], "B": [-1.0]})
    with pytest.raises(ValueError):
        normalize_weights(raw)
