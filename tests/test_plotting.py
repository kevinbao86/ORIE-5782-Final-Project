import pandas as pd

from volrisk.plotting import plot_cumulative_returns_common_start


def test_plot_cumulative_returns_common_start_writes_file(tmp_path):
    early_index = pd.date_range("2020-01-01", periods=4, freq="D")
    late_index = pd.date_range("2020-01-03", periods=2, freq="D")
    results = {
        "early": pd.DataFrame({"net_return": [0.01, 0.02, -0.01, 0.01]}, index=early_index),
        "late": pd.DataFrame({"net_return": [0.03, 0.01]}, index=late_index),
    }
    out_path = tmp_path / "common_start.png"

    plot_cumulative_returns_common_start(results, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
