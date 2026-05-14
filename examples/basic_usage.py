from volrisk.data import download_prices
from volrisk.features import compute_returns, realized_volatility
from volrisk.portfolio import inverse_vol_weights

prices = download_prices(["SPY", "TLT", "GLD"], start="2020-01-01")
returns = compute_returns(prices)
vol = realized_volatility(returns, window=21)
weights = inverse_vol_weights(vol)

print(weights.tail())
