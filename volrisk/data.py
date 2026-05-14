from __future__ import annotations

from pathlib import Path
from io import StringIO
import os

import pandas as pd


def download_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
    source: str = "stooq",
    verify_ssl: bool = False,
    stooq_api_key: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices.

    Uses Yahoo Finance by default. If Yahoo fails due to SSL/network issues,
    falls back to Stooq.
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    if source not in {"yahoo", "stooq"}:
        raise ValueError("source must be either 'yahoo' or 'stooq'")

    if source == "yahoo":
        try:
            return _download_prices_yahoo(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                verify_ssl=verify_ssl,
            )
        except Exception as exc:
            print(f"Yahoo download failed: {exc}")
            print("Falling back to Stooq...")
            return _download_prices_stooq(
                tickers=tickers,
                start=start,
                end=end,
                verify_ssl=verify_ssl,
                api_key=stooq_api_key,
            )

    return _download_prices_stooq(
        tickers=tickers,
        start=start,
        end=end,
        verify_ssl=verify_ssl,
        api_key=stooq_api_key,
    )


def _download_prices_yahoo(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
    verify_ssl: bool = False,
) -> pd.DataFrame:
    """Download prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        raise ImportError("Install yfinance and curl_cffi to download Yahoo data.") from exc

    session = curl_requests.Session(verify=verify_ssl)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        session=session,
        threads=False,
    )

    if raw.empty:
        raise ValueError("No data returned from Yahoo.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_field = "Close" if auto_adjust else "Adj Close"
        if price_field not in raw.columns.get_level_values(0):
            price_field = "Close"
        prices = raw[price_field].copy()
    else:
        col = "Close" if auto_adjust else ("Adj Close" if "Adj Close" in raw.columns else "Close")
        prices = raw[[col]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all").ffill().dropna(how="any")

    if prices.empty:
        raise ValueError("Yahoo returned data, but no usable prices remained after cleaning.")

    return prices


def _download_prices_stooq(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    verify_ssl: bool = False,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download prices directly from Stooq CSV endpoints.

    This avoids pandas-datareader, which can break with newer pandas versions.
    Stooq commonly uses symbols like 'SPY.US', 'QQQ.US', etc.
    """
    import requests
    import urllib3

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    api_key = api_key or os.getenv("STOOQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Stooq now requires an API key for CSV downloads. "
            "Set the STOOQ_API_KEY environment variable or pass stooq_api_key=... "
            "to download_prices()."
        )

    prices = {}
    errors = {}

    start_clean = start.replace("-", "") if start is not None else "19000101"
    end_clean = end.replace("-", "") if end is not None else "20991231"

    for ticker in tickers:
        stooq_symbol = f"{ticker.lower()}.us"
        url = (
            "https://stooq.com/q/d/l/"
            f"?s={stooq_symbol}&d1={start_clean}&d2={end_clean}&i=d&apikey={api_key}"
        )

        try:
            response = requests.get(url, timeout=30, verify=verify_ssl)
            response.raise_for_status()

            text = response.text.strip()
            if not text or text.lower() == "no data":
                raise ValueError("empty response")

            # Stooq usually returns a clean CSV beginning with
            # Date,Open,High,Low,Close,Volume. Under some network filters,
            # a few preamble lines can be inserted before the CSV body.
            lines = text.splitlines()
            header_idx = next(
                (i for i, line in enumerate(lines) if line.strip().lower().startswith("date,")),
                None,
            )
            if header_idx is None:
                snippet = text[:500].replace("\n", " ")
                raise ValueError(
                    f"Stooq response did not contain a CSV header for {ticker}. "
                    f"First 500 chars: {snippet}"
                )

            csv_text = "\n".join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_text), engine="python")
            if df.empty or "Date" not in df.columns or "Close" not in df.columns:
                raise ValueError(f"unexpected Stooq response for {ticker}: {text[:200]}")

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            prices[ticker] = df["Close"].astype(float)
        except Exception as exc:
            errors[ticker] = str(exc)

    if not prices:
        raise ValueError(f"No Stooq data returned. Errors: {errors}")

    if errors:
        print(f"Warning: Stooq failed for some tickers and they will be omitted: {errors}")

    price_df = pd.DataFrame(prices)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index().dropna(how="all").ffill().dropna(how="any")

    if price_df.empty:
        raise ValueError("No usable Stooq prices remained after cleaning.")

    return price_df

# def download_prices(
#     tickers: list[str],
#     start: str = "2015-01-01",
#     end: str | None = None,
#     auto_adjust: bool = True,
# ) -> pd.DataFrame:
#     """Download adjusted close prices from Yahoo Finance via yfinance.

#     Parameters
#     ----------
#     tickers:
#         List of ticker symbols.
#     start:
#         Start date in YYYY-MM-DD format.
#     end:
#         Optional end date in YYYY-MM-DD format.
#     auto_adjust:
#         Whether yfinance should adjust prices for splits/dividends.

#     Returns
#     -------
#     pd.DataFrame
#         Date-indexed price matrix with one column per ticker.
#     """
#     if not tickers:
#         raise ValueError("tickers must be non-empty")

#     try:
#         import yfinance as yf
#     except ImportError as exc:
#         raise ImportError("Install yfinance to download data: pip install yfinance") from exc

#     raw = yf.download(
#         tickers=tickers,
#         start=start,
#         end=end,
#         auto_adjust=auto_adjust,
#         progress=False,
#         group_by="column",
#     )

#     if raw.empty:
#         raise ValueError("No data returned. Check tickers and date range.")

#     if isinstance(raw.columns, pd.MultiIndex):
#         if auto_adjust:
#             price_field = "Close"
#         else:
#             price_field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
#         prices = raw[price_field].copy()
#     else:
#         col = "Close" if auto_adjust else ("Adj Close" if "Adj Close" in raw.columns else "Close")
#         prices = raw[[col]].copy()
#         prices.columns = tickers

#     prices.index = pd.to_datetime(prices.index)
#     prices = prices.sort_index().dropna(how="all")
#     prices = prices.ffill().dropna(how="any")
#     return prices


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    """Save price matrix to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a price matrix from CSV."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if df.empty:
        raise ValueError("Loaded price file is empty")
    return df.sort_index()
