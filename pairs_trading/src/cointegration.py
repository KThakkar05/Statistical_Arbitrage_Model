"""Cointegration analysis for pairs trading.

Covers price loading, OLS hedge-ratio estimation, ADF stationarity testing,
and multi-pair screening across a universe of tickers.
"""
from __future__ import annotations

import itertools

import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller


def load_prices(ticker_1: str, ticker_2: str, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """Download adjusted closing prices for two tickers over a date range.

    Args:
        ticker_1: Yahoo Finance symbol for the dependent series (e.g. ``"RIO.L"``).
        ticker_2: Yahoo Finance symbol for the independent series (e.g. ``"BHP.L"``).
        start: Start date string in ``"YYYY-MM-DD"`` format.
        end: End date string in ``"YYYY-MM-DD"`` format.

    Returns:
        Tuple of ``(series_1, series_2)`` — two ``pd.Series`` of daily close
        prices aligned on the same dates, with any rows containing NaN dropped.
    """
    raw = yf.download([ticker_1, ticker_2], start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    return prices[ticker_1], prices[ticker_2]


def fit_ols(series_y: pd.Series, series_x: pd.Series):
    """Estimate the OLS hedge ratio between two price series.

    Args:
        series_y: Dependent price series.
        series_x: Independent price series. Its ``.name`` is used to look up
            the fitted coefficient.

    Returns:
        Tuple of ``(alpha, beta, model)`` — intercept, hedge ratio, and the
        fitted ``statsmodels.OLSResults`` object.
    """
    model = OLS(series_y, add_constant(series_x)).fit()
    alpha = model.params["const"]
    beta = model.params[series_x.name]
    return alpha, beta, model


def calc_spread(series_y: pd.Series, series_x: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Compute the residual spread: ``series_y - (beta * series_x + alpha)``.

    Args:
        series_y: Dependent price series.
        series_x: Independent price series.
        alpha: OLS intercept from ``fit_ols``.
        beta: OLS hedge ratio from ``fit_ols``.

    Returns:
        ``pd.Series`` of residual spread values, named ``"Spread"``.
    """
    spread = series_y - (beta * series_x + alpha)
    spread.name = "Spread"
    return spread


def adf_test(spread: pd.Series) -> dict:
    """Run an Augmented Dickey-Fuller test on a spread series.

    Args:
        spread: Residual spread series from ``calc_spread``.

    Returns:
        Dict with keys ``adf_stat``, ``p_value``, ``lags``, ``nobs``,
        ``crit_values`` and ``icbest``.
    """
    adf_stat, p_value, lags, nobs, crit_values, icbest = adfuller(spread.dropna())
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "lags": lags,
        "nobs": nobs,
        "crit_values": crit_values,
        "icbest": icbest,
    }


def screen_pair(ticker_1: str, ticker_2: str, start: str, end: str) -> dict:
    """Download, fit, and cointegration-test a single candidate pair.

    Convenience wrapper chaining ``load_prices -> fit_ols -> calc_spread ->
    adf_test``. Useful for quickly checking one pair before committing it to
    a backtest.

    Args:
        ticker_1: Dependent ticker.
        ticker_2: Independent ticker.
        start: Start date string ``"YYYY-MM-DD"``.
        end: End date string ``"YYYY-MM-DD"``.

    Returns:
        Dict summarizing the pair's cointegration diagnostics: ``ticker_1``,
        ``ticker_2``, ``alpha``, ``beta``, ``n_obs``, plus all keys from
        ``adf_test``.
    """
    s1, s2 = load_prices(ticker_1, ticker_2, start, end)
    alpha, beta, _ = fit_ols(s1, s2)
    spread = calc_spread(s1, s2, alpha, beta)
    result = adf_test(spread)
    result.update({
        "ticker_1": ticker_1,
        "ticker_2": ticker_2,
        "alpha": alpha,
        "beta": beta,
        "n_obs": len(spread),
    })
    return result


def scan_pairs(tickers: list[str], start: str, end: str, min_rows: int = 200) -> pd.DataFrame:
    """Screen every combination of tickers for cointegration, ranked by ADF statistic.

    Downloads all tickers in a single batch request, then fits an OLS hedge
    ratio and ADF test for every pairwise combination. More negative ADF
    statistics indicate stronger evidence of a stationary (mean-reverting)
    spread.

    Args:
        tickers: List of Yahoo Finance ticker symbols to test pairwise.
        start: Start date string ``"YYYY-MM-DD"``.
        end: End date string ``"YYYY-MM-DD"``.
        min_rows: Minimum overlapping trading days required to test a pair.

    Returns:
        DataFrame sorted by ADF statistic ascending (best first), with columns
        ``rank``, ``ticker_1``, ``ticker_2``, ``adf_stat``, ``p_value``,
        ``beta``, ``n_obs``, ``stationary_5pct``, ``stationary_1pct``. Empty
        DataFrame if no pair had enough overlapping data.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty or "Close" not in raw.columns:
        return pd.DataFrame()
    prices = raw["Close"]

    results = []
    for t1, t2 in itertools.combinations(tickers, 2):
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        pair_px = prices[[t1, t2]].dropna()
        if len(pair_px) < min_rows:
            continue
        try:
            alpha, beta, _ = fit_ols(pair_px[t1], pair_px[t2])
            spread = calc_spread(pair_px[t1], pair_px[t2], alpha, beta)
            adf = adf_test(spread)
        except Exception:
            continue
        results.append({
            "ticker_1": t1,
            "ticker_2": t2,
            "adf_stat": adf["adf_stat"],
            "p_value": adf["p_value"],
            "beta": beta,
            "n_obs": len(pair_px),
            "stationary_5pct": adf["p_value"] < 0.05,
            "stationary_1pct": adf["p_value"] < 0.01,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("adf_stat").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
