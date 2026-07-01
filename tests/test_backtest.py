"""Tests for src/backtest.py using a synthetic, deterministic spread.

We set series_x to a constant and alpha=beta=0 so that
``calc_spread(series_y, series_x, 0, 0) == series_y`` exactly. This lets us
hand-craft the spread the backtest sees (flat noise, then a sharp jump, then
a reversion) without needing real price data or network calls.
"""
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.metrics import perf_metrics


def _synthetic_spread_prices(n=300, window=20):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    spread = rng.normal(0, 0.2, n)          # small baseline noise
    spread[100:130] += 8.0                  # sharp excursion -> should trigger a short entry
    spread[200:230] -= 8.0                  # sharp excursion -> should trigger a long entry

    series_y = pd.Series(spread, index=dates, name="Y")
    series_x = pd.Series(1.0, index=dates, name="X")  # constant -> beta has no effect
    return series_y, series_x


def test_run_backtest_output_shape_and_columns():
    series_y, series_x = _synthetic_spread_prices()
    pnl_df, trade_log = run_backtest(series_y, series_x, alpha=0.0, beta=0.0, window=20)

    expected_cols = {"ZScore", "Spread", "Position", "DailyPnL",
                      "Slippage", "GrossPnL", "CumulativePnL", "GrossCumPnL"}
    assert expected_cols.issubset(pnl_df.columns)
    assert len(pnl_df) == len(series_y)
    assert set(pnl_df["Position"].unique()).issubset({-1, 0, 1})


def test_run_backtest_cumulative_pnl_matches_daily_cumsum():
    series_y, series_x = _synthetic_spread_prices()
    pnl_df, _ = run_backtest(series_y, series_x, alpha=0.0, beta=0.0, window=20)

    np.testing.assert_allclose(
        pnl_df["CumulativePnL"].values,
        pnl_df["DailyPnL"].cumsum().values,
    )


def test_run_backtest_produces_trades_on_sharp_excursions():
    series_y, series_x = _synthetic_spread_prices()
    pnl_df, trade_log = run_backtest(series_y, series_x, alpha=0.0, beta=0.0, window=20,
                                      entry_z=2.0, exit_z=0.0)

    assert len(trade_log) >= 1
    # Each trade's P&L should equal (exit - entry) signed by direction.
    sign = trade_log["Direction"].map({"Long": 1, "Short": -1})
    expected_pnl = (trade_log["Exit_Spread"] - trade_log["Entry_Spread"]) * sign
    np.testing.assert_allclose(trade_log["Profit/Loss"].values, expected_pnl.values)


def test_perf_metrics_runs_on_backtest_output():
    series_y, series_x = _synthetic_spread_prices()
    pnl_df, trade_log = run_backtest(series_y, series_x, alpha=0.0, beta=0.0, window=20)

    metrics = perf_metrics(pnl_df, trade_log, cost=True)

    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert metrics["n_trades"] == len(trade_log)
