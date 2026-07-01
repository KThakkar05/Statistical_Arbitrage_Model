"""Pairs trading library: cointegration screening, signal generation,
backtesting, and performance metrics."""

from .cointegration import (
    load_prices,
    fit_ols,
    calc_spread,
    adf_test,
    screen_pair,
    scan_pairs,
)
from .signals import calc_zscore, generate_signals
from .backtest import run_backtest, walk_forward_backtest
from .metrics import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    payoff_ratio,
    avg_trade_duration,
    perf_metrics,
    print_metrics,
)
from .plotting import plot_adf_spread, plot_spread_zscore, plot_backtest

__all__ = [
    "load_prices",
    "fit_ols",
    "calc_spread",
    "adf_test",
    "screen_pair",
    "scan_pairs",
    "calc_zscore",
    "generate_signals",
    "run_backtest",
    "walk_forward_backtest",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "payoff_ratio",
    "avg_trade_duration",
    "perf_metrics",
    "print_metrics",
    "plot_adf_spread",
    "plot_spread_zscore",
    "plot_backtest",
]
