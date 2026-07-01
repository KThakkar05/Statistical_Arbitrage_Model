# Pairs Trading

A statistical arbitrage pipeline for equity pairs trading: cointegration
screening, a z-score mean-reversion signal, an event-driven backtest with
slippage, and a Streamlit dashboard for interactive exploration.

This repo consolidates two previous prototypes (`stats_arb` package +
exploratory scripts) into a single, tested package.

## Structure

```
pairs_trading/
├── data/                     # optional local price cache (empty by default; see data/README.md)
├── src/
│   ├── cointegration.py      # price loading, OLS hedge ratio, ADF test, pair screening/scanning
│   ├── signals.py            # rolling z-score, entry/exit signal generation
│   ├── backtest.py           # event-driven backtest + walk-forward re-fit loop
│   ├── metrics.py            # Sharpe, drawdown, win rate, payoff ratio
│   └── plotting.py           # matplotlib charts used by the notebook
├── notebooks/
│   └── exploration.ipynb     # end-to-end walkthrough on RIO.L / BHP.L
├── streamlit_app.py          # interactive dashboard (pair analysis + universe rankings)
├── tests/                    # unit tests on synthetic data (no network calls)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Notebook** — full pipeline walkthrough with plots:

```bash
jupyter notebook notebooks/exploration.ipynb
```

**Dashboard** — pick any two tickers, inspect cointegration/z-score, or scan
an entire industry for candidate pairs:

```bash
streamlit run streamlit_app.py
```

**Tests**:

```bash
pytest tests/
```

**Library** — import directly in your own scripts:

```python
from src.cointegration import load_prices, fit_ols, calc_spread, adf_test
from src.signals import calc_zscore, generate_signals
from src.backtest import run_backtest, walk_forward_backtest
from src.metrics import perf_metrics, print_metrics

s1, s2 = load_prices("RIO.L", "BHP.L", "2018-01-01", "2020-12-31")
alpha, beta, _ = fit_ols(s1, s2)
spread = calc_spread(s1, s2, alpha, beta)
print(adf_test(spread))

pnl_df, trade_log = run_backtest(s1, s2, alpha, beta, window=60, entry_z=2.0, exit_z=0.0)
print_metrics(perf_metrics(pnl_df, trade_log, cost=True))
```

## Methodology

1. **Cointegration screening** (`cointegration.py`) — regress one price series
   on the other via OLS to get a hedge ratio (β), form the residual spread,
   and run an Augmented Dickey-Fuller test. A low p-value is evidence the
   spread is stationary (mean-reverting), which is the premise the strategy
   trades on. `scan_pairs` batches this across a whole ticker universe and
   ranks candidates by ADF statistic.
2. **Signal generation** (`signals.py`) — a rolling z-score of the spread.
   Entering short when the z-score exceeds `+entry_z` (spread too wide, bet
   it narrows), long when it drops below `-entry_z`, and flattening once the
   z-score reverts past `exit_z`.
3. **Backtest** (`backtest.py`) — marks P&L to the daily change in spread
   while a position is held, charges an approximate two-sided slippage cost
   on entry/exit, and reconstructs a round-trip trade log.
   `walk_forward_backtest` avoids look-ahead bias by re-fitting the hedge
   ratio on a rolling training window and trading only the following
   out-of-sample block before re-fitting again.
4. **Metrics** (`metrics.py`) — annualized Sharpe ratio, max drawdown, win
   rate, payoff ratio, and average trade duration, computed pre- and
   post-slippage.

## Notes

- Price data is pulled live via `yfinance` — there's no bundled dataset, so
  an internet connection is needed to run the notebook, dashboard, or your
  own scripts against real tickers. The test suite uses synthetic data and
  needs no network access.
- The included fixed-hedge backtest fits α/β once on a training window and
  evaluates a separate test window (see the notebook). For a hedge ratio
  that adapts over time, use `walk_forward_backtest`.
