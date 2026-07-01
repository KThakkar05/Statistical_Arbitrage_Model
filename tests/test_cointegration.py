"""Tests for src/cointegration.py using synthetic data (no network calls)."""
import numpy as np
import pandas as pd

from src.cointegration import adf_test, calc_spread, fit_ols


def _synthetic_pair(n=500, beta=1.5, alpha=2.0, noise_std=0.5, seed=0):
    """Build a synthetic cointegrated pair: y = alpha + beta * x + stationary noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=dates, name="X")
    noise = rng.normal(0, noise_std, n)
    y = pd.Series(alpha + beta * x.values + noise, index=dates, name="Y")
    return y, x


def test_fit_ols_recovers_known_hedge_ratio():
    y, x = _synthetic_pair(beta=1.5, alpha=2.0, noise_std=0.1)
    alpha, beta, model = fit_ols(y, x)

    assert np.isclose(beta, 1.5, atol=0.05)
    assert np.isclose(alpha, 2.0, atol=0.5)
    assert model is not None


def test_calc_spread_is_stationary_residual():
    y, x = _synthetic_pair(beta=1.5, alpha=2.0, noise_std=0.1)
    alpha, beta, _ = fit_ols(y, x)
    spread = calc_spread(y, x, alpha, beta)

    assert spread.name == "Spread"
    assert len(spread) == len(y)
    # Residual should be small and centered near zero given the low noise_std used above.
    assert abs(spread.mean()) < 1.0


def test_adf_test_rejects_nonstationary_spread():
    # A random walk spread should NOT look stationary (high p-value).
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    random_walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)), index=dates)

    result = adf_test(random_walk)
    assert result["p_value"] > 0.05


def test_adf_test_accepts_stationary_spread():
    # A mean-reverting (AR(1) with small coefficient) series should look stationary.
    rng = np.random.default_rng(2)
    n = 500
    stationary = np.zeros(n)
    for i in range(1, n):
        stationary[i] = 0.3 * stationary[i - 1] + rng.normal(0, 1)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    spread = pd.Series(stationary, index=dates)

    result = adf_test(spread)
    assert result["p_value"] < 0.05
