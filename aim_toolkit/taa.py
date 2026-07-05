"""
Tactical asset allocation: signals + honest backtesting.

The scientific core here is NOT the signals (those are simple, deliberately)
but the validation machinery that separates professional quant work from
curve-fitting:

  - PurgedKFold  : time-series cross-validation with purging (drop training
    samples whose label window overlaps the test set) and an embargo buffer
    after each test fold (Lopez de Prado, "Advances in Financial ML").
  - probabilistic_sharpe / deflated_sharpe : PSR corrects the Sharpe ratio
    for non-normality and sample length; DSR additionally corrects for the
    number of strategies tried (multiple testing) using the expected max
    Sharpe under the null (Bailey & Lopez de Prado 2014).

A strategy is only interesting if its DEFLATED Sharpe is significant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252


# ------------------------------------------------------------- signals
def momentum(prices: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Classic 12-1 momentum: return over lookback excluding last `skip` days."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def value_z(yield_series: pd.Series, window: int = 756) -> pd.Series:
    """Valuation z-score: current yield vs trailing distribution (high = cheap)."""
    mu = yield_series.rolling(window).mean()
    sd = yield_series.rolling(window).std()
    return (yield_series - mu) / sd


def carry(yield_series: pd.Series, funding: pd.Series) -> pd.Series:
    return yield_series - funding


def zscore_position(signal: pd.Series, window: int = 252,
                    cap: float = 2.0) -> pd.Series:
    """Convert signal to position in [-cap, cap] via rolling z-score."""
    z = (signal - signal.rolling(window).mean()) / signal.rolling(window).std()
    return z.clip(-cap, cap)


# --------------------------------------------------------- purged CV
class PurgedKFold:
    """K-fold CV for financial series: purge overlapping labels + embargo."""

    def __init__(self, n_splits: int = 5, label_horizon: int = 21,
                 embargo_pct: float = 0.02):
        self.k, self.h, self.embargo_pct = n_splits, label_horizon, embargo_pct

    def split(self, n: int):
        idx = np.arange(n)
        embargo = int(n * self.embargo_pct)
        folds = np.array_split(idx, self.k)
        for test in folds:
            t0, t1 = test[0], test[-1]
            train = idx[(idx < t0 - self.h) |            # purge pre-overlap
                        (idx > t1 + self.h + embargo)]   # purge + embargo post
            yield train, test


# ---------------------------------------------------------- backtest
def backtest(position: pd.Series, asset_ret: pd.Series,
             tcost_bp: float = 2.0) -> pd.DataFrame:
    """Daily P&L of position (entered at close, earns next-day return),
    net of proportional transaction costs on position changes."""
    pos = position.shift(1).fillna(0.0)
    gross = pos * asset_ret
    costs = pos.diff().abs().fillna(0.0) * tcost_bp / 1e4
    net = gross - costs
    return pd.DataFrame({"gross": gross, "costs": costs, "net": net})


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark), adjusting for skew/kurtosis/sample size."""
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std()                      # per-period SR
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3          # raw kurtosis
    sr_b = sr_benchmark / np.sqrt(ANN)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_b) * np.sqrt(T - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    trial_sr_var: float | None = None) -> float:
    """DSR: PSR against the expected max Sharpe from n_trials tries."""
    r = returns.dropna()
    sr_var = trial_sr_var if trial_sr_var is not None else (1.0 / len(r))
    em = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_max = np.sqrt(sr_var) * ((1 - em) * z1 + em * z2)   # per-period
    return probabilistic_sharpe(r, sr_benchmark=sr_max * np.sqrt(ANN))


def cv_sharpes(position_fn, params_grid: list[dict], signal_inputs,
               asset_ret: pd.Series, cv: PurgedKFold) -> pd.DataFrame:
    """Evaluate a param grid with purged CV. position_fn(inputs, **params)
    must return a position series aligned to asset_ret."""
    rows = []
    n = len(asset_ret)
    for params in params_grid:
        pos = position_fn(signal_inputs, **params)
        fold_srs = []
        for train, test in cv.split(n):
            net = backtest(pos.iloc[test], asset_ret.iloc[test])["net"]
            fold_srs.append(sharpe(net))
        rows.append({**params, "cv_sharpe_mean": np.mean(fold_srs),
                     "cv_sharpe_std": np.std(fold_srs)})
    return pd.DataFrame(rows).round(3)
