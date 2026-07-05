"""
Dynamic Nelson-Siegel (Diebold-Li) yield curve engine.

Pipeline:
  1. Cross-sectional fit: for each date, OLS of yields on NS loadings
     (lambda chosen by grid search over the full panel).
  2. Time-series dynamics: VAR(1) on the level/slope/curvature factors.
  3. Forecast: iterate the VAR forward h steps, reconstruct the curve.

Upgrade paths (interfaces kept compatible):
  - AFNS: add the Christensen-Diebold-Rudebusch yield-adjustment term.
  - ACM term premium: bolt a linear-regression-based affine model on the
    same factor panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ns_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    """Nelson-Siegel factor loadings. maturities in years. Returns (n_mat, 3)."""
    tau = np.asarray(maturities, dtype=float)
    x = lam * tau
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curv])


def fit_cross_section(yields: pd.DataFrame, lam: float) -> pd.DataFrame:
    """OLS fit of NS factors for every date. yields: index=date, cols=maturities(yrs)."""
    mats = yields.columns.to_numpy(dtype=float)
    X = ns_loadings(mats, lam)
    # beta = (X'X)^-1 X' y  for all dates at once
    beta, *_ = np.linalg.lstsq(X, yields.to_numpy().T, rcond=None)
    return pd.DataFrame(beta.T, index=yields.index,
                        columns=["level", "slope", "curvature"])


def fit_lambda(yields: pd.DataFrame, grid=None) -> float:
    """Grid-search lambda minimizing total squared fitting error."""
    if grid is None:
        grid = np.linspace(0.2, 2.0, 37)
    mats = yields.columns.to_numpy(dtype=float)
    Y = yields.to_numpy().T  # (n_mat, n_dates)
    best_lam, best_sse = None, np.inf
    for lam in grid:
        X = ns_loadings(mats, lam)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        sse = np.sum((Y - X @ beta) ** 2)
        if sse < best_sse:
            best_lam, best_sse = float(lam), sse
    return best_lam


class VAR1:
    """Minimal VAR(1) with intercept, OLS-estimated equation by equation."""

    def fit(self, F: pd.DataFrame) -> "VAR1":
        Z = F.to_numpy()
        X = np.column_stack([np.ones(len(Z) - 1), Z[:-1]])
        Y = Z[1:]
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.c = B[0]
        self.A = B[1:].T                      # Y_t = c + A Y_{t-1} + e
        resid = Y - X @ B
        self.Sigma = np.cov(resid.T)
        self.cols = list(F.columns)
        self.last = Z[-1]
        return self

    def forecast(self, h: int) -> pd.DataFrame:
        out, z = [], self.last.copy()
        for _ in range(h):
            z = self.c + self.A @ z
            out.append(z.copy())
        return pd.DataFrame(out, columns=self.cols,
                            index=pd.RangeIndex(1, h + 1, name="h"))

    def long_run_mean(self) -> np.ndarray:
        return np.linalg.solve(np.eye(len(self.c)) - self.A, self.c)


class DNSModel:
    """End-to-end Dynamic Nelson-Siegel model."""

    def fit(self, yields: pd.DataFrame) -> "DNSModel":
        self.maturities = yields.columns.to_numpy(dtype=float)
        self.lam = fit_lambda(yields)
        self.factors = fit_cross_section(yields, self.lam)
        self.var = VAR1().fit(self.factors)
        fitted = self.reconstruct(self.factors)
        self.rmse_bp = float(np.sqrt(np.mean(
            (fitted.to_numpy() - yields.to_numpy()) ** 2)) * 100)
        return self

    def reconstruct(self, factors: pd.DataFrame) -> pd.DataFrame:
        X = ns_loadings(self.maturities, self.lam)
        return pd.DataFrame(factors.to_numpy() @ X.T,
                            index=factors.index, columns=self.maturities)

    def forecast_curve(self, h: int) -> pd.DataFrame:
        return self.reconstruct(self.var.forecast(h))
