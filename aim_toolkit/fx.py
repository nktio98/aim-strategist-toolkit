"""
FX analytics for an insurance investor (Asian entity, USD assets).

Components:

1. Hedge-cost engine (covered interest parity + cross-currency basis).
   Annualized cost of hedging USD exposure back to local currency:
       hedge_cost ~= r_usd - r_local - xccy_basis
   (basis quoted on the non-USD leg; negative basis makes hedging USD
   assets MORE expensive for the Asian investor).
   Hedged yield pickup = usd_asset_yield - hedge_cost - local_asset_yield.
   This single number drives the "buy hedged USD credit vs local bonds"
   decision that dominates Asian insurance portfolio construction.

2. Fair-value engine: Engle-Granger cointegration + error-correction model.
   Long-run: log spot regressed on fundamentals (rate differential, terms
   of trade proxy, ...). ADF test on residuals (from scratch), ECM speed
   of adjustment -> half-life of misvaluation. This is the workhorse
   behind BEER-style FX valuation used on macro desks.

3. Minimum-variance hedge ratio: rolling OLS of unhedged asset returns
   (local ccy) on FX returns; h* = -beta. Modern practice treats the
   hedge ratio as a time-varying estimate, not a fixed policy number
   (upgrade path: DCC-GARCH conditional betas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------- 1. hedge costs
def hedge_cost(r_usd: pd.Series | float, r_local: pd.Series | float,
               basis_bp: pd.Series | float = 0.0):
    """Annualized cost (%) of hedging USD exposure to local currency."""
    return r_usd - r_local - basis_bp / 100.0


def hedged_pickup(usd_asset_yield, r_usd, r_local, local_asset_yield,
                  basis_bp=0.0):
    """Hedged yield pickup (%) of USD asset vs local asset."""
    return usd_asset_yield - hedge_cost(r_usd, r_local, basis_bp) \
        - local_asset_yield


def pickup_table(markets: dict) -> pd.DataFrame:
    """markets: name -> dict(usd_asset_yield, r_usd, r_local,
    local_asset_yield, basis_bp). Returns decision table sorted by pickup."""
    rows = {}
    for name, m in markets.items():
        hc = hedge_cost(m["r_usd"], m["r_local"], m["basis_bp"])
        pu = m["usd_asset_yield"] - hc - m["local_asset_yield"]
        rows[name] = {
            "USD asset yld": m["usd_asset_yield"],
            "hedge cost": round(hc, 2),
            "hedged USD yld": round(m["usd_asset_yield"] - hc, 2),
            "local asset yld": m["local_asset_yield"],
            "pickup (bp)": round(pu * 100, 0),
        }
    return pd.DataFrame(rows).T.sort_values("pickup (bp)", ascending=False)


# ----------------------------------------------- 2. cointegration / ECM
def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    T, k = X.shape
    sigma2 = resid @ resid / (T - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, se, resid


def adf_tstat(u: np.ndarray, lags: int = 1) -> float:
    """ADF t-stat (no constant; u is a residual, mean ~ 0) on du = rho*u(-1)+..."""
    du = np.diff(u)
    X = [u[lags:-1] if lags else u[:-1]]
    du_dep = du[lags:]
    for j in range(1, lags + 1):
        X.append(du[lags - j:-j])
    X = np.column_stack(X)
    beta, se, _ = _ols(X, du_dep)
    return float(beta[0] / se[0])


# Engle-Granger critical values (2 variables, constant in coint. regression)
EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


class ECMFairValue:
    """Engle-Granger two-step: long-run fair value + error-correction dynamics."""

    def fit(self, log_spot: pd.Series, fundamentals: pd.DataFrame) -> "ECMFairValue":
        y = log_spot.to_numpy()
        F = fundamentals.to_numpy()
        X = np.column_stack([np.ones(len(y)), F])
        self.beta, _, u = _ols(X, y)
        self.resid = pd.Series(u, index=log_spot.index, name="misvaluation")
        self.adf_t = adf_tstat(u, lags=1)
        self.cointegrated_5pct = self.adf_t < EG_CRIT[0.05]
        # ECM: d(log_spot) = a + gamma * u(-1) + phi * d(log_spot)(-1)
        dy = np.diff(y)
        Xe = np.column_stack([np.ones(len(dy) - 1), u[1:-1], dy[:-1]])
        be, se, _ = _ols(Xe, dy[1:])
        self.gamma, self.gamma_t = float(be[1]), float(be[1] / se[1])
        self.half_life = float(np.log(0.5) / np.log(1 + self.gamma)) \
            if -1 < self.gamma < 0 else np.inf
        self.cols = list(fundamentals.columns)
        return self

    def fair_value(self, fundamentals: pd.DataFrame) -> pd.Series:
        X = np.column_stack([np.ones(len(fundamentals)),
                             fundamentals.to_numpy()])
        return pd.Series(X @ self.beta, index=fundamentals.index)

    def summary(self) -> str:
        sig = "YES" if self.cointegrated_5pct else "NO"
        return (f"ADF t-stat on residual: {self.adf_t:.2f} "
                f"(5% crit {EG_CRIT[0.05]}) -> cointegrated: {sig}\n"
                f"ECM speed gamma: {self.gamma:.3f} (t={self.gamma_t:.1f}) "
                f"-> half-life of misvaluation: {self.half_life:.1f} periods")


# --------------------------------------- 3. minimum-variance hedge ratio
def min_var_hedge_ratio(asset_ret_local: pd.Series, fx_ret: pd.Series,
                        window: int = 126) -> pd.Series:
    """Rolling h* = Cov(unhedged_ret, fx_ret)/Var(fx_ret). h*=1 -> full hedge.
    asset_ret_local: UNHEDGED asset return measured in the investor's ccy."""
    cov = asset_ret_local.rolling(window).cov(fx_ret)
    var = fx_ret.rolling(window).var()
    return (cov / var).rename("min_var_hedge_ratio")
