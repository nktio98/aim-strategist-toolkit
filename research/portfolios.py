"""
Portfolio machinery: mispricing quintile sorts, long-short strategies,
the corporate three-factor model (MKT / CRD / TERM), Newey-West
time-series regressions, and the two-pass Fama-MacBeth.

Reuses the toolkit's Newey-West implementation (validated against
statsmodels to 1e-10 in tests/test_validation.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from toolkit.managers import _newey_west_se

from .config import HY_LABEL, IG_LABEL, MIN_OBS_QUINTILES


def quintile_returns(panel: pd.DataFrame, sort_col: str = "spread_resid_w",
                     rating_class: str = IG_LABEL,
                     min_bonds: int = MIN_OBS_QUINTILES,
                     require_all: bool = True) -> pd.DataFrame:
    """Monthly EW next-month returns by sort-quintile within a rating
    class. Columns Q0..Q4 (Q0 = lowest sort value = most rich)."""
    seg = panel[(panel["RATING_CLASS"] == rating_class)
                & panel[sort_col].notna()
                & panel["RET_EOM_next"].notna()]
    rows = []
    for date, df_m in seg.groupby("DATE"):
        if len(df_m) < min_bonds:
            continue
        try:
            q = pd.qcut(df_m[sort_col], q=5, labels=False,
                        duplicates="drop")
        except ValueError:
            continue
        mean_ret = df_m.groupby(q)["RET_EOM_next"].mean()
        have = set(mean_ret.index)
        if require_all and not {0, 1, 2, 3, 4} <= have:
            continue
        if not {0, 4} <= have:
            continue
        rows.append({"DATE": date,
                     **{f"Q{k}": mean_ret.get(k, np.nan) for k in range(5)}})
    return pd.DataFrame(rows).set_index("DATE").sort_index()


def long_short_stats(q_df: pd.DataFrame) -> dict:
    """Q4-Q0 strategy summary (paper Section 5.5 metrics)."""
    ls = (q_df["Q4"] - q_df["Q0"]).dropna()
    t = len(ls)
    mean, std = ls.mean(), ls.std(ddof=1)
    return {"n_months": t, "mean_monthly": float(mean),
            "std_monthly": float(std),
            "t_stat": float(mean / (std / np.sqrt(t))),
            "ann_mean": float(12 * mean),
            "ann_sharpe": float(np.sqrt(12) * mean / std),
            "returns": ls.rename("LS_ret")}


def build_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """IG corporate three factors (paper Section 6.2):
    MKT  = EW next-month return of all IG bonds,
    CRD  = Q4-Q0 on T_Spread within IG (high minus low spread),
    TERM = EW long-maturity (7-15y, >15y) minus short (0-3y) IG bonds."""
    ig = panel[(panel["RATING_CLASS"] == IG_LABEL)
               & panel["RET_EOM_next"].notna()]
    mkt = ig.groupby("DATE")["RET_EOM_next"].mean().rename("MKT")
    crd_q = quintile_returns(panel, sort_col="T_Spread",
                             rating_class=IG_LABEL, require_all=False)
    crd = (crd_q["Q4"] - crd_q["Q0"]).rename("CRD")
    long_m = ig[ig["TMT_bucket"].isin(["7-15y", ">15y"])] \
        .groupby("DATE")["RET_EOM_next"].mean()
    short_m = ig[ig["TMT_bucket"] == "0-3y"] \
        .groupby("DATE")["RET_EOM_next"].mean()
    term = (long_m - short_m).rename("TERM")
    return pd.concat([mkt, crd, term], axis=1).dropna()


def nw_regression(y: pd.Series, X: pd.DataFrame, lags: int = 12) -> pd.DataFrame:
    """OLS with Newey-West t-stats; returns coef/t per regressor + R2."""
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0].to_numpy()
    Xc = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].to_numpy()])
    coef, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
    resid = yv - Xc @ coef
    se = _newey_west_se(Xc, resid, lags=lags)
    out = pd.DataFrame({"coef": coef, "t_stat": coef / se},
                       index=["const"] + list(df.columns[1:]))
    out.attrs["r2"] = float(1 - resid.var() / yv.var())
    out.attrs["n"] = len(df)
    return out


def two_pass_fmb(panel: pd.DataFrame, factors: pd.DataFrame,
                 mis_col: str = "spread_resid_w",
                 rating_class: str = IG_LABEL,
                 min_months: int = 24) -> dict:
    """Paper Section 6.3: first-pass per-bond factor betas (>=24 obs),
    second-pass monthly cross-sections of returns on mispricing + betas."""
    seg = panel[(panel["RATING_CLASS"] == rating_class)
                & panel["RET_EOM_next"].notna()
                & panel[mis_col].notna()].copy()
    fac = factors.copy()
    fcols = list(fac.columns)
    merged = seg.merge(fac, left_on="DATE", right_index=True, how="inner")

    # -- first pass: per-bond time-series betas
    betas = {}
    for issue, g in merged.groupby("ISSUE_ID"):
        if len(g) < min_months:
            continue
        Xc = np.column_stack([np.ones(len(g)), g[fcols].to_numpy()])
        coef, *_ = np.linalg.lstsq(Xc, g["RET_EOM_next"].to_numpy(),
                                   rcond=None)
        betas[issue] = coef[1:]
    if not betas:
        return {"summary": pd.DataFrame()}
    B = pd.DataFrame(betas, index=[f"beta_{c}" for c in fcols]).T
    B.index.name = "ISSUE_ID"

    # -- second pass: monthly cross-sections
    second = merged.merge(B, on="ISSUE_ID", how="inner")
    lambdas, dates = [], []
    cols = [mis_col] + [f"beta_{c}" for c in fcols]
    for date, df_m in second.groupby("DATE"):
        df_m = df_m.dropna(subset=cols + ["RET_EOM_next"])
        if len(df_m) < 40:
            continue
        Xc = np.column_stack([np.ones(len(df_m)), df_m[cols].to_numpy()])
        coef, *_ = np.linalg.lstsq(Xc, df_m["RET_EOM_next"].to_numpy(),
                                   rcond=None)
        lambdas.append(pd.Series(coef, index=["const"] + cols))
        dates.append(date)
    L = pd.DataFrame(lambdas, index=pd.to_datetime(dates)).sort_index()
    n = L.notna().sum()
    summary = pd.DataFrame({
        "coef": L.mean(),
        "t_stat": L.mean() / (L.std(ddof=1) / np.sqrt(n)),
        "n_months": n})
    return {"summary": summary, "lambdas": L, "n_bonds": len(B)}
