"""
Signal construction: monthly cross-sectional spread-pricing model and
the mispricing residual (paper Section 3.1, eq. 2-3).

For each month t:  T_Spread ~ DDCamp + log_size + duration
                              + rating FE + TMT-bucket FE
Residual = observed - fitted ("cheap" if positive, "rich" if negative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_OBS_CROSS_SECTION, WINSOR_Q


def month_design(df_m: pd.DataFrame, extra_cols: list[str] | None = None,
                 rating_fe: bool = True, tmt_fe: bool = True
                 ) -> pd.DataFrame:
    """Design matrix for one monthly cross-section, mirroring the
    original script: per-month dummies with drop_first (the omitted
    category is the alphabetically-first PRESENT that month)."""
    X = pd.DataFrame(index=df_m.index)
    for c in (extra_cols or []):
        X[c] = df_m[c]
    if "DDCamp" in df_m.columns:
        X["DDCamp"] = df_m["DDCamp"]
    X["log_size"] = df_m["log_size"]
    X["duration"] = df_m["DURATION"]
    if rating_fe and "RATING_CAT" in df_m.columns:
        X = pd.concat([X, pd.get_dummies(df_m["RATING_CAT"], prefix="rat",
                                         drop_first=True)], axis=1)
    if tmt_fe and "TMT_bucket" in df_m.columns:
        X = pd.concat([X, pd.get_dummies(df_m["TMT_bucket"], prefix="tmt",
                                         drop_first=True)], axis=1)
    return X


def _clean_xy(X: pd.DataFrame, y: pd.Series):
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & y.notna()
    Xv = X[valid].astype(float)
    yv = y[valid].astype(float)
    return Xv, yv


def spread_residuals(panel: pd.DataFrame,
                     winsorize: str = "pooled",
                     min_obs: int = MIN_OBS_CROSS_SECTION) -> dict:
    """Estimate the monthly spread model; attach spread_resid and
    spread_resid_w to the panel.

    winsorize="pooled" reproduces the original code (1%/99% over the
    FULL panel); "monthly" winsorizes within each month (what the paper
    text describes). The difference is a documented code/paper
    discrepancy -- run both.

    Returns dict(panel, first_stage(coef/t per regressor), r2(Series)).
    """
    p = panel.copy()
    p["spread_resid"] = np.nan
    betas, dates, r2s = [], [], []
    for date, df_m in p.groupby("DATE"):
        if len(df_m) < min_obs:
            continue
        X = month_design(df_m)
        Xv, yv = _clean_xy(X, df_m["T_Spread"])
        if len(Xv) < Xv.shape[1] + 6:
            continue
        Xc = np.column_stack([np.ones(len(Xv)), Xv.to_numpy()])
        coef, *_ = np.linalg.lstsq(Xc, yv.to_numpy(), rcond=None)
        fitted = Xc @ coef
        resid = yv.to_numpy() - fitted
        p.loc[Xv.index, "spread_resid"] = resid
        betas.append(pd.Series(coef, index=["const"] + list(Xv.columns)))
        dates.append(date)
        r2s.append(1 - resid.var() / yv.to_numpy().var())
    p = p.dropna(subset=["spread_resid"])

    sr = p["spread_resid"]
    if winsorize == "pooled":
        lo, hi = sr.quantile(WINSOR_Q)
        p["spread_resid_w"] = sr.clip(lo, hi)
    elif winsorize == "monthly":
        p["spread_resid_w"] = sr.groupby(p["DATE"]).transform(
            lambda s: s.clip(*s.quantile(WINSOR_Q)))
    else:
        raise ValueError(f"unknown winsorize mode {winsorize!r}")

    B = pd.DataFrame(betas, index=pd.to_datetime(dates)).sort_index()
    first_stage = pd.DataFrame({
        "coef": B.mean(),
        "t_stat": B.mean() / (B.std(ddof=1) / np.sqrt(B.notna().sum())),
        "n_months": B.notna().sum()})
    return {"panel": p.reset_index(drop=True), "first_stage": first_stage,
            "r2": pd.Series(r2s, index=pd.to_datetime(dates)).sort_index()}


def add_dd_changes(panel: pd.DataFrame,
                   horizons: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    """Issuer-level cumulative DD changes over h months (paper eq. 5):
    DDCamp_chg{h} = DDCamp_t - DDCamp_{t-h}, aligned back to bonds."""
    p = panel.copy()
    iss = (p[["PERMNO", "DATE", "DDCamp"]].dropna()
           .drop_duplicates(subset=["PERMNO", "DATE"])
           .sort_values(["PERMNO", "DATE"]))
    for h in horizons:
        iss[f"DDCamp_chg{h}"] = iss.groupby("PERMNO")["DDCamp"].diff(h)
    cols = ["PERMNO", "DATE"] + [f"DDCamp_chg{h}" for h in horizons]
    return p.merge(iss[cols], on=["PERMNO", "DATE"], how="left")
