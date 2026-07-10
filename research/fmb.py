"""
Generic Fama-MacBeth engine. Every specification in the paper is a
configuration of this single function (the original script had five
near-identical ~100-line copies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HY_LABEL, IG_LABEL, MIN_OBS_CROSS_SECTION
from .signals import _clean_xy, month_design


def fama_macbeth(panel: pd.DataFrame, y_col: str = "RET_EOM_next",
                 key_cols: list[str] = ("spread_resid_w",),
                 extra_controls: list[str] = (),
                 rating_fe: bool = True, tmt_fe: bool = True,
                 min_obs: int = MIN_OBS_CROSS_SECTION) -> dict:
    """Monthly cross-sections of y_col on key_cols + DDCamp + log_size +
    duration (+ extra controls + FE); time-series averages and FMB
    t-stats. Returns dict(summary, betas)."""
    need = list(key_cols) + list(extra_controls) + [y_col]
    df = panel.dropna(subset=[c for c in need if c in panel.columns])
    betas, dates = [], []
    for date, df_m in df.groupby("DATE"):
        if len(df_m) < min_obs:
            continue
        X = month_design(df_m,
                         extra_cols=list(key_cols) + list(extra_controls),
                         rating_fe=rating_fe, tmt_fe=tmt_fe)
        Xv, yv = _clean_xy(X, df_m[y_col])
        if len(Xv) < Xv.shape[1] + 6:
            continue
        Xc = np.column_stack([np.ones(len(Xv)), Xv.to_numpy()])
        coef, *_ = np.linalg.lstsq(Xc, yv.to_numpy(), rcond=None)
        betas.append(pd.Series(coef, index=["const"] + list(Xv.columns)))
        dates.append(date)
    if not betas:
        return {"summary": pd.DataFrame(), "betas": pd.DataFrame()}
    B = pd.DataFrame(betas, index=pd.to_datetime(dates)).sort_index()
    n = B.notna().sum()
    summary = pd.DataFrame({
        "coef": B.mean(),
        "t_stat": B.mean() / (B.std(ddof=1) / np.sqrt(n)),
        "n_months": n})
    return {"summary": summary, "betas": B}


def fmb_interaction(panel: pd.DataFrame, y_col: str = "RET_EOM_next",
                    mis_col: str = "spread_resid_w",
                    extra_controls: list[str] = ()) -> dict:
    """Separate IG/HY mispricing slopes in one regression (paper eq. 4
    interaction form)."""
    df = panel[panel["RATING_CLASS"].isin([IG_LABEL, HY_LABEL])].copy()
    df["spread_resid_IG"] = df[mis_col] * (df["RATING_CLASS"] == IG_LABEL)
    df["spread_resid_HY"] = df[mis_col] * (df["RATING_CLASS"] == HY_LABEL)
    return fama_macbeth(df, y_col=y_col,
                        key_cols=["spread_resid_IG", "spread_resid_HY"],
                        extra_controls=extra_controls)


def fmb_by_segment(panel: pd.DataFrame, mis_col: str = "spread_resid_w",
                   extra_controls: list[str] = ()) -> pd.DataFrame:
    """The paper's headline table: full sample, IG-only, HY-only."""
    rows = {}
    samples = {"full": panel,
               "IG": panel[panel["RATING_CLASS"] == IG_LABEL],
               "HY": panel[panel["RATING_CLASS"] == HY_LABEL]}
    for name, s in samples.items():
        out = fama_macbeth(s, key_cols=[mis_col],
                           extra_controls=extra_controls)
        if mis_col in out["summary"].index:
            r = out["summary"].loc[mis_col]
            rows[name] = {"coef": r["coef"], "t_stat": r["t_stat"],
                          "n_months": int(r["n_months"])}
    return pd.DataFrame(rows).T


def fmb_dd_horizons(panel_dd: pd.DataFrame,
                    horizons: tuple[int, ...] = (1, 3, 6, 12),
                    mis_col: str = "spread_resid_w") -> pd.DataFrame:
    """Paper Table 6: mispricing + multi-horizon DD changes, IG and HY."""
    rows = []
    for seg, label in (("IG", IG_LABEL), ("HY", HY_LABEL)):
        seg_df = panel_dd[panel_dd["RATING_CLASS"] == label]
        for h in horizons:
            col = f"DDCamp_chg{h}"
            out = fama_macbeth(seg_df, key_cols=[mis_col],
                               extra_controls=[col])
            s = out["summary"]
            if mis_col in s.index and col in s.index:
                rows.append({
                    "segment": seg, "horizon_m": h,
                    "n_months": int(s.loc[mis_col, "n_months"]),
                    "beta_mis": s.loc[mis_col, "coef"],
                    "t_mis": s.loc[mis_col, "t_stat"],
                    "beta_dDD": s.loc[col, "coef"],
                    "t_dDD": s.loc[col, "t_stat"]})
    return pd.DataFrame(rows)
