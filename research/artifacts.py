"""
Run the full pipeline and save PUBLIC-SAFE aggregate artifacts.

Only aggregated statistics leave this module (regression tables, factor
and portfolio return series, strategy stats) -- never bond-level rows
of the licensed WRDS data. The artifact bundle is what the website's
Research tab renders.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_DIR
from .fmb import fama_macbeth, fmb_by_segment, fmb_dd_horizons, \
    fmb_interaction
from .panel import build_panel
from .portfolios import build_factors, long_short_stats, nw_regression, \
    quintile_returns, two_pass_fmb
from .signals import add_dd_changes, spread_residuals


def run_pipeline(mode: str = "window", winsorize: str = "pooled",
                 panel: pd.DataFrame | None = None) -> dict:
    """End-to-end run. Pass a prebuilt panel to skip the data build
    (used by tests with synthetic data)."""
    t0 = time.time()
    if panel is None:
        panel = build_panel(mode=mode)
    sig = spread_residuals(panel, winsorize=winsorize)
    p = sig["panel"]

    seg_table = fmb_by_segment(p)
    interact = fmb_interaction(p)["summary"]
    full_fmb = fama_macbeth(p)["summary"]

    ig_q = quintile_returns(p, rating_class="0.IG")
    hy_q = quintile_returns(p, rating_class="1.HY")
    ls = long_short_stats(ig_q)

    factors = build_factors(p)
    mis_ret = ls.pop("returns")
    three_factor = nw_regression(mis_ret, factors, lags=12)

    p_dd = add_dd_changes(p)
    dd_table = fmb_dd_horizons(p_dd)

    twopass = two_pass_fmb(p, factors)

    return {
        "mode": mode, "winsorize": winsorize,
        "n_obs": int(len(p)),
        "n_months": int(p["DATE"].nunique()),
        "first_stage": sig["first_stage"], "r2": sig["r2"],
        "fmb_full": full_fmb, "fmb_segments": seg_table,
        "fmb_interaction": interact,
        "ig_quintiles": ig_q, "hy_quintiles": hy_q,
        "ls_stats": ls, "ls_returns": mis_ret,
        "factors": factors, "three_factor": three_factor,
        "dd_horizons": dd_table, "two_pass": twopass["summary"],
        "runtime_s": round(time.time() - t0, 1),
    }


def save_artifacts(results: dict, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else ARTIFACT_DIR / results["mode"]
    out.mkdir(parents=True, exist_ok=True)
    for key in ("first_stage", "fmb_full", "fmb_segments",
                "fmb_interaction", "three_factor", "dd_horizons",
                "two_pass"):
        df = results[key]
        if isinstance(df, pd.DataFrame) and len(df):
            df.round(6).to_csv(out / f"{key}.csv")
    for key in ("ig_quintiles", "hy_quintiles", "factors"):
        results[key].round(6).to_csv(out / f"{key}.csv")
    results["ls_returns"].round(6).to_csv(out / "ls_returns.csv")
    meta = {k: results[k] for k in ("mode", "winsorize", "n_obs",
                                    "n_months", "runtime_s")}
    meta["ls_stats"] = {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in results["ls_stats"].items()}
    meta["avg_first_stage_r2"] = round(float(results["r2"].mean()), 4)
    (out / "meta.json").write_text(json.dumps(meta, indent=2),
                                   encoding="utf-8")
    return out
