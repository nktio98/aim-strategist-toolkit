"""Orchestrator: reproduce the paper, then re-run with the link fix.

Usage:
    python -m research.run_paper                 # both modes
    python -m research.run_paper --mode naive    # replication only
    python -m research.run_paper --mode window   # fixed merge only
"""
from __future__ import annotations

import argparse

from .artifacts import run_pipeline, save_artifacts


def _headline(res: dict) -> str:
    seg = res["fmb_segments"]
    ls = res["ls_stats"]
    lines = [
        f"mode={res['mode']}  winsorize={res['winsorize']}  "
        f"obs={res['n_obs']:,}  months={res['n_months']}  "
        f"({res['runtime_s']}s)",
        f"  avg first-stage R2: {res['r2'].mean():.4f}",
    ]
    for name in ("full", "IG", "HY"):
        if name in seg.index:
            r = seg.loc[name]
            lines.append(f"  FMB {name:4s}: coef={r['coef']:.4f} "
                         f"t={r['t_stat']:.2f} ({int(r['n_months'])} months)")
    it = res["fmb_interaction"]
    for c in ("spread_resid_IG", "spread_resid_HY"):
        if c in it.index:
            lines.append(f"  {c}: coef={it.loc[c, 'coef']:.4f} "
                         f"t={it.loc[c, 't_stat']:.2f}")
    lines.append(f"  IG LS: mean={ls['mean_monthly']:.4f}/m "
                 f"t={ls['t_stat']:.2f} Sharpe={ls['ann_sharpe']:.2f} "
                 f"({ls['n_months']} months)")
    tp = res["two_pass"]
    if "spread_resid_w" in tp.index:
        lines.append(f"  two-pass lambda_mis: "
                     f"{tp.loc['spread_resid_w', 'coef']:.4f} "
                     f"t={tp.loc['spread_resid_w', 't_stat']:.2f}")
    tf = res["three_factor"]
    lines.append(f"  3-factor alpha: {tf.loc['const', 'coef']:.5f} "
                 f"t={tf.loc['const', 't_stat']:.2f} R2={tf.attrs['r2']:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "window", "both"],
                    default="both")
    ap.add_argument("--winsorize", choices=["pooled", "monthly"],
                    default="pooled")
    args = ap.parse_args()
    modes = ["naive", "window"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n===== running pipeline: mode={mode} =====")
        res = run_pipeline(mode=mode, winsorize=args.winsorize)
        out = save_artifacts(res)
        print(_headline(res))
        print(f"  artifacts -> {out}")


if __name__ == "__main__":
    main()
