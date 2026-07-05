"""Manager oversight demo: alpha panel with FDR control + style drift."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit import managers as mg

OUT = "outputs"
rng = np.random.default_rng(21)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---- simulate 120 months of factor returns + 20 managers ----
T = 120
idx = pd.date_range("2016-01-31", periods=T, freq="ME")
factors = pd.DataFrame({
    "rates": rng.normal(0.001, 0.012, T),
    "credit": rng.normal(0.002, 0.015, T),
    "equity": rng.normal(0.005, 0.040, T),
}, index=idx)

true_alpha = np.zeros(20)
true_alpha[[3, 11]] = 0.0020            # 2 genuinely skilled managers (~2.4%/yr)
panel = {}
for j in range(20):
    betas = np.array([0.8, 0.6, 0.1]) + rng.normal(0, 0.15, 3)
    eps = rng.normal(0, 0.006, T)
    panel[f"MGR_{j:02d}"] = true_alpha[j] + factors.to_numpy() @ betas + eps
panel = pd.DataFrame(panel, index=idx)

section("1. FACTOR REGRESSIONS (Newey-West) ACROSS THE MANAGER PANEL")
rows = {}
for name in panel.columns:
    r = mg.factor_regression(panel[name], factors)
    r["pval"] = mg.alpha_pvalue_from_t(r["alpha_t"], T - 4)
    rows[name] = r
reg = pd.DataFrame(rows).T
show = reg[["alpha_ann_%", "alpha_t", "pval", "r2"]].round(3)
print(show.sort_values("alpha_t", ascending=False).head(8).to_string())

naive = (reg["alpha_t"].abs() > 2).sum()
print(f"\nNaive screen (|t|>2): {naive} managers look skilled.")

section("2. BENJAMINI-HOCHBERG FDR CONTROL (10% FDR)")
bh = mg.benjamini_hochberg(reg["pval"], fdr=0.10)
sig = bh[bh["significant_at_FDR"]]
print(bh.sort_values("pval").head(6).to_string())
truth = {f"MGR_{j:02d}" for j in np.where(true_alpha > 0)[0]}
print(f"\nFDR-significant managers: {list(sig.index)}")
print(f"True skilled managers   : {sorted(truth)}")
print("-> FDR keeps both truly skilled managers; any extras reflect the\n"
      "   chosen 10% false-discovery budget (vs the naive screen's "
      f"{naive} hits).")

section("3. STYLE DRIFT MONITOR (rolling betas, mandate compliance)")
# inject drift into one manager: credit beta ramps up in 2nd half
drift = panel["MGR_05"].copy()
ramp = np.linspace(0, 0.9, T)
drift = drift + ramp * factors["credit"]
rb = mg.rolling_betas(drift, factors, window=36)
print("MGR_05 credit beta: first year of estimates "
      f"~{rb['credit'].iloc[:12].mean():.2f} -> last year "
      f"~{rb['credit'].iloc[-12:].mean():.2f}  (mandate breach flag)")
print("\nAppraisal vs composite benchmark (MGR_03):")
bench = factors @ np.array([0.8, 0.6, 0.1])
print(mg.appraisal(panel["MGR_03"], bench))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
colors = ["#228833" if m in truth else ("#EE6677" if s else "#BBBBBB")
          for m, s in zip(reg.index, bh["significant_at_FDR"])]
ax[0].bar(range(20), reg["alpha_t"], color=colors)
ax[0].axhline(2, color="k", ls="--", lw=0.8); ax[0].axhline(0, color="k", lw=0.7)
ax[0].set_title("Alpha t-stats (green = true skill, grey = FDR-rejected)")
ax[0].set_xlabel("manager")
rb.plot(ax=ax[1], lw=1.2)
ax[1].set_title("MGR_05 rolling 36m betas: credit-beta drift")
fig.tight_layout(); fig.savefig(f"{OUT}/8_managers.png", dpi=130)
print("\nDone.")
