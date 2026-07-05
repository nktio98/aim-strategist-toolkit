"""FX module demo: hedged yield pickup, ECM fair value, min-var hedge ratio."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit import fx

OUT = "outputs"
rng = np.random.default_rng(11)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# --------------------------------------------- 1. hedged yield pickup
section("1. HEDGED YIELD PICKUP - buy hedged USD credit or local bonds?")
# Snapshot: USD IG credit (10y, 5.4%) hedged back to each local ccy vs
# the local 10y govt/credit alternative. Basis on non-USD leg (bp, ann.).
markets = {
    "SGD": dict(usd_asset_yield=5.4, r_usd=4.3, r_local=3.1,
                local_asset_yield=3.3, basis_bp=-35),
    "KRW": dict(usd_asset_yield=5.4, r_usd=4.3, r_local=3.0,
                local_asset_yield=3.6, basis_bp=-55),
    "TWD": dict(usd_asset_yield=5.4, r_usd=4.3, r_local=1.7,
                local_asset_yield=1.9, basis_bp=-80),
    "THB": dict(usd_asset_yield=5.4, r_usd=4.3, r_local=2.2,
                local_asset_yield=2.9, basis_bp=-25),
    "JPY": dict(usd_asset_yield=5.4, r_usd=4.3, r_local=0.6,
                local_asset_yield=1.1, basis_bp=-60),
}
tbl = fx.pickup_table(markets)
print(tbl.to_string())
print("\nReading: negative pickup means hedging cost eats the entire USD "
      "yield\nadvantage -> local bonds win. This is the daily reality for "
      "TWD/JPY books\nwhen US rates are high and the xccy basis is negative.")

fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#228833" if v > 0 else "#EE6677" for v in tbl["pickup (bp)"]]
ax.bar(tbl.index, tbl["pickup (bp)"], color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("bp"); ax.set_title(
    "Hedged USD IG yield pickup vs local asset, by investor currency")
fig.tight_layout(); fig.savefig(f"{OUT}/4_fx_pickup.png", dpi=130)

# ------------------------------------------------ 2. ECM fair value
section("2. FX FAIR VALUE - Engle-Granger cointegration + ECM")
# Simulate: fundamentals drive long-run log spot; spot mean-reverts to it.
n = 520  # weekly, ~10y
rate_diff = np.cumsum(rng.normal(0, 0.06, n)) + 1.5        # r_usd - r_local
tot = np.cumsum(rng.normal(0, 0.15, n))                    # terms-of-trade proxy
fair = 0.30 + 0.045 * rate_diff - 0.012 * tot              # true long-run relation
mis = np.zeros(n)
for t in range(1, n):
    mis[t] = 0.965 * mis[t - 1] + rng.normal(0, 0.006)     # AR(1) misvaluation
log_spot = pd.Series(fair + mis,
                     index=pd.date_range("2016-01-01", periods=n, freq="W"),
                     name="log_usd_asia")
fund = pd.DataFrame({"rate_diff": rate_diff, "tot": tot}, index=log_spot.index)

m = fx.ECMFairValue().fit(log_spot, fund)
print(m.summary())
print(f"Long-run betas: const={m.beta[0]:.3f}, "
      + ", ".join(f"{c}={b:.4f}" for c, b in zip(m.cols, m.beta[1:])))
cur = m.resid.iloc[-1] * 100
print(f"\nCurrent misvaluation: {cur:+.1f}% "
      f"({'ccy pair rich vs fundamentals' if cur > 0 else 'cheap vs fundamentals'})")

fv = m.fair_value(fund)
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(log_spot.index, log_spot, label="log spot", color="k", lw=1)
ax[0].plot(fv.index, fv, label="ECM fair value", color="#4477AA", lw=1.4)
ax[0].legend(); ax[0].set_title("Spot vs cointegration-based fair value")
ax[1].plot(m.resid.index, m.resid * 100, color="#EE6677", lw=1)
ax[1].axhline(0, color="k", lw=0.7)
sd = m.resid.std() * 100
for k in (2, -2):
    ax[1].axhline(k * sd, color="gray", ls="--", lw=0.8)
ax[1].set_title(f"Misvaluation (%), half-life {m.half_life:.0f} weeks; "
                "dashed = +/-2 sd (signal bands)")
fig.tight_layout(); fig.savefig(f"{OUT}/5_fx_fairvalue.png", dpi=130)

# ------------------------------------- 3. min-variance hedge ratio
section("3. TIME-VARYING MINIMUM-VARIANCE HEDGE RATIO")
# USD asset held by SGD investor: unhedged local-ccy return = usd asset ret + fx ret
nd = 1500
fx_ret = pd.Series(rng.normal(0, 0.004, nd),
                   index=pd.bdate_range("2020-01-01", periods=nd))
# correlation regime flips sign halfway (risk-on/risk-off shift)
corr_load = np.where(np.arange(nd) < nd // 2, 0.5, -0.4)
asset_usd = 0.0002 + corr_load * fx_ret + rng.normal(0, 0.005, nd)
unhedged = pd.Series(asset_usd, index=fx_ret.index) + fx_ret

h = fx.min_var_hedge_ratio(unhedged, fx_ret, window=126).dropna()
print(f"Hedge ratio: start-of-sample ~{h.iloc[:250].mean():.2f}, "
      f"end-of-sample ~{h.iloc[-250:].mean():.2f}")
print("A fixed 100% hedge policy is optimal only if asset/FX correlation "
      "is zero;\nwhen correlation flips (risk-off USD strength), the "
      "optimal ratio moves\nmaterially - the case for monitoring it, "
      "not setting-and-forgetting.")

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(h.index, h, color="#4477AA")
ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="full hedge")
ax.set_title("Rolling 6m minimum-variance hedge ratio (USD asset, SGD investor)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/6_fx_hedge_ratio.png", dpi=130)

print("\nCharts saved to outputs/. Done.")
