"""Allocation demo: Black-Litterman and entropy pooling -> TAA weights."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit import allocation as al

OUT = "outputs"
rng = np.random.default_rng(5)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


assets = ["Govt bonds", "IG credit", "EM debt", "Equity", "Real assets"]
vols = np.array([0.05, 0.07, 0.10, 0.16, 0.11])
corr = np.array([
    [1.00, 0.60, 0.30, -0.10, 0.10],
    [0.60, 1.00, 0.55, 0.25, 0.25],
    [0.30, 0.55, 1.00, 0.45, 0.30],
    [-0.10, 0.25, 0.45, 1.00, 0.45],
    [0.10, 0.25, 0.30, 0.45, 1.00]])
Sigma = np.outer(vols, vols) * corr
w_saa = np.array([0.45, 0.25, 0.08, 0.10, 0.12])       # insurance-like SAA

section("1. BLACK-LITTERMAN: equilibrium + strategist views")
Pi = al.implied_returns(w_saa, Sigma)
print("Equilibrium (reverse-optimized) excess returns (%):",
      np.round(Pi * 100, 2))

# Views: (a) IG credit outperforms govvies by 1.5%; (b) equity returns 5%
P = np.array([[-1, 1, 0, 0, 0],
              [0, 0, 0, 1, 0]])
Q = np.array([0.015, 0.05])
mu_bl, _ = al.black_litterman(Sigma, w_saa, P, Q)
print("BL posterior returns (%):          ", np.round(mu_bl * 100, 2))

w_bl = al.mv_optimize(mu_bl, Sigma, w_max=0.55)
tilt = pd.DataFrame({"SAA": w_saa, "BL-optimal": np.round(w_bl, 3),
                     "tilt": np.round(w_bl - w_saa, 3)}, index=assets)
print("\n" + tilt.to_string())

section("2. ENTROPY POOLING: same views, full-distribution version")
# scenario set: 10k joint draws (in practice: stress-engine simulations)
L = np.linalg.cholesky(Sigma)
scen = (Pi[None, :] + (rng.standard_normal((10_000, 5)) @ L.T))
A = np.vstack([al.view_on_mean(scen, 1) - al.view_on_mean(scen, 0),  # credit-govt
               al.view_on_mean(scen, 3)])                            # equity
ep = al.EntropyPooling().fit(scen, A, np.array([0.015, 0.05]))
mu_ep, Sigma_ep = ep.posterior_moments()
print("EP posterior returns (%):", np.round(mu_ep * 100, 2))
print(f"Effective scenario count: {ep.effective_n:,.0f} / 10,000  "
      f"(KL divergence {ep.kl:.4f}) - views are mild, distribution intact")
w_ep = al.mv_optimize(mu_ep, Sigma_ep, w_max=0.55)
print("EP-optimal weights:", dict(zip(assets, np.round(w_ep, 3))))
print("\nBL and EP agree here because views are on means of a Gaussian set;\n"
      "EP's advantage appears with non-normal scenarios (stress-engine\n"
      "output) or views on vols/tails/probabilities, which BL cannot express.")

fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = np.arange(len(assets)); wdt = 0.27
ax.bar(x - wdt, w_saa, wdt, label="SAA", color="#BBBBBB")
ax.bar(x, w_bl, wdt, label="Black-Litterman", color="#4477AA")
ax.bar(x + wdt, w_ep, wdt, label="Entropy pooling", color="#228833")
ax.set_xticks(x, assets, rotation=12); ax.set_ylabel("weight")
ax.set_title("SAA vs view-conditioned TAA allocations"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/9_allocation.png", dpi=130)
print("\nDone.")
