"""Build the unified strategist dashboard from all module outputs."""
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

from aim_toolkit.dashboard import Dashboard
from aim_toolkit import fx

# 1) refresh all module outputs
for script in ("run_demo.py", "run_fx_demo.py", "run_taa_demo.py",
               "run_manager_demo.py", "run_allocation_demo.py"):
    print(f">>> {script}")
    subprocess.run([sys.executable, script], check=True,
                   capture_output=True)

# 2) recompute the headline pickup table for the dashboard front page
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

d = Dashboard(
    "Regional Investment Strategy - Analytics Dashboard",
    f"Asia portfolios | generated {date.today()} | demo data (simulated)")

d.add_section(
    "Rates: Dynamic Nelson-Siegel curve model & 12m forecast",
    "Level/slope/curvature factors extracted daily; VAR(1) dynamics drive "
    "the forward curve path used in income and reinvestment projections.",
    image_path="outputs/1_yield_curve.png")

d.add_section(
    "FX: hedged yield pickup by investor currency",
    "Hedged USD IG vs local asset. Negative pickup = local bonds win after "
    "hedge costs (US rate differential + xccy basis).",
    image_path="outputs/4_fx_pickup.png", table=tbl)

d.add_section(
    "FX fair value: cointegration / error-correction model",
    "Misvaluation vs macro fundamentals with +/-2sd tactical signal bands; "
    "half-life quantifies expected reversion speed.",
    image_path="outputs/5_fx_fairvalue.png")

d.add_section(
    "Market regimes: statistical jump model vs Markov-switching",
    "Persistent regime states feed hedge-ratio policy, TAA risk budgets "
    "and stress-scenario probabilities.",
    image_path="outputs/2_regimes.png")

d.add_section(
    "ALM stress testing: scenario P&L decomposition",
    "Instantaneous shocks through key-rate durations, spread duration, "
    "equity and FX exposures; duration gap gives the surplus lens.",
    image_path="outputs/3_stress.png")

d.add_section(
    "TAA research: purged cross-validation + deflated Sharpe",
    "Signals only deploy if the deflated Sharpe (corrected for number of "
    "trials) is significant - the anti-overfitting gate.",
    image_path="outputs/7_taa.png")

d.add_section(
    "Manager oversight: FDR-controlled alpha & style drift",
    "Newey-West alpha t-stats across the manager panel with "
    "Benjamini-Hochberg false-discovery control; rolling betas flag "
    "mandate drift.",
    image_path="outputs/8_managers.png")

d.add_section(
    "Allocation: Black-Litterman & entropy pooling",
    "House views blended with equilibrium (BL) or imposed on the full "
    "scenario distribution (entropy pooling), then optimized under "
    "constraints.",
    image_path="outputs/9_allocation.png")

out = d.save("outputs/strategist_dashboard.html",
             footer="AIM Strategist Toolkit demo. All figures based on "
                    "simulated data; illustrative only, not investment advice.")
print(f"\nDashboard written to {out}")
