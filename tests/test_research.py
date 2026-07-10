"""Tests for the research pipeline — all on SYNTHETIC data (the licensed
WRDS panel never enters the test suite)."""
import numpy as np
import pandas as pd
import pytest

from research.fmb import fama_macbeth, fmb_by_segment, fmb_interaction
from research.panel import link_bonds
from research.portfolios import build_factors, long_short_stats, \
    nw_regression, quintile_returns, two_pass_fmb
from research.signals import add_dd_changes, spread_residuals

rng = np.random.default_rng(9)


# ------------------------------------------------------- synthetic panel
def synth_panel(T=90, n_bonds=300, mis_premium=0.08, seed=9):
    """Bond-month panel with a PLANTED mispricing premium: spreads are a
    known function of fundamentals plus a persistent bond-level pricing
    error; next-month returns load on that error with slope
    `mis_premium` (IG only)."""
    r = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=T, freq="ME")
    ids = np.arange(n_bonds)
    dd = r.normal(7, 2, n_bonds)
    size = r.uniform(4, 9, n_bonds)
    dur = r.uniform(1, 12, n_bonds)
    tmt = np.clip(dur * r.uniform(1.0, 1.6, n_bonds), 0.5, 40)
    rating = r.choice(["AAA", "AA", "A", "BBB", "BB", "B"], n_bonds,
                      p=[0.05, 0.15, 0.3, 0.3, 0.12, 0.08])
    is_hy = np.isin(rating, ["BB", "B"])
    mis = r.normal(0, 0.005, (T, n_bonds))          # pricing error (pp)
    rows = []
    for t in range(T):
        spread = (0.02 - 0.001 * dd - 0.002 * size + 0.0005 * dur
                  + 0.004 * is_hy + mis[t])
        ret_next = (0.003
                    + mis_premium * mis[t] * (~is_hy)
                    + r.normal(0, 0.006, n_bonds))
        for i in ids:
            rows.append({
                "ISSUE_ID": i, "DATE": dates[t], "PERMNO": 1000 + i,
                "T_Spread": spread[i], "RET_EOM": 0.0,
                "RET_EOM_next": ret_next[i],
                "DDCamp": dd[i] + r.normal(0, 0.05),
                "log_size": size[i], "DURATION": dur[i],
                "AMOUNT_OUTSTANDING": np.exp(size[i]),
                "TMT": tmt[i],
                "TMT_bucket": pd.cut([tmt[i]], bins=[0, 3, 7, 15, 100],
                                     labels=["0-3y", "3-7y", "7-15y",
                                             ">15y"])[0],
                "RATING_CAT": rating[i],
                "RATING_CLASS": "1.HY" if is_hy[i] else "0.IG"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def planted():
    panel = synth_panel()
    sig = spread_residuals(panel)
    return panel, sig


# ------------------------------------------------------------- link fix
def _link_fixture():
    bond = pd.DataFrame({
        "ISSUE_ID": [1, 1, 2],
        "CUSIP": ["00000000A"] * 2 + ["00000000B"],
        "DATE": pd.to_datetime(["2005-06-30", "2015-06-30", "2010-01-31"]),
    })
    link = pd.DataFrame({
        "CUSIP": ["00000000A", "00000000A", "00000000B"],
        "PERMNO": [111, 222, 333],
        "link_startdt": pd.to_datetime(["2000-01-01", "2011-01-01",
                                        "2000-01-01"]),
        "link_enddt": pd.to_datetime(["2010-12-31", "2099-12-31",
                                      "2099-12-31"]),
    })
    return bond, link


def test_window_merge_assigns_correct_issuer_over_time():
    bond, link = _link_fixture()
    out = link_bonds(bond, link, mode="window")
    assert len(out) == 3                              # no duplication
    a = out[out["CUSIP"] == "00000000A"].sort_values("DATE")
    assert list(a["PERMNO"]) == [111, 222]            # ownership change
    assert out[out["CUSIP"] == "00000000B"]["PERMNO"].iloc[0] == 333


def test_naive_merge_duplicates_rows_documenting_the_bug():
    bond, link = _link_fixture()
    out = link_bonds(bond, link, mode="naive")
    # CUSIP A maps to two PERMNOs -> each of its bond-months DUPLICATES
    assert len(out) == 5
    assert (out.groupby(["CUSIP", "DATE"]).size().max()) == 2


# ---------------------------------------------------------- first stage
def test_spread_residuals_recover_planted_mispricing(planted):
    panel, sig = planted
    p = sig["panel"]
    # residual should be centered and correlate with the DGP error via
    # the return equation: FMB below is the real check; here sanity only
    assert abs(p["spread_resid"].mean()) < 5e-4
    assert sig["r2"].mean() > 0.3                     # fundamentals load
    fs = sig["first_stage"]
    assert fs.loc["DDCamp", "coef"] == pytest.approx(-0.001, abs=3e-4)
    assert fs.loc["log_size", "coef"] == pytest.approx(-0.002, abs=5e-4)


def test_monthly_winsorize_mode_differs_from_pooled():
    panel = synth_panel(T=24, n_bonds=150, seed=3)
    pooled = spread_residuals(panel, winsorize="pooled")["panel"]
    monthly = spread_residuals(panel, winsorize="monthly")["panel"]
    assert not np.allclose(pooled["spread_resid_w"].to_numpy(),
                           monthly["spread_resid_w"].to_numpy())


# ------------------------------------------------------------------ FMB
def test_fmb_recovers_planted_premium_in_ig_not_hy(planted):
    _, sig = planted
    seg = fmb_by_segment(sig["panel"])
    assert seg.loc["IG", "coef"] == pytest.approx(0.08, abs=0.03)
    assert seg.loc["IG", "t_stat"] > 3
    assert abs(seg.loc["HY", "t_stat"]) < 2           # nothing planted
    inter = fmb_interaction(sig["panel"])["summary"]
    assert inter.loc["spread_resid_IG", "coef"] == pytest.approx(0.08,
                                                                 abs=0.03)


def test_fmb_null_panel_finds_nothing():
    panel = synth_panel(T=40, n_bonds=200, mis_premium=0.0, seed=11)
    sig = spread_residuals(panel)
    seg = fmb_by_segment(sig["panel"])
    assert abs(seg.loc["IG", "t_stat"]) < 2.2


# ------------------------------------------------------------ portfolios
def test_quintile_sort_monotone_and_ls_positive(planted):
    _, sig = planted
    q = quintile_returns(sig["panel"], rating_class="0.IG", min_bonds=30)
    means = q.mean()
    assert means["Q4"] > means["Q0"]                  # cheap beats rich
    ls = long_short_stats(q)
    assert ls["t_stat"] > 2 and ls["ann_sharpe"] > 0.5


def test_factors_and_nw_regression(planted):
    _, sig = planted
    p = sig["panel"]
    fac = build_factors(p)
    assert list(fac.columns) == ["MKT", "CRD", "TERM"]
    assert len(fac) > 40
    q = quintile_returns(p, rating_class="0.IG", min_bonds=30)
    ls_ret = (q["Q4"] - q["Q0"]).dropna()
    reg = nw_regression(ls_ret, fac, lags=6)
    assert set(reg.index) == {"const", "MKT", "CRD", "TERM"}
    assert np.isfinite(reg["t_stat"]).all()


def test_two_pass_prices_mispricing(planted):
    _, sig = planted
    p = sig["panel"]
    fac = build_factors(p)
    out = two_pass_fmb(p, fac, min_months=24)
    s = out["summary"]
    assert s.loc["spread_resid_w", "t_stat"] > 2     # planted premium

# ------------------------------------------------------------- DD changes
def test_add_dd_changes_horizons():
    panel = synth_panel(T=30, n_bonds=50, seed=5)
    out = add_dd_changes(panel, horizons=(1, 3))
    assert {"DDCamp_chg1", "DDCamp_chg3"} <= set(out.columns)
    one = out[out["ISSUE_ID"] == 0].sort_values("DATE")
    manual = one["DDCamp"].diff(3)
    assert np.allclose(one["DDCamp_chg3"].iloc[3:], manual.iloc[3:],
                       atol=1e-12)
