"""
Panel construction: load WRDS parquet files, link bonds to issuers,
merge distance-to-default, apply the paper's sample filters.

The CUSIP->PERMNO link fix
--------------------------
The original script merged on CUSIP alone via
``link[["CUSIP","PERMNO"]].drop_duplicates()``. When one CUSIP maps to
different PERMNOs over time (M&A, spin-offs), that merge (a) DUPLICATES
every bond-month once per candidate PERMNO and (b) assigns issuers
outside their link validity window, so bonds can inherit the wrong
firm's distance-to-default. ``mode="window"`` enforces
link_startdt <= DATE <= link_enddt and dedupes to one issuer per
bond-month. ``mode="naive"`` reproduces the original behavior exactly
so the paper's numbers can be replicated and the fix quantified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PARQUET_DIR

FAR_FUTURE = pd.Timestamp("2099-12-31")


def _parse_wrds_date(s: pd.Series) -> pd.Series:
    """Parse WRDS-style '31jul2002' dates; blanks -> NaT."""
    return pd.to_datetime(s, format="%d%b%Y", errors="coerce")


def load_bonds(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PARQUET_DIR / "bond_returns.parquet")
    df["CUSIP"] = df["CUSIP"].astype("string").str.upper().str.zfill(9)
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = _parse_wrds_date(df["DATE"].astype("string"))
    return df


def load_link(path: Path | None = None) -> pd.DataFrame:
    link = pd.read_parquet(path or PARQUET_DIR / "bond_crsp_link.parquet")
    link["CUSIP"] = link["CUSIP"].astype("string").str.upper().str.zfill(9)
    for c in ("link_startdt", "link_enddt"):
        if not pd.api.types.is_datetime64_any_dtype(link[c]):
            link[c] = _parse_wrds_date(link[c].astype("string"))
    link["link_enddt"] = link["link_enddt"].fillna(FAR_FUTURE)
    link["link_startdt"] = link["link_startdt"].fillna(pd.Timestamp("1900-01-01"))
    return link


def load_dd(path: Path | None = None) -> pd.DataFrame:
    dd = pd.read_parquet(path or PARQUET_DIR / "Merton_dd.parquet")
    dd["PERMNO"] = pd.to_numeric(dd["PERMNO"], errors="coerce").astype("Int64")
    if not pd.api.types.is_datetime64_any_dtype(dd["DATE"]):
        dd["DATE"] = pd.to_datetime(dd["DATE"], errors="coerce")
    return dd


def link_bonds(bond: pd.DataFrame, link: pd.DataFrame,
               mode: str = "window") -> pd.DataFrame:
    """Attach PERMNO to each bond-month.

    mode="window": respect link validity windows; one issuer per
                   bond-month (ties broken by longest-lived link).
    mode="naive" : original script's CUSIP-only merge (can duplicate
                   rows and mis-assign issuers) -- for replication only.
    """
    if mode == "naive":
        link_simple = link[["CUSIP", "PERMNO"]].drop_duplicates()
        return bond.merge(link_simple, on="CUSIP", how="left")
    if mode != "window":
        raise ValueError(f"unknown link mode {mode!r}")
    lk = link[["CUSIP", "PERMNO", "link_startdt", "link_enddt"]].copy()
    lk["_span"] = (lk["link_enddt"] - lk["link_startdt"]).dt.days
    merged = bond.merge(lk, on="CUSIP", how="left")
    in_win = (merged["DATE"] >= merged["link_startdt"]) \
        & (merged["DATE"] <= merged["link_enddt"])
    merged.loc[~in_win, ["PERMNO", "_span"]] = np.nan
    # keep the best candidate per original bond row: valid window first,
    # then longest link span
    merged["_orig"] = merged.index
    merged = merged.sort_values(["_span"], ascending=False)
    keep = merged.drop_duplicates(subset=["CUSIP", "DATE", "ISSUE_ID"],
                                  keep="first")
    keep = keep.sort_index().drop(
        columns=["link_startdt", "link_enddt", "_span", "_orig"])
    keep["PERMNO"] = pd.to_numeric(keep["PERMNO"],
                                   errors="coerce").astype("Int64")
    return keep.reset_index(drop=True)


def merge_dd(panel: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    """Attach issuer DDCamp at PERMNO x calendar-month."""
    panel = panel.copy()
    panel["_ym"] = panel["DATE"].dt.year * 100 + panel["DATE"].dt.month
    dd = dd.copy()
    dd["_ym"] = dd["DATE"].dt.year * 100 + dd["DATE"].dt.month
    dd_sub = dd[["PERMNO", "_ym", "DDCamp"]].dropna().drop_duplicates(
        subset=["PERMNO", "_ym"])
    out = panel.merge(dd_sub, on=["PERMNO", "_ym"], how="left")
    return out.drop(columns=["_ym"])


def apply_filters(panel: pd.DataFrame) -> pd.DataFrame:
    """The paper's sample filters + derived variables (Section 2.2)."""
    p = panel.copy()
    if "RATING_CLASS" in p.columns:
        p = p[p["RATING_CLASS"].notna()
              & (p["RATING_CLASS"].astype(str) != "")]
    p = p[p["DATE"].notna()]
    for col in ("T_Spread", "RET_EOM", "DURATION", "AMOUNT_OUTSTANDING",
                "DDCamp", "TMT"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.dropna(subset=["T_Spread", "RET_EOM", "DURATION",
                         "AMOUNT_OUTSTANDING"])
    p["TMT_bucket"] = pd.cut(p["TMT"], bins=[0, 3, 7, 15, 100],
                             labels=["0-3y", "3-7y", "7-15y", ">15y"])
    p["log_size"] = np.log(p["AMOUNT_OUTSTANDING"])
    p = p.replace([np.inf, -np.inf], np.nan)
    p = p.sort_values(["ISSUE_ID", "DATE"])
    p["RET_EOM_next"] = p.groupby("ISSUE_ID")["RET_EOM"].shift(-1)
    return p.dropna(subset=["RET_EOM_next"]).reset_index(drop=True)


def build_panel(mode: str = "window", data_dir: Path | None = None) -> pd.DataFrame:
    """Full build: load -> link -> merge DD -> filter."""
    kw = {}
    if data_dir is not None:
        data_dir = Path(data_dir)
        bond = load_bonds(data_dir / "bond_returns.parquet")
        link = load_link(data_dir / "bond_crsp_link.parquet")
        dd = load_dd(data_dir / "Merton_dd.parquet")
    else:
        bond, link, dd = load_bonds(), load_link(), load_dd()
    panel = link_bonds(bond, link, mode=mode)
    panel = merge_dd(panel, dd)
    return apply_filters(panel)
