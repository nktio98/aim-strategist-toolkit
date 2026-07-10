"""Corporate bond mispricing research pipeline (Tiosudarmin).

Refactor of the monolithic corpbondnewnorman.py into a testable package:

    panel      -- data loading + CUSIP->PERMNO linking (window-correct)
    signals    -- monthly cross-sectional spread model -> mispricing residual
    fmb        -- generic Fama-MacBeth engine (all paper specs are configs)
    portfolios -- quintile sorts, long-short strategies, MKT/CRD/TERM
                  factors, three-factor NW regressions, two-pass FMB
    artifacts  -- run the full pipeline and save PUBLIC-SAFE aggregate
                  tables/figures (no bond-level licensed data ever leaves)

The licensed WRDS inputs live in SFU-bondpaper/parquet/ (gitignored).
Every module is unit-tested on synthetic panels in tests/test_research.py.
"""

__version__ = "0.1.0"
