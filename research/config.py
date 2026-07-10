"""Paths and constants for the research pipeline."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "SFU-bondpaper"          # licensed, gitignored
PARQUET_DIR = DATA_DIR / "parquet"
ARTIFACT_DIR = REPO_ROOT / "research_artifacts"  # public-safe aggregates

MIN_OBS_CROSS_SECTION = 40      # min bonds per month for regressions
MIN_OBS_QUINTILES = 50          # min bonds per month for quintile sorts
WINSOR_Q = (0.01, 0.99)
IG_LABEL, HY_LABEL = "0.IG", "1.HY"
