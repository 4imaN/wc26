from __future__ import annotations

import pandas as pd

from wc26.config import PROCESSED_DIR, RAW_DIR, canonicalize_team_name

RESULTS_DIR = RAW_DIR / "results"
GITHUB_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"]


def download() -> None:
    import requests

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        r = requests.get(f"{GITHUB_BASE}/{f}", timeout=60)
        r.raise_for_status()
        (RESULTS_DIR / f).write_bytes(r.content)


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "results.csv", parse_dates=["date"])
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(lambda x: canonicalize_team_name(x) if pd.notna(x) else x)
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index
    return df


def load_shootouts() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "shootouts.csv", parse_dates=["date"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].map(lambda x: canonicalize_team_name(x) if pd.notna(x) else x)
    return df


def build() -> pd.DataFrame:
    df = load_matches()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "matches.parquet")
    return df


def wc_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    lo, hi = f"{year}-01-01", f"{year}-12-31"
    return df[(df["tournament"] == "FIFA World Cup") & df["date"].between(lo, hi)]
