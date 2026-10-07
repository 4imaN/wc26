from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from wc26 import config

WC_STAGES = {
    2022: [
        ("2022-11-20", "2022-12-02", 0.0),
        ("2022-12-03", "2022-12-06", 2.0),
        ("2022-12-09", "2022-12-10", 3.0),
        ("2022-12-13", "2022-12-14", 4.0),
        ("2022-12-17", "2022-12-17", 4.0),
        ("2022-12-18", "2022-12-18", 5.0),
    ],
    2026: [
        ("2026-06-11", "2026-06-27", 0.0),
        ("2026-06-28", "2026-07-03", 1.0),
        ("2026-07-04", "2026-07-07", 2.0),
        ("2026-07-09", "2026-07-11", 3.0),
        ("2026-07-14", "2026-07-15", 4.0),
        ("2026-07-18", "2026-07-18", 4.0),
        ("2026-07-19", "2026-07-19", 5.0),
    ],
}

WC_STARTS = {2022: pd.Timestamp("2022-11-20"), 2026: pd.Timestamp("2026-06-11")}
PREVIOUS_CHAMPION = {2022: "France", 2026: "Argentina"}


def wc_stage(date: pd.Timestamp) -> float:
    for year, ranges in WC_STAGES.items():
        for lo, hi, stage in ranges:
            if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
                return stage
    return 0.0


@functools.lru_cache
def _profiles() -> pd.DataFrame:
    return config.country_profiles().set_index("team")


def motivation_features(team: str, wc_year: int, date: pd.Timestamp, elo_rating: float) -> dict:
    prof = _profiles()
    if team not in prof.index:
        return {}
    p = prof.loc[team]
    titles = float(p["titles_won"])
    years_since = float(wc_year - p["last_title_year"]) if titles > 0 else 100.0
    hosts = str(p["host_years"]) if pd.notna(p["host_years"]) else ""
    is_host = float(str(wc_year) in hosts.split(";"))
    first_timer = float(p["first_wc_year"] == wc_year)
    stage = wc_stage(date)
    elo_pct = float(np.clip((elo_rating - 1300) / 700, 0, 1))
    return {
        "mot_is_host": is_host,
        "mot_titles": titles,
        "mot_years_since_title": years_since,
        "mot_first_timer": first_timer,
        "mot_defending_champ": float(PREVIOUS_CHAMPION.get(wc_year) == team),
        "mot_stage": stage,
        "mot_is_knockout": float(stage > 0),
        "mot_drought_pressure": float(np.log1p(years_since) * elo_pct),
    }
