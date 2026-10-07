from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.config import PROCESSED_DIR

INIT_RATING = 1500.0
NEW_TEAM_RATING = 1300.0
HOME_ADVANTAGE = 100.0

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Championship": 50,
    "Gold Cup": 50,
    "Oceania Nations Cup": 50,
    "Confederations Cup": 40,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "UEFA Euro qualification": 40,
    "Copa América qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "Friendly": 20,
}
K_DEFAULT = 30


def _goal_multiplier(margin: int) -> float:
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def compute_elo_history(
    matches: pd.DataFrame, start: str = "1946-01-01", persist: bool = False
) -> pd.DataFrame:
    """Replay all matches chronologically; return per-match PRE-game ratings."""
    df = matches[matches["date"] >= start].sort_values(["date", "match_id"])
    ratings: dict[str, float] = {}
    rows = []
    for m in df.itertuples():
        home, away = m.home_team, m.away_team
        if pd.isna(home) or pd.isna(away):
            continue
        r_home = ratings.get(home, NEW_TEAM_RATING if ratings else INIT_RATING)
        r_away = ratings.get(away, NEW_TEAM_RATING if ratings else INIT_RATING)
        row = {
            "match_id": m.match_id,
            "date": m.date,
            "home_team": home,
            "away_team": away,
            "home_elo_pre": r_home,
            "away_elo_pre": r_away,
            "home_elo_post": r_home,
            "away_elo_post": r_away,
        }
        if not (pd.isna(m.home_score) or pd.isna(m.away_score)):
            ha = 0.0 if m.neutral else HOME_ADVANTAGE
            expected_home = 1.0 / (1.0 + 10 ** (-((r_home + ha) - r_away) / 400))
            if m.home_score > m.away_score:
                w = 1.0
            elif m.home_score < m.away_score:
                w = 0.0
            else:
                w = 0.5
            k = K_BY_TOURNAMENT.get(m.tournament, K_DEFAULT)
            g = _goal_multiplier(int(abs(m.home_score - m.away_score)))
            delta = k * g * (w - expected_home)
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta
            row["home_elo_post"] = ratings[home]
            row["away_elo_post"] = ratings[away]
        rows.append(row)
    hist = pd.DataFrame(rows)
    if persist:
        hist.to_parquet(PROCESSED_DIR / "elo_history.parquet")
    return hist


class EloLookup:
    """Leakage-safe rating lookup built from the pre-game rating history."""

    def __init__(self, hist: pd.DataFrame):
        home = hist[["date", "home_team", "home_elo_post"]].rename(
            columns={"home_team": "team", "home_elo_post": "elo"}
        )
        away = hist[["date", "away_team", "away_elo_post"]].rename(
            columns={"away_team": "team", "away_elo_post": "elo"}
        )
        long = pd.concat([home, away]).sort_values("date")
        self._by_team = {
            team: (grp["date"].to_numpy(), grp["elo"].to_numpy())
            for team, grp in long.groupby("team")
        }

    def rating(self, team: str, as_of: pd.Timestamp) -> float:
        entry = self._by_team.get(team)
        if entry is None:
            return NEW_TEAM_RATING
        dates, elos = entry
        idx = np.searchsorted(dates, np.datetime64(as_of)) - 1
        if idx < 0:
            return NEW_TEAM_RATING
        return float(elos[idx])

    def change(self, team: str, as_of: pd.Timestamp, days: int = 90) -> float:
        return self.rating(team, as_of) - self.rating(team, as_of - pd.Timedelta(days=days))
