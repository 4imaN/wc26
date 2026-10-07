from __future__ import annotations

import numpy as np
import pandas as pd


class TeamHistory:
    """Per-team long view of completed matches for fast as-of slicing."""

    def __init__(self, matches: pd.DataFrame):
        played = matches.dropna(subset=["home_score", "away_score"])
        home = pd.DataFrame(
            {
                "date": played["date"],
                "team": played["home_team"],
                "opponent": played["away_team"],
                "gf": played["home_score"],
                "ga": played["away_score"],
                "tournament": played["tournament"],
                "match_id": played["match_id"],
            }
        )
        away = pd.DataFrame(
            {
                "date": played["date"],
                "team": played["away_team"],
                "opponent": played["home_team"],
                "gf": played["away_score"],
                "ga": played["home_score"],
                "tournament": played["tournament"],
                "match_id": played["match_id"],
            }
        )
        long = pd.concat([home, away]).dropna(subset=["team"]).sort_values("date")
        long["points"] = np.select(
            [long["gf"] > long["ga"], long["gf"] == long["ga"]], [3.0, 1.0], default=0.0
        )
        self._by_team: dict[str, pd.DataFrame] = {
            team: grp.reset_index(drop=True) for team, grp in long.groupby("team")
        }

    def before(self, team: str, as_of: pd.Timestamp) -> pd.DataFrame:
        grp = self._by_team.get(team)
        if grp is None:
            return pd.DataFrame(
                columns=["date", "team", "opponent", "gf", "ga", "tournament", "match_id", "points"]
            )
        idx = grp["date"].searchsorted(as_of)
        return grp.iloc[:idx]
