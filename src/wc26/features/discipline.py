from __future__ import annotations

import numpy as np
import pandas as pd

from wc26 import config

TEAM_STATS = [
    "fouls",
    "yellows",
    "reds",
    "offsides",
    "possession",
    "shots",
    "shots_on_target",
    "pass_pct",
    "tackles",
    "saves",
]


class MatchStats:
    """As-of aggregates of per-match team boxscore stats (WC 2022 + 2026) and
    referee strictness, from ESPN data."""

    def __init__(self):
        try:
            df = pd.read_parquet(config.PROCESSED_DIR / "referee_cards.parquet")
        except FileNotFoundError:
            self.long = None
            self.by_match = {}
            return
        frames = []
        for side, opp in (("home", "away"), ("away", "home")):
            frames.append(
                pd.DataFrame(
                    {
                        "date": df["date"],
                        "team": df[f"{side}_team"],
                        "referee": df["referee"],
                        **{k: df[f"{side}_{k}"] for k in TEAM_STATS},
                    }
                )
            )
        self.long = pd.concat(frames).sort_values("date").reset_index(drop=True)
        self.by_match = {
            (r.date, r.home_team, r.away_team): r for r in df.itertuples()
        }

    def referee_for(self, date: pd.Timestamp, home: str, away: str) -> str | None:
        rec = self.by_match.get((date, home, away)) or self.by_match.get(
            (date, away, home)
        )
        return getattr(rec, "referee", None) if rec is not None else None

    def team_features(self, team: str, as_of: pd.Timestamp) -> dict:
        out = {f"disc_{k}_pm": np.nan for k in TEAM_STATS}
        if self.long is None:
            return out
        past = self.long[(self.long["team"] == team) & (self.long["date"] < as_of)]
        if len(past) == 0:
            return out
        for k in TEAM_STATS:
            vals = past[k].dropna()
            if len(vals):
                out[f"disc_{k}_pm"] = float(vals.mean())
        return out

    def ref_strictness(self, referee: str | None, as_of: pd.Timestamp) -> dict:
        out = {"ref_yellows_pm": np.nan, "ref_fouls_pm": np.nan, "ref_n_matches": 0.0}
        if self.long is None or not referee:
            return out
        past = self.long[
            (self.long["referee"] == referee) & (self.long["date"] < as_of)
        ]
        n_matches = len(past) / 2  # long format has two rows per match
        out["ref_n_matches"] = float(n_matches)
        if n_matches >= 1:
            out["ref_yellows_pm"] = float(past["yellows"].dropna().sum() / n_matches)
            out["ref_fouls_pm"] = float(past["fouls"].dropna().sum() / n_matches)
        return out
