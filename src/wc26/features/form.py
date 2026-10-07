from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.features.elo import EloLookup
from wc26.features.history import TeamHistory

DECAY_HALF_LIFE_DAYS = 365.0


def form_features(
    hist: TeamHistory,
    elo: EloLookup,
    team: str,
    as_of: pd.Timestamp,
    tournament_start: pd.Timestamp | None = None,
) -> dict:
    past = hist.before(team, as_of)
    out: dict[str, float] = {}
    for window in (5, 10):
        recent = past.tail(window)
        if len(recent) == 0:
            out.update(
                {
                    f"form{window}_ppg": np.nan,
                    f"form{window}_gfpg": np.nan,
                    f"form{window}_gapg": np.nan,
                    f"form{window}_cs_rate": np.nan,
                    f"form{window}_vs_expected": np.nan,
                }
            )
            continue
        ages = (as_of - recent["date"]).dt.days.to_numpy()
        w = 0.5 ** (ages / DECAY_HALF_LIFE_DAYS)
        w = w / w.sum()
        out[f"form{window}_ppg"] = float((recent["points"].to_numpy() * w).sum())
        out[f"form{window}_gfpg"] = float((recent["gf"].to_numpy() * w).sum())
        out[f"form{window}_gapg"] = float((recent["ga"].to_numpy() * w).sum())
        out[f"form{window}_cs_rate"] = float(((recent["ga"].to_numpy() == 0) * w).sum())
        expected_pts = []
        for r in recent.itertuples():
            e = 1.0 / (
                1.0 + 10 ** (-(elo.rating(r.team, r.date) - elo.rating(r.opponent, r.date)) / 400)
            )
            expected_pts.append(3 * e)
        actual = recent["points"].to_numpy()
        out[f"form{window}_vs_expected"] = float(((actual - np.array(expected_pts)) * w).sum())

    if len(past):
        out["rest_days"] = float(min((as_of - past["date"].iloc[-1]).days, 60))
    else:
        out["rest_days"] = 60.0
    if tournament_start is not None:
        out["tournament_matches_played"] = float(
            (past["date"] >= tournament_start).sum()
        )
    else:
        out["tournament_matches_played"] = 0.0
    return out
