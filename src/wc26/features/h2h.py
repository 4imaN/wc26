from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.features.history import TeamHistory

DECAY_PER_YEAR = 0.85


def h2h_features(hist: TeamHistory, team_a: str, team_b: str, as_of: pd.Timestamp) -> dict:
    """Head-to-head from team_a's perspective, excluding World Cup finals matches."""
    past = hist.before(team_a, as_of)
    meetings = past[past["opponent"] == team_b]
    non_wc = meetings[meetings["tournament"] != "FIFA World Cup"]

    out: dict[str, float] = {"h2h_n_meetings": float(len(non_wc))}
    if len(non_wc) == 0:
        out.update(
            {
                "h2h_winrate": np.nan,
                "h2h_goal_diff": np.nan,
                "h2h_comp_winrate": np.nan,
                "h2h_years_since_last": 50.0,
            }
        )
        return out

    years_ago = (as_of - non_wc["date"]).dt.days.to_numpy() / 365.25
    w = DECAY_PER_YEAR**years_ago
    w = w / w.sum()
    win = (non_wc["gf"] > non_wc["ga"]).to_numpy().astype(float)
    draw = (non_wc["gf"] == non_wc["ga"]).to_numpy().astype(float)
    out["h2h_winrate"] = float(((win + 0.5 * draw) * w).sum())
    out["h2h_goal_diff"] = float(((non_wc["gf"] - non_wc["ga"]).to_numpy() * w).sum())
    out["h2h_years_since_last"] = float(min(years_ago.min(), 50.0))

    comp = non_wc[non_wc["tournament"] != "Friendly"]
    if len(comp):
        cy = (as_of - comp["date"]).dt.days.to_numpy() / 365.25
        cw = DECAY_PER_YEAR**cy
        cw = cw / cw.sum()
        cwin = (comp["gf"] > comp["ga"]).to_numpy().astype(float)
        cdraw = (comp["gf"] == comp["ga"]).to_numpy().astype(float)
        out["h2h_comp_winrate"] = float(((cwin + 0.5 * cdraw) * cw).sum())
    else:
        out["h2h_comp_winrate"] = np.nan
    return out


def wc_knockout_revenge(matches: pd.DataFrame, team_a: str, team_b: str, as_of: pd.Timestamp) -> float:
    """1.0 if team_a lost a World Cup match to team_b in the previous 12 years."""
    lo = as_of - pd.Timedelta(days=12 * 365)
    wc = matches[
        (matches["tournament"] == "FIFA World Cup")
        & (matches["date"] >= lo)
        & (matches["date"] < as_of)
    ]
    lost = wc[
        ((wc["home_team"] == team_a) & (wc["away_team"] == team_b) & (wc["home_score"] < wc["away_score"]))
        | ((wc["away_team"] == team_a) & (wc["home_team"] == team_b) & (wc["away_score"] < wc["home_score"]))
    ]
    return 1.0 if len(lost) else 0.0
