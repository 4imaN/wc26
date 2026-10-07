from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.features.history import TeamHistory

WINDOW = 20
DECAY_HALF_LIFE_DAYS = 730.0


def style_features(hist: TeamHistory, team: str, as_of: pd.Timestamp) -> dict:
    """Play-style proxies from recent results: tempo, attack share, draw
    propensity, and scoring volatility (all as-of-date, time-decayed)."""
    past = hist.before(team, as_of).tail(WINDOW)
    if len(past) < 5:
        return {
            "style_tempo": np.nan,
            "style_attack_share": np.nan,
            "style_draw_rate": np.nan,
            "style_margin_std": np.nan,
        }
    ages = (as_of - past["date"]).dt.days.to_numpy()
    w = 0.5 ** (ages / DECAY_HALF_LIFE_DAYS)
    w = w / w.sum()
    gf = past["gf"].to_numpy(dtype=float)
    ga = past["ga"].to_numpy(dtype=float)
    total = gf + ga
    margin = gf - ga
    wmean_margin = float((margin * w).sum())
    return {
        "style_tempo": float((total * w).sum()),
        "style_attack_share": float((gf * w).sum() / max((total * w).sum(), 0.5)),
        "style_draw_rate": float(((gf == ga).astype(float) * w).sum()),
        "style_margin_std": float(np.sqrt(((margin - wmean_margin) ** 2 * w).sum())),
    }
