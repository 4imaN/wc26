from __future__ import annotations

import functools
from collections import Counter

import numpy as np
import pandas as pd

from wc26 import config
from wc26.ingest.results import load_shootouts
from wc26.model import train as T
from wc26.model.predict import Predictor


@functools.lru_cache
def _shootout_winrates() -> dict[str, float]:
    so = load_shootouts()
    counts: dict[str, list[int]] = {}
    for r in so.itertuples():
        for team in (r.home_team, r.away_team):
            won = int(r.winner == team)
            counts.setdefault(team, [0, 0])
            counts[team][0] += won
            counts[team][1] += 1
    return {t: w / n for t, (w, n) in counts.items() if n >= 3}


def penalty_win_prob(team_a: str, team_b: str) -> float:
    wr = _shootout_winrates()
    p = 0.5 + 0.5 * (wr.get(team_a, 0.5) - wr.get(team_b, 0.5))
    return float(np.clip(p, 0.35, 0.65))


def knockout_win_prob(pred: Predictor, team_a: str, team_b: str, date, venue) -> dict:
    """P(team_a advances) resolving draws via extra time then penalties."""
    r = pred.predict_match(team_a, team_b, date, venue)
    lam_a, lam_b = r["exp_goals_a"] / 3, r["exp_goals_b"] / 3
    et = T.score_matrix(lam_a, lam_b, 0.0)
    et_win, et_draw, et_loss = T.wdl_from_matrix(et)
    pen = penalty_win_prob(team_a, team_b)
    p_advance = r["p_win"] + r["p_draw"] * (et_win + et_draw * pen)
    return {**r, "p_advance": float(p_advance)}


def remaining_fixtures(matches: pd.DataFrame, as_of: str) -> pd.DataFrame:
    wc = matches[
        (matches["tournament"] == "FIFA World Cup") & (matches["date"] >= "2026-06-01")
    ]
    return wc[wc["date"] >= as_of]


def simulate_remaining(
    pred: Predictor,
    matches: pd.DataFrame,
    weather: pd.DataFrame,
    as_of: str = "2026-07-15",
    n: int = 10000,
    rng_seed: int = 7,
) -> dict:
    """Simulate the remaining SF2 / third-place / final bracket."""
    rem = remaining_fixtures(matches, as_of).sort_values("date")
    wmap = weather.set_index("match_id") if "match_id" in weather.columns else weather

    def venue_of(match) -> dict | None:
        if match.match_id in wmap.index:
            return wmap.loc[match.match_id].to_dict()
        return None

    fixtures = list(rem.itertuples())
    sf2 = next(f for f in fixtures if pd.notna(f.away_team))
    third = next(f for f in fixtures if f.date.strftime("%m-%d") == "07-18")
    final = next(f for f in fixtures if f.date.strftime("%m-%d") == "07-19")

    p_sf2 = knockout_win_prob(pred, sf2.home_team, sf2.away_team, sf2.date, venue_of(sf2))

    finalists = {}
    third_probs = {}
    for team, p_reach in ((sf2.home_team, p_sf2["p_advance"]), (sf2.away_team, 1 - p_sf2["p_advance"])):
        kf = knockout_win_prob(pred, final.home_team, team, final.date, venue_of(final))
        finalists[team] = {"p_reach_final": p_reach, "p_champ_spain_side": kf}
        third_probs[team] = knockout_win_prob(
            pred, third.home_team, team, third.date, venue_of(third)
        )

    rng = np.random.default_rng(rng_seed)
    champs: Counter = Counter()
    thirds: Counter = Counter()
    for _ in range(n):
        sf2_winner = sf2.home_team if rng.random() < p_sf2["p_advance"] else sf2.away_team
        sf2_loser = sf2.away_team if sf2_winner == sf2.home_team else sf2.home_team
        kf = finalists[sf2_winner]["p_champ_spain_side"]
        champ = final.home_team if rng.random() < kf["p_advance"] else sf2_winner
        champs[champ] += 1
        k3 = third_probs[sf2_loser]
        third_winner = third.home_team if rng.random() < k3["p_advance"] else sf2_loser
        thirds[third_winner] += 1

    return {
        "as_of": as_of,
        "sf2": p_sf2,
        "p_champion": {t: c / n for t, c in champs.most_common()},
        "p_third_place": {t: c / n for t, c in thirds.most_common()},
        "finalist_details": finalists,
    }
