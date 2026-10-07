from __future__ import annotations

import numpy as np
import pandas as pd

from wc26 import config
from wc26.features import motivation as mot
from wc26.features import squad_quality
from wc26.features.elo import EloLookup, compute_elo_history
from wc26.features.form import form_features
from wc26.features.h2h import h2h_features, wc_knockout_revenge
from wc26.features.history import TeamHistory
from wc26.features.style import style_features
from wc26.features.discipline import MatchStats

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 3.0,
    "FIFA World Cup qualification": 2.0,
    "UEFA Euro": 2.5,
    "Copa América": 2.5,
    "African Cup of Nations": 2.0,
    "AFC Asian Cup": 2.0,
    "Gold Cup": 2.0,
    "CONCACAF Championship": 2.0,
    "UEFA Nations League": 2.0,
    "CONCACAF Nations League": 1.5,
    "Friendly": 1.0,
}


class FeatureBuilder:
    def __init__(self, matches: pd.DataFrame, as_of_limit: pd.Timestamp | None = None):
        """as_of_limit truncates ALL history sources — nothing on/after it is visible."""
        if as_of_limit is not None:
            visible = matches[matches["date"] < as_of_limit].copy()
        else:
            visible = matches.copy()
        self.matches = visible
        self.elo_hist = compute_elo_history(visible)
        self.elo = EloLookup(self.elo_hist)
        self.hist = TeamHistory(visible)
        try:
            self.weather = pd.read_parquet(
                config.PROCESSED_DIR / "wc_match_weather.parquet"
            ).set_index("match_id")
        except FileNotFoundError:
            self.weather = None
        self.match_stats = MatchStats()

    def wc_year_of(self, date: pd.Timestamp) -> int | None:
        for year, start in mot.WC_STARTS.items():
            if start <= date <= start + pd.Timedelta(days=45):
                return year
        return None

    def side_features(
        self,
        team: str,
        opp: str,
        date: pd.Timestamp,
        is_home: bool,
        wc_year: int | None,
        venue: dict | None,
    ) -> dict:
        f: dict[str, float] = {"is_home": float(is_home)}
        elo_t = self.elo.rating(team, date)
        elo_o = self.elo.rating(opp, date)
        f["elo"] = elo_t
        f["elo_diff"] = elo_t - elo_o
        f["elo_mom_diff"] = self.elo.change(team, date) - self.elo.change(opp, date)

        tstart = mot.WC_STARTS.get(wc_year) if wc_year else None
        ft = form_features(self.hist, self.elo, team, date, tstart)
        fo = form_features(self.hist, self.elo, opp, date, tstart)
        for k in ("form5_ppg", "form5_gfpg", "form5_gapg", "form5_cs_rate", "form5_vs_expected",
                  "form10_ppg", "form10_gfpg", "form10_gapg", "form10_vs_expected"):
            f[f"{k}_diff"] = ft[k] - fo[k]
        f["form10_gfpg"] = ft["form10_gfpg"]
        f["opp_form10_gapg"] = fo["form10_gapg"]
        f["rest_diff"] = ft["rest_days"] - fo["rest_days"]
        f["tourn_matches"] = ft["tournament_matches_played"]
        f["fatigue"] = ft["tournament_matches_played"] / max(ft["rest_days"], 1.0)
        f["fatigue_diff"] = f["fatigue"] - fo["tournament_matches_played"] / max(fo["rest_days"], 1.0)

        f.update(h2h_features(self.hist, team, opp, date))

        st_t = style_features(self.hist, team, date)
        st_o = style_features(self.hist, opp, date)
        for k, v in st_t.items():
            f[k] = v
            f[f"opp_{k}"] = st_o[k]

        if wc_year is not None:
            d_t = self.match_stats.team_features(team, date)
            d_o = self.match_stats.team_features(opp, date)
            for k, tv in d_t.items():
                f[k] = tv
                f[f"{k}_diff"] = tv - d_o[k]
            referee = (venue or {}).get("referee") or self.match_stats.referee_for(
                date, team, opp
            )
            f.update(self.match_stats.ref_strictness(referee, date))

            sq_t = squad_quality.squad_features(team, wc_year, date.month)
            sq_o = squad_quality.squad_features(opp, wc_year, date.month)
            for k in set(sq_t) | set(sq_o):
                tv, ov = sq_t.get(k, np.nan), sq_o.get(k, np.nan)
                f[k] = tv
                f[f"{k}_diff"] = tv - ov

            m_t = mot.motivation_features(team, wc_year, date, elo_t)
            m_o = mot.motivation_features(opp, wc_year, date, elo_o)
            for k in set(m_t) | set(m_o):
                f[k] = m_t.get(k, np.nan)
                f[f"{k}_diff"] = m_t.get(k, np.nan) - m_o.get(k, np.nan)
            f["mot_revenge"] = wc_knockout_revenge(self.matches, team, opp, date)

            if venue is not None:
                alt = venue.get("venue_altitude_m", np.nan)
                cc = float(venue.get("climate_controlled", 0))
                f["venue_known"] = 1.0
                f["venue_altitude"] = alt
                f["altitude_delta"] = alt - sq_t.get("sq_accustomed_altitude", np.nan)
                f["altitude_delta_diff"] = (
                    (alt - sq_t.get("sq_accustomed_altitude", np.nan))
                    - (alt - sq_o.get("sq_accustomed_altitude", np.nan))
                )
                f["altitude_delta_origin"] = alt - sq_t.get("sq_origin_altitude", np.nan)
                f["altitude_delta_origin_diff"] = sq_o.get(
                    "sq_origin_altitude", np.nan
                ) - sq_t.get("sq_origin_altitude", np.nan)
                temp = venue.get("apparent_temp_max_c", np.nan)
                if cc:
                    f["heat_delta"] = 0.0
                    f["heat_delta_diff"] = 0.0
                    f["heat_delta_origin"] = 0.0
                    f["heat_delta_origin_diff"] = 0.0
                    f["match_temp"] = 21.0
                else:
                    f["match_temp"] = temp
                    f["heat_delta"] = (
                        np.nan if temp is None else temp - sq_t.get("sq_climate_baseline", np.nan)
                    )
                    f["heat_delta_diff"] = f["heat_delta"] - (
                        temp - sq_o.get("sq_climate_baseline", np.nan)
                    )
                    f["heat_delta_origin"] = temp - sq_t.get("sq_origin_temp", np.nan)
                    f["heat_delta_origin_diff"] = sq_o.get(
                        "sq_origin_temp", np.nan
                    ) - sq_t.get("sq_origin_temp", np.nan)
                f["humidity"] = 21.0 if cc else venue.get("humidity_mean", np.nan)
        else:
            f["venue_known"] = 0.0
        return f

    def _venue_for_match(self, match) -> dict | None:
        if self.weather is None or match.match_id not in self.weather.index:
            return None
        return self.weather.loc[match.match_id].to_dict()

    def _is_home(self, team: str, match) -> bool:
        if not match.neutral:
            return team == match.home_team
        return team == match.country

    def build_training_matrix(self, start: str = "2010-01-01") -> pd.DataFrame:
        rows = []
        played = self.matches.dropna(subset=["home_score", "away_score"])
        played = played[played["date"] >= start]
        for match in played.itertuples():
            wc_year = (
                self.wc_year_of(match.date)
                if match.tournament == "FIFA World Cup"
                else None
            )
            venue = self._venue_for_match(match) if wc_year else None
            base_weight = TOURNAMENT_WEIGHT.get(match.tournament, 1.2)
            for side, (team, opp, gf, ga) in enumerate(
                (
                    (match.home_team, match.away_team, match.home_score, match.away_score),
                    (match.away_team, match.home_team, match.away_score, match.home_score),
                )
            ):
                f = self.side_features(
                    team, opp, match.date, self._is_home(team, match), wc_year, venue
                )
                f.update(
                    match_id=match.match_id,
                    date=match.date,
                    team=team,
                    opponent=opp,
                    side=side,
                    goals=float(gf),
                    goals_against=float(ga),
                    weight=base_weight,
                    is_wc=float(wc_year is not None),
                )
                rows.append(f)
        df = pd.DataFrame(rows)
        return df


FEATURE_EXCLUDE = {
    "match_id", "date", "team", "opponent", "side", "goals", "goals_against", "weight", "is_wc",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
