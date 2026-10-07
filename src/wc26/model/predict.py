from __future__ import annotations

import numpy as np
import pandas as pd

from wc26.features.build import FeatureBuilder
from wc26.model import train as T


class Predictor:
    def __init__(
        self,
        builder: FeatureBuilder,
        model,
        rho: float,
        temperature: float,
    ):
        self.builder = builder
        self.model = model
        self.rho = rho
        self.temperature = temperature

    def _pair_frame(
        self,
        team_a: str,
        team_b: str,
        date: pd.Timestamp,
        venue: dict | None,
        host_country: str | None,
    ) -> pd.DataFrame:
        wc_year = self.builder.wc_year_of(date)
        rows = []
        for team, opp in ((team_a, team_b), (team_b, team_a)):
            is_home = host_country is not None and team == host_country
            rows.append(
                self.builder.side_features(team, opp, date, is_home, wc_year, venue)
            )
        df = pd.DataFrame(rows)
        for col in self.model.feature_names_in_:
            if col not in df.columns:
                df[col] = np.nan
        return df[list(self.model.feature_names_in_)]

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        date: str | pd.Timestamp,
        venue: dict | None = None,
        host_country: str | None = "United States",
    ) -> dict:
        date = pd.Timestamp(date)
        pair = self._pair_frame(team_a, team_b, date, venue, host_country)
        lam = self.model.predict(pair)
        m = T.score_matrix(float(lam[0]), float(lam[1]), self.rho)
        w, d, l = T.wdl_from_matrix(m)
        probs = np.array([w, d, l])
        if self.temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-9, 1)) / self.temperature
            probs = np.exp(logits) / np.exp(logits).sum()
        top = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1][:5], m.shape))[0]
        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_win": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_loss": float(probs[2]),
            "exp_goals_a": float(lam[0]),
            "exp_goals_b": float(lam[1]),
            "matrix": m,
            "top_scorelines": [(int(i), int(j), float(m[i, j])) for i, j in top],
        }
