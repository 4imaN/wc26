from __future__ import annotations

import pandas as pd

from wc26 import config
from wc26.features.build import FeatureBuilder
from wc26.model import train as T
from wc26.model.predict import Predictor

VAL_WINDOWS = {
    "2022": ("2022-11-20", "2022-12-19"),
    "2026": ("2026-06-11", "2026-07-15"),
}


def _load_features(path_suffix: str = "") -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / f"features{path_suffix}.parquet")


def run_training(validate: str = "2026") -> dict:
    df = _load_features()
    lo, hi = VAL_WINDOWS[validate]
    val = df[(df["is_wc"] == 1) & df["date"].between(lo, hi)]
    train_df = df[df["date"] < lo]

    model = T.train_poisson(train_df)
    rho = T.fit_rho(model, val)
    temperature = T.fit_temperature(model, val, rho)
    gbm_ll = T._pairs_log_loss(model, val, rho, temperature)
    elo_ll = T.elo_baseline_log_loss(val)

    print(f"validation ({validate} WC, {val['match_id'].nunique()} matches):")
    print(f"  GBM log loss:          {gbm_ll:.4f} (rho={rho:.2f}, T={temperature:.2f})")
    print(f"  Elo baseline log loss: {elo_ll:.4f}")

    if validate == "2026":
        # rho/T must come from the 2026 window only: it is the sole data that is
        # out-of-sample for the validation model (2022 is in its training set).
        full_model = T.train_poisson(df)
        T.save(full_model, rho, temperature, "prod")
        print(
            f"saved production model (rho={rho:.2f}, T={temperature:.2f}, "
            "calibrated out-of-sample on 2026 WC, trained on all data through 2026-07-14)"
        )
    return {"gbm_ll": gbm_ll, "elo_ll": elo_ll, "rho": rho, "temperature": temperature}


def load_predictor() -> Predictor:
    matches = pd.read_parquet(config.PROCESSED_DIR / "matches.parquet")
    builder = FeatureBuilder(matches)
    model, rho, temperature = T.load("prod")
    return Predictor(builder, model, rho, temperature)


def venue_for(city: str, date: str) -> dict | None:
    from wc26.ingest.weather import match_day_weather

    v = config.venue_for_city(city)
    if v is None:
        return None
    venues = pd.read_parquet(config.PROCESSED_DIR / "venues.parquet")
    alt = venues[venues["city"] == config.CITY_ALIASES.get(city, city)]["altitude_m"].iloc[0]
    w = match_day_weather(v["lat"], v["lon"], date, v["tz"], forecast=date >= "2026-07-15")
    return {
        "venue_altitude_m": float(alt),
        "climate_controlled": float(v["climate_controlled"]),
        **w,
    }
