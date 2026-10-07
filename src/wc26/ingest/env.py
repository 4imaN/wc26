from __future__ import annotations

import pandas as pd

from wc26 import config
from wc26.ingest import weather
from wc26.ingest.weather import _cached_get, ARCHIVE_URL


def season_months(start: int, end: int) -> list[int]:
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 13)) + list(range(1, end + 1))


def _decade_daily_max(lat: float, lon: float, label: str) -> pd.DataFrame:
    data = _cached_get(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "daily": "temperature_2m_max",
            "timezone": "UTC",
        },
        f"decade_{label}_{lat:.2f}_{lon:.2f}",
    )
    d = data["daily"]
    df = pd.DataFrame({"date": d["time"], "tmax": d["temperature_2m_max"]})
    df["month"] = df["date"].str.slice(5, 7).astype(int)
    return df.dropna()


def build_league_env() -> pd.DataFrame:
    """Per league country: altitude and season-months climate baseline."""
    rows = []
    for _, lp in config.league_profiles().iterrows():
        lat, lon = lp["lat"], lp["lon"]
        label = lp["league_country"].replace(" ", "_")
        alt = weather.elevation(lat, lon)
        daily = _decade_daily_max(lat, lon, label)
        months = season_months(int(lp["season_start_month"]), int(lp["season_end_month"]))
        season_temp = daily[daily["month"].isin(months)]["tmax"].mean()
        summer_share = daily[daily["month"].isin(months)]["tmax"].ge(28).mean()
        rows.append(
            {
                "league_country": lp["league_country"],
                "altitude_m": alt,
                "season_temp_c": round(float(season_temp), 2),
                "hot_share": round(float(summer_share), 3),
            }
        )
    df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_DIR / "league_env.parquet")
    return df


def build_origin_env() -> pd.DataFrame:
    """Climate baseline per birth country (year-round, where players grew up).

    Reuses league_env rep-city data when the birth country hosts a profiled
    league; otherwise fetches a decade of daily temps at the mean coordinates
    of that country's players' birth cities."""
    bp = pd.read_parquet(config.PROCESSED_DIR / "birthplaces.parquet")
    league_env = pd.read_parquet(config.PROCESSED_DIR / "league_env.parquet")
    lp = config.league_profiles()

    rows = []
    for country, grp in bp.dropna(subset=["birth_country"]).groupby("birth_country"):
        if country in set(league_env["league_country"]):
            prof = lp[lp["league_country"] == country].iloc[0]
            lat, lon = prof["lat"], prof["lon"]
        else:
            coords = grp.dropna(subset=["birth_lat", "birth_lon"])
            if len(coords) == 0:
                continue
            lat, lon = coords["birth_lat"].mean(), coords["birth_lon"].mean()
        daily = _decade_daily_max(lat, lon, f"origin_{country.replace(' ', '_')}")
        rows.append(
            {
                "birth_country": country,
                "origin_temp_c": round(float(daily["tmax"].mean()), 2),
                "origin_hot_share": round(float(daily["tmax"].ge(28).mean()), 3),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(config.PROCESSED_DIR / "origin_env.parquet")
    return df


def build_venue_altitudes() -> pd.DataFrame:
    v = config.venues().copy()
    v["altitude_m"] = [
        weather.elevation(r["lat"], r["lon"]) for _, r in v.iterrows()
    ]
    v.to_parquet(config.PROCESSED_DIR / "venues.parquet")
    return v


def build_wc_match_weather(today: str) -> pd.DataFrame:
    """Match-day weather for every WC 2022 and 2026 match."""
    m = pd.read_parquet(config.PROCESSED_DIR / "matches.parquet")
    wc = m[
        (m["tournament"] == "FIFA World Cup")
        & (m["date"] >= "2022-01-01")
    ]
    venues = pd.read_parquet(config.PROCESSED_DIR / "venues.parquet")
    vmap = {r["city"]: r for _, r in venues.iterrows()}
    rows = []
    for _, match in wc.iterrows():
        city = config.CITY_ALIASES.get(match["city"], match["city"])
        v = vmap.get(city)
        if v is None:
            continue
        day = match["date"].strftime("%Y-%m-%d")
        w = weather.match_day_weather(
            v["lat"], v["lon"], day, v["tz"], forecast=day >= today
        )
        rows.append(
            {
                "match_id": match["match_id"],
                "city": city,
                "venue_altitude_m": v["altitude_m"],
                "climate_controlled": v["climate_controlled"],
                **w,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(config.PROCESSED_DIR / "wc_match_weather.parquet")
    return df
