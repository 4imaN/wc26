from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from wc26.config import RAW_DIR

WEATHER_DIR = RAW_DIR / "weather"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_max,apparent_temperature_max,relative_humidity_2m_mean"


def _cached_get(url: str, params: dict, cache_key: str) -> dict:
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    cache = WEATHER_DIR / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data))
            time.sleep(0.4)
            return data
        except requests.RequestException as e:
            last_err = e
            time.sleep(2**attempt * 2)
    raise last_err


def elevation(lat: float, lon: float) -> float:
    data = _cached_get(
        ELEVATION_URL, {"latitude": lat, "longitude": lon}, f"elev_{lat:.4f}_{lon:.4f}"
    )
    return float(data["elevation"][0])


def match_day_weather(lat: float, lon: float, day: str, tz: str, forecast: bool = False) -> dict:
    """Daily max temperature / apparent temperature / mean humidity for a date."""
    url = FORECAST_URL if forecast else ARCHIVE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day,
        "end_date": day,
        "daily": DAILY_VARS,
        "timezone": tz,
    }
    data = _cached_get(url, params, f"day_{lat:.4f}_{lon:.4f}_{day}")
    d = data.get("daily", {})

    def first(key):
        vals = d.get(key) or [None]
        return vals[0]

    return {
        "temp_max_c": first("temperature_2m_max"),
        "apparent_temp_max_c": first("apparent_temperature_max"),
        "humidity_mean": first("relative_humidity_2m_mean"),
    }


def monthly_climate_normal(lat: float, lon: float, month: int, label: str) -> float | None:
    """Mean daily-max temperature for a month, averaged over 2015-2024."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2015-01-01",
        "end_date": "2024-12-31",
        "daily": "temperature_2m_max",
        "timezone": "UTC",
    }
    data = _cached_get(ARCHIVE_URL, params, f"normals_{label}_{lat:.2f}_{lon:.2f}")
    days = data.get("daily", {}).get("time", [])
    temps = data.get("daily", {}).get("temperature_2m_max", [])
    vals = [t for day, t in zip(days, temps) if t is not None and int(day[5:7]) == month]
    return sum(vals) / len(vals) if vals else None
