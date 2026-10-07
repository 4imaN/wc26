from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

CITY_ALIASES = {"Dallas": "Arlington"}


@functools.lru_cache
def venues() -> pd.DataFrame:
    frames = [
        pd.read_csv(CONFIG_DIR / "venues_2026.csv"),
        pd.read_csv(CONFIG_DIR / "venues_2022.csv"),
    ]
    return pd.concat(frames, ignore_index=True)


@functools.lru_cache
def league_profiles() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "league_profiles.csv")


@functools.lru_cache
def country_profiles() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "country_profiles.csv")


@functools.lru_cache
def _team_name_map() -> dict[str, str]:
    df = pd.read_csv(CONFIG_DIR / "team_name_map.csv")
    return dict(zip(df["alias"], df["canonical"]))


def canonicalize_team_name(name: str) -> str:
    name = str(name).strip()
    return _team_name_map().get(name, name)


def venue_for_city(city: str) -> pd.Series | None:
    city = CITY_ALIASES.get(city, city)
    v = venues()
    match = v[v["city"] == city]
    return match.iloc[0] if len(match) else None
