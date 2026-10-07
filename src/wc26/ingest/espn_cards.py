from __future__ import annotations

import json
import time

import pandas as pd
import requests

from wc26 import config

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
CACHE = config.RAW_DIR / "espn"
WC_WINDOWS = [("2022-11-20", "2022-12-18"), ("2026-06-11", "2026-07-19")]


def _get(path: str, params: dict, cache_key: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data))
            time.sleep(0.25)
            return data
        except requests.RequestException:
            time.sleep(2**attempt)
    raise RuntimeError(f"espn request failed: {cache_key}")


def build() -> pd.DataFrame:
    rows = []
    for lo, hi in WC_WINDOWS:
        for day in pd.date_range(lo, hi):
            ymd = day.strftime("%Y%m%d")
            sb = _get("scoreboard", {"dates": ymd}, f"sb_{ymd}")
            for event in sb.get("events", []):
                rows.extend(_parse_event(event["id"], day))
    df = pd.DataFrame(rows)
    df.to_parquet(config.PROCESSED_DIR / "referee_cards.parquet")
    return df


def _parse_event(event_id: str, day: pd.Timestamp) -> list[dict]:
    s = _get("summary", {"event": event_id}, f"sum_{event_id}")
    comps = s.get("header", {}).get("competitions", [])
    if not comps:
        return []
    comp = comps[0]
    teams = {}
    for c in comp.get("competitors", []):
        name = config.canonicalize_team_name(c.get("team", {}).get("displayName", ""))
        teams[c.get("homeAway")] = name
    if "home" not in teams or "away" not in teams:
        return []

    offs = s.get("gameInfo", {}).get("officials", [])
    referee = next(
        (o.get("fullName") for o in offs if o.get("position", {}).get("name") == "Referee"),
        offs[0].get("fullName") if offs else None,
    )

    STATS = {
        "foulsCommitted": "fouls",
        "yellowCards": "yellows",
        "redCards": "reds",
        "offsides": "offsides",
        "possessionPct": "possession",
        "totalShots": "shots",
        "shotsOnTarget": "shots_on_target",
        "passPct": "pass_pct",
        "totalTackles": "tackles",
        "interceptions": "interceptions",
        "saves": "saves",
    }
    stats = {"home": {}, "away": {}}
    for t in s.get("boxscore", {}).get("teams", []):
        name = config.canonicalize_team_name(t.get("team", {}).get("displayName", ""))
        side = next((k for k, v in teams.items() if v == name), None)
        if side is None:
            continue
        for st in t.get("statistics", []):
            key = STATS.get(st.get("name"))
            if key:
                stats[side][key] = pd.to_numeric(st.get("displayValue"), errors="coerce")

    row = {
        "date": day,
        "home_team": teams["home"],
        "away_team": teams["away"],
        "referee": referee,
    }
    for side in ("home", "away"):
        for key in STATS.values():
            row[f"{side}_{key}"] = stats[side].get(key)
    return [row]
