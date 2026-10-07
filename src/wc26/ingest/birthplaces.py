from __future__ import annotations

import json
import time

import pandas as pd
import requests

from wc26 import config

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "wc26-model/0.1 (research; contact: local)"}
CACHE = config.RAW_DIR / "wikidata"

BIRTH_COUNTRY_ALIASES = {
    "United Kingdom": "United Kingdom",
    "Kingdom of the Netherlands": "Netherlands",
    "Czechoslovakia": "Czech Republic",
    "Socialist Federal Republic of Yugoslavia": "Serbia",
    "Federal Republic of Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia",
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "Zaire": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "People's Republic of China": "China PR",
}


def _api(params: dict, cache_key: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    last = None
    for attempt in range(6):
        try:
            r = requests.get(
                WIKIDATA_API,
                params={**params, "format": "json"},
                headers=UA,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data))
            time.sleep(1.5)
            return data
        except requests.RequestException as e:
            last = e
            time.sleep(min(3 * 2**attempt, 60))
    raise RuntimeError(f"wikidata request failed: {cache_key}: {last!r}")


def _chunks(items: list, n: int = 50):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _first_claim(entity: dict, prop: str):
    claims = entity.get("claims", {}).get(prop, [])
    for c in claims:
        val = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if val is not None:
            return val
    return None


def build() -> pd.DataFrame:
    sq = pd.read_parquet(config.PROCESSED_DIR / "squads.parquet")
    titles = sorted(set(sq["wiki_title"].dropna()))

    # pass 1: player article -> P19 birthplace entity id + P2048 height
    birth_city_of: dict[str, str] = {}
    height_of: dict[str, float] = {}
    for i, chunk in enumerate(_chunks(titles)):
        data = _api(
            {
                "action": "wbgetentities",
                "sites": "enwiki",
                "titles": "|".join(chunk),
                "props": "claims|sitelinks",
                "sitefilter": "enwiki",
            },
            f"players_{i:03d}",
        )
        for ent in data.get("entities", {}).values():
            title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")
            if not title:
                continue
            val = _first_claim(ent, "P19")
            if isinstance(val, dict) and "id" in val:
                birth_city_of[title] = val["id"]
            h = _first_claim(ent, "P2048")
            if isinstance(h, dict) and "amount" in h:
                amount = abs(float(h["amount"]))
                unit = h.get("unit", "")
                if unit.endswith("Q11573"):  # metre
                    height_of[title] = amount * 100
                elif unit.endswith("Q174728"):  # centimetre
                    height_of[title] = amount

    # pass 2: birth city entity -> P17 country id + P625 coordinates
    city_ids = sorted(set(birth_city_of.values()))
    city_info: dict[str, dict] = {}
    for i, chunk in enumerate(_chunks(city_ids)):
        data = _api(
            {"action": "wbgetentities", "ids": "|".join(chunk), "props": "claims|labels"},
            f"cities_{i:03d}",
        )
        for qid, ent in data.get("entities", {}).items():
            country = _first_claim(ent, "P17")
            coords = _first_claim(ent, "P625")
            city_info[qid] = {
                "birth_city": ent.get("labels", {}).get("en", {}).get("value"),
                "country_qid": country["id"] if isinstance(country, dict) else None,
                "birth_lat": coords.get("latitude") if isinstance(coords, dict) else None,
                "birth_lon": coords.get("longitude") if isinstance(coords, dict) else None,
            }

    # pass 3: country entity -> english label
    country_ids = sorted({c["country_qid"] for c in city_info.values() if c["country_qid"]})
    country_label: dict[str, str] = {}
    for i, chunk in enumerate(_chunks(country_ids)):
        data = _api(
            {"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels"},
            f"countries_{i:03d}",
        )
        for qid, ent in data.get("entities", {}).items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            if label:
                country_label[qid] = label

    rows = []
    for title in sorted(set(birth_city_of) | set(height_of)):
        city_qid = birth_city_of.get(title)
        info = city_info.get(city_qid, {}) if city_qid else {}
        raw_country = country_label.get(info.get("country_qid"))
        country = None
        if raw_country:
            country = BIRTH_COUNTRY_ALIASES.get(raw_country, raw_country)
            country = config.canonicalize_team_name(country)
        rows.append(
            {
                "wiki_title": title,
                "birth_city": info.get("birth_city"),
                "birth_country": country,
                "birth_lat": info.get("birth_lat"),
                "birth_lon": info.get("birth_lon"),
                "height_cm": height_of.get(title),
            }
        )
    df = pd.DataFrame(rows)
    df = _add_elevations(df)
    df.to_parquet(config.PROCESSED_DIR / "birthplaces.parquet")
    return df


def _add_elevations(df: pd.DataFrame) -> pd.DataFrame:
    from wc26.ingest.weather import _cached_get, ELEVATION_URL

    coords = df.dropna(subset=["birth_lat", "birth_lon"])[["birth_lat", "birth_lon"]]
    coords = coords.round(3).drop_duplicates().reset_index(drop=True)
    elev: dict[tuple, float] = {}
    batch = 80
    for i in range(0, len(coords), batch):
        part = coords.iloc[i : i + batch]
        data = _cached_get(
            ELEVATION_URL,
            {
                "latitude": ",".join(str(v) for v in part["birth_lat"]),
                "longitude": ",".join(str(v) for v in part["birth_lon"]),
            },
            f"belev_{i:04d}_{len(part)}",
        )
        for (lat, lon), e in zip(part.itertuples(index=False), data["elevation"]):
            elev[(lat, lon)] = float(e)
    df["birth_altitude_m"] = [
        elev.get((round(la, 3), round(lo, 3)))
        if pd.notna(la) and pd.notna(lo)
        else None
        for la, lo in zip(df["birth_lat"], df["birth_lon"])
    ]
    return df
