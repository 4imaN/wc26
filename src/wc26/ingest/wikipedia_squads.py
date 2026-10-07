from __future__ import annotations

import re
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

from wc26.config import PROCESSED_DIR, RAW_DIR, canonicalize_team_name

POSITIONS = {"GK", "DF", "MF", "FW"}

FLAG_COUNTRY_FIXES = {
    "People's Republic of China": "China PR",
    "the People's Republic of China": "China PR",
    "Republic of Ireland": "Ireland",
}


def _clean(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text).strip()


def _flag_country(cell) -> str:
    img = cell.find("img")
    if img is None:
        return ""
    src = img.get("resource") or img.get("src") or ""
    m = re.search(r"Flag_of_(?:the_)?([^./]+?)(?:_\(.*?\))?\.svg", src)
    if not m:
        return ""
    from urllib.parse import unquote

    country = unquote(m.group(1)).replace("_", " ").strip()
    return FLAG_COUNTRY_FIXES.get(country, country)


def parse_squads(year: int) -> pd.DataFrame:
    html = (RAW_DIR / "wikipedia" / f"squads_{year}.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    rows = []
    for heading in soup.select("div.mw-heading3 > h3, h3"):
        team_raw = _clean(heading.get_text())
        node = heading.parent if "mw-heading" in (heading.parent.get("class") or []) else heading
        table = node.find_next_sibling()
        hops = 0
        while table is not None and hops < 6 and not (
            table.name == "table" and "wikitable" in (table.get("class") or [])
        ):
            table = table.find_next_sibling()
            hops += 1
        if table is None or table.name != "table":
            continue

        header = [_clean(th.get_text()) for th in table.find("tr").find_all("th")]
        if "Player" not in " ".join(header) or "Club" not in " ".join(header):
            continue

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 7:
                continue
            texts = [_clean(c.get_text(" ")) for c in cells]
            pos = texts[1].replace("1", "").replace("2", "").replace("3", "").replace("4", "").strip()
            if pos not in POSITIONS:
                continue
            league_country = _flag_country(cells[-1])
            player_link = cells[2].find("a")
            wiki_title = (
                player_link["title"]
                if player_link is not None and player_link.get("title")
                else None
            )
            bday = cells[3].find("span", class_="bday")
            dob = bday.get_text().strip() if bday is not None else None
            if dob is None:
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", texts[3])
                dob = m.group(0) if m else None
            caps = pd.to_numeric(texts[4], errors="coerce")
            goals = pd.to_numeric(texts[5], errors="coerce")
            rows.append(
                {
                    "year": year,
                    "team": canonicalize_team_name(team_raw),
                    "position": pos,
                    "player": texts[2],
                    "wiki_title": wiki_title,
                    "dob": dob,
                    "caps": caps,
                    "intl_goals": goals,
                    "club": texts[-1],
                    "league_country": canonicalize_team_name(league_country),
                }
            )

    df = pd.DataFrame(rows)
    if len(df):
        ref = date(year, 6, 11 if year == 2026 else 20)
        dob = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = (pd.Timestamp(ref) - dob).dt.days / 365.25
    return df


def build() -> pd.DataFrame:
    frames = [parse_squads(y) for y in (2022, 2026)]
    df = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "squads.parquet")
    return df
