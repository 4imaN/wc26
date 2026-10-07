from __future__ import annotations

import functools
import unicodedata

import numpy as np
import pandas as pd

from wc26 import config

TOP5 = {"England", "Spain", "Italy", "Germany", "France"}

HOME_EQUIV = {
    "England": {"England", "United Kingdom"},
    "Scotland": {"Scotland", "United Kingdom"},
    "Wales": {"Wales", "United Kingdom"},
    "Curaçao": {"Curaçao", "Netherlands"},
}


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


@functools.lru_cache
def _squads() -> pd.DataFrame:
    sq = pd.read_parquet(config.PROCESSED_DIR / "squads.parquet")
    lp = config.league_profiles()
    env = pd.read_parquet(config.PROCESSED_DIR / "league_env.parquet")
    sq = sq.merge(lp, on="league_country", how="left").merge(
        env, on="league_country", how="left"
    )
    try:
        bp = pd.read_parquet(config.PROCESSED_DIR / "birthplaces.parquet")
        oe = pd.read_parquet(config.PROCESSED_DIR / "origin_env.parquet")
        bp = bp.merge(oe, on="birth_country", how="left")
        sq = sq.merge(
            bp[
                [
                    "wiki_title",
                    "birth_country",
                    "birth_altitude_m",
                    "origin_temp_c",
                    "origin_hot_share",
                    "height_cm",
                ]
            ],
            on="wiki_title",
            how="left",
        )
    except FileNotFoundError:
        pass
    sq["name_key"] = sq["player"].map(_norm_name)
    return sq


@functools.lru_cache
def _player_stats() -> pd.DataFrame | None:
    path = config.EXTERNAL_DIR / "player_stats.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["name_key"] = df["player"].map(_norm_name)
    return df


@functools.lru_cache
def _injuries() -> pd.DataFrame | None:
    path = config.EXTERNAL_DIR / "injuries.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["name_key"] = df["player"].map(_norm_name)
    return df


def _in_season(start: int, end: int, month: int) -> bool:
    if start <= end:
        return start <= month <= end
    return month >= start or month <= end


def _months_since_season_end(end: int, month: int) -> float:
    return float((month - end) % 12)


def squad_features(team: str, year: int, month: int) -> dict:
    sq = _squads()
    squad = sq[(sq["team"] == team) & (sq["year"] == year)]
    if len(squad) == 0:
        return {}

    w = (squad["caps"].fillna(0) + 1.0).to_numpy()
    w = w / w.sum()

    def wmean(col) -> float:
        vals = squad[col].to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        if not mask.any():
            return np.nan
        return float((vals[mask] * w[mask]).sum() / w[mask].sum())

    top_by_caps = squad.nlargest(5, "caps")
    out = {
        "sq_league_strength": wmean("strength"),
        "sq_top5_share": float(squad["league_country"].isin(TOP5).mean()),
        "sq_mean_caps": float(squad["caps"].mean()),
        "sq_mean_age": float(squad["age"].mean()),
        "sq_star_score": float(top_by_caps["strength"].mean()),
        "sq_accustomed_altitude": wmean("altitude_m"),
        "sq_climate_baseline": wmean("season_temp_c"),
        "sq_hot_familiarity": wmean("hot_share"),
    }

    out["sq_fw_share"] = float((squad["position"] == "FW").mean())
    out["sq_df_share"] = float((squad["position"] == "DF").mean())
    if "height_cm" in squad.columns and squad["height_cm"].notna().any():
        heights = squad["height_cm"]
        out["sq_height_mean"] = float(heights.mean())
        out["sq_tall_share"] = float((heights >= 185).mean())
        defenders = squad[squad["position"] == "DF"]["height_cm"]
        out["sq_height_def"] = (
            float(defenders.mean()) if defenders.notna().any() else np.nan
        )

    if "birth_country" in squad.columns:
        home = HOME_EQUIV.get(team, {team})
        known = squad["birth_country"].notna()
        if known.any():
            foreign = ~squad.loc[known, "birth_country"].isin(home)
            out["sq_foreign_born_share"] = float(foreign.mean())
        else:
            out["sq_foreign_born_share"] = np.nan
        out["sq_origin_altitude"] = wmean("birth_altitude_m")
        out["sq_origin_temp"] = wmean("origin_temp_c")
        out["sq_origin_hot_share"] = wmean("origin_hot_share")

    in_season = squad.apply(
        lambda r: _in_season(r["season_start_month"], r["season_end_month"], month)
        if pd.notna(r["season_start_month"])
        else np.nan,
        axis=1,
    ).to_numpy(dtype=float)
    mask = ~np.isnan(in_season)
    out["sq_in_season_share"] = (
        float((in_season[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() else np.nan
    )
    since_end = squad.apply(
        lambda r: 0.0
        if pd.isna(r["season_end_month"])
        or _in_season(r["season_start_month"], r["season_end_month"], month)
        else _months_since_season_end(int(r["season_end_month"]), month),
        axis=1,
    ).to_numpy(dtype=float)
    out["sq_months_since_season_end"] = float((since_end * w).sum())

    stats = _player_stats()
    if stats is not None:
        st = squad.merge(
            stats[stats["season_year"] == year],
            on="name_key",
            how="left",
            suffixes=("", "_st"),
        )
        matched = st["minutes"].notna()
        out["sq_stats_coverage"] = float(matched.mean())
        if matched.any():
            mw = st.loc[matched, "minutes"].to_numpy(dtype=float) + 90.0
            mw = mw / mw.sum()

            def smean(col: str) -> float:
                vals = st.loc[matched, col].to_numpy(dtype=float)
                ok = ~np.isnan(vals)
                if not ok.any():
                    return np.nan
                return float((vals[ok] * mw[ok]).sum() / mw[ok].sum())

            out["sq_minutes_load"] = float(st.loc[matched, "minutes"].mean())
            out["sq_goals90"] = smean("goals90")
            out["sq_assists90"] = smean("assists90")
            out["sq_shots90"] = smean("shots90")
            out["sq_passes90"] = smean("passes90")
            out["sq_pass_acc"] = smean("pass_acc")
            out["sq_def_actions90"] = smean("def_actions90")
            out["sq_fouls90"] = smean("fouls90")
            out["sq_cards90"] = smean("cards90")
            gk = st[matched & (st["position"] == "GK")]
            out["sq_gk_save_pct"] = (
                float(gk["save_pct"].mean()) if gk["save_pct"].notna().any() else np.nan
            )

    inj = _injuries()
    if inj is not None:
        team_inj = inj[(inj["team"] == team) & (inj["year"] == year)]
        out["sq_injured_share"] = float(
            squad["name_key"].isin(set(team_inj["name_key"])).mean()
        )
    else:
        out["sq_injured_share"] = 0.0
    return out
