from __future__ import annotations

import typer

app = typer.Typer(help="FIFA World Cup 2026 prediction model")


@app.command()
def ingest(target: str = "all", today: str = "2026-07-15"):
    """Download and normalize raw data (results, squads, env, weather)."""
    from wc26.ingest import env, results, wikipedia_squads

    if target in ("all", "results"):
        results.download()
        df = results.build()
        typer.echo(f"matches: {len(df)} (max date {df['date'].max().date()})")
    if target in ("all", "squads"):
        sq = wikipedia_squads.build()
        typer.echo(f"squad players: {len(sq)}")
    if target in ("all", "env"):
        env.build_venue_altitudes()
        env.build_league_env()
        w = env.build_wc_match_weather(today=today)
        typer.echo(f"match weather rows: {len(w)}")


@app.command()
def features(asof: str = None):
    """Build the training feature matrix (optionally truncated at --asof)."""
    import pandas as pd

    from wc26.config import PROCESSED_DIR
    from wc26.features.build import FeatureBuilder

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    limit = pd.Timestamp(asof) if asof else None
    fb = FeatureBuilder(matches, as_of_limit=limit)
    df = fb.build_training_matrix()
    suffix = f"_asof_{asof}" if asof else ""
    df.to_parquet(PROCESSED_DIR / f"features{suffix}.parquet")
    typer.echo(f"feature rows: {len(df)}, cols: {len(df.columns)}")


@app.command()
def train(validate: str = "2026"):
    """Train the Poisson model; validate on the 2022 or 2026 World Cup."""
    from wc26.model.pipeline import run_training

    run_training(validate)


@app.command()
def predict(
    team_a: str,
    team_b: str,
    date: str = "2026-07-19",
    city: str = "East Rutherford",
):
    """Predict a single matchup at a 2026 venue."""
    from wc26.model.pipeline import load_predictor, venue_for

    pred = load_predictor()
    venue = venue_for(city, date)
    r = pred.predict_match(team_a, team_b, date, venue)
    typer.echo(f"{team_a} vs {team_b} — {date} at {city}")
    typer.echo(
        f"  win {r['p_win']:.1%} | draw {r['p_draw']:.1%} | loss {r['p_loss']:.1%}"
        f" | exp score {r['exp_goals_a']:.2f}-{r['exp_goals_b']:.2f}"
    )
    for i, j, p in r["top_scorelines"]:
        typer.echo(f"    {i}-{j}: {p:.1%}")


@app.command()
def simulate(n: int = 10000, as_of: str = "2026-07-15"):
    """Monte Carlo simulation of the remaining bracket."""
    import pandas as pd

    from wc26.config import PROCESSED_DIR
    from wc26.model.pipeline import load_predictor
    from wc26.sim.monte_carlo import simulate_remaining

    pred = load_predictor()
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    weather = pd.read_parquet(PROCESSED_DIR / "wc_match_weather.parquet")
    out = simulate_remaining(pred, matches, weather, as_of=as_of, n=n)
    sf2 = out["sf2"]
    typer.echo(f"SF2 {sf2['team_a']} vs {sf2['team_b']}:")
    typer.echo(
        f"  90' win {sf2['p_win']:.1%} / draw {sf2['p_draw']:.1%} / loss {sf2['p_loss']:.1%}"
        f" -> advance {sf2['p_advance']:.1%}"
    )
    typer.echo("P(champion):")
    for t, p in out["p_champion"].items():
        typer.echo(f"  {t}: {p:.1%}")
    typer.echo("P(third place):")
    for t, p in out["p_third_place"].items():
        typer.echo(f"  {t}: {p:.1%}")


if __name__ == "__main__":
    app()
