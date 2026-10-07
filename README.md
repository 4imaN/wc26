# WC26 — FIFA World Cup 2026 prediction model

Predicts match outcomes (win/draw/loss probabilities + expected score) and Monte-Carlo-simulates
the World Cup bracket. Trained on all international matches since 2010 (weighted by tournament
importance and recency), validated on the 2022 World Cup and the 2026 matches played so far.

## Model

Paired XGBoost Poisson regressors predict each side's expected goals (λ) from engineered
features; a Dixon-Coles-adjusted score matrix converts (λ_A, λ_B) into W/D/L probabilities and
scoreline distributions, with temperature calibration fit on held-out World Cup matches.

Feature families (one row per match side, strict as-of-date semantics — no future data):

- **Elo**: full replay of internationals since 1946 (K by tournament, goal-margin multiplier,
  host advantage), plus 90-day momentum.
- **Form**: time-decayed points/goals/clean sheets over last 5/10 matches, opposition-adjusted
  (actual minus Elo-expected points), rest days, tournament fatigue (matches played / rest).
- **Head-to-head outside World Cups** (Nations League, qualifiers, continental cups,
  friendlies): time-decayed win rate and goal diff, competitive vs friendly split.
- **Squad quality** (Wikipedia squads): caps-weighted club-league strength tier, top-5-league
  share, caps/age, star score.
- **Player performance layer** (optional `data/external/player_stats.csv` — FBref blocks
  automated access, so this is a manual input): minutes (season load → exhaustion), goals/
  assists/shots/passes per 90, pass accuracy, defensive actions, fouls/cards, GK saves.
  The model degrades gracefully when absent. Same for `data/external/injuries.csv`.
- **Venue context**: altitude and altitude-vs-accustomed delta (squad-weighted club league
  altitudes), match-day apparent temperature and heat delta vs squad climate baseline
  (Open-Meteo), climate-controlled stadiums zeroed out.
- **Seasonality**: share of squad in-season in the match month, months since season end (rust),
  hot-weather familiarity.
- **Motivation**: host, defending champion, titles won, years since last title (drought
  pressure × team strength), first-time qualifier, knockout stage depth, revenge flag.

## Usage

```bash
source .venv/bin/activate
wc26 ingest            # download results / squads / weather (cached)
wc26 features          # build training matrix (~4 min)
wc26 train --validate 2022   # backtest on 2022 WC
wc26 train --validate 2026   # validate on played 2026 matches + save prod model
wc26 predict Spain England --date 2026-07-19 --city "East Rutherford"
wc26 simulate --n 10000      # remaining-bracket championship odds
pytest                 # leakage + Elo sanity tests
```

`notebooks/explore.ipynb` has Elo trajectories, feature importance, calibration, and the
simulated bracket.

## Data sources

- Match results & fixtures: github.com/martj42/international_results (updated daily, includes
  2026 WC through the semifinals)
- Squads: Wikipedia 2022/2026 FIFA World Cup squads pages
- Weather / elevation / climate normals: Open-Meteo (cached in `data/raw/weather/`)
- Static configs (hand-curated): `config/*.csv` — venues with coordinates and roof flags,
  league strength tiers and calendars, country titles/hosts/first appearances, name mapping
