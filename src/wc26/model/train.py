from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

from wc26.config import MODELS_DIR
from wc26.features.build import feature_columns

MAX_GOALS = 8

POISSON_PARAMS = dict(
    objective="count:poisson",
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    n_jobs=-1,
)


def train_poisson(train: pd.DataFrame) -> xgb.XGBRegressor:
    cols = feature_columns(train)
    model = xgb.XGBRegressor(**POISSON_PARAMS)
    model.fit(
        train[cols],
        train["goals"],
        sample_weight=_time_decayed_weight(train),
        verbose=False,
    )
    return model


def _time_decayed_weight(df: pd.DataFrame) -> np.ndarray:
    max_date = df["date"].max()
    years_ago = (max_date - df["date"]).dt.days.to_numpy() / 365.25
    return df["weight"].to_numpy() * 0.9**years_ago


def _poisson_pmf_matrix(lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    fa = poisson.pmf(g, lam_a)
    fb = poisson.pmf(g, lam_b)
    m = np.outer(fa, fb)
    # Dixon-Coles low-score adjustment
    tau = {
        (0, 0): 1 - lam_a * lam_b * rho,
        (0, 1): 1 + lam_a * rho,
        (1, 0): 1 + lam_b * rho,
        (1, 1): 1 - rho,
    }
    for (i, j), t in tau.items():
        m[i, j] *= max(t, 1e-9)
    return m / m.sum()


def score_matrix(lam_a: float, lam_b: float, rho: float = 0.0) -> np.ndarray:
    return _poisson_pmf_matrix(max(lam_a, 0.05), max(lam_b, 0.05), rho)


def wdl_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    win = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    loss = float(np.triu(m, 1).sum())
    return win, draw, loss


def predict_lambdas(model: xgb.XGBRegressor, pair: pd.DataFrame) -> tuple[float, float]:
    """pair: two perspective rows (side 0 then side 1) with feature columns."""
    lam = model.predict(pair[model.feature_names_in_])
    return float(lam[0]), float(lam[1])


def match_probs(
    model: xgb.XGBRegressor, pair: pd.DataFrame, rho: float, temperature: float = 1.0
) -> dict:
    lam_a, lam_b = predict_lambdas(model, pair)
    m = score_matrix(lam_a, lam_b, rho)
    w, d, l = wdl_from_matrix(m)
    probs = np.array([w, d, l])
    if temperature != 1.0:
        logits = np.log(np.clip(probs, 1e-9, 1)) / temperature
        probs = np.exp(logits) / np.exp(logits).sum()
    return {
        "p_win": float(probs[0]),
        "p_draw": float(probs[1]),
        "p_loss": float(probs[2]),
        "lambda_a": lam_a,
        "lambda_b": lam_b,
        "matrix": m,
    }


def fit_rho(model: xgb.XGBRegressor, val: pd.DataFrame) -> float:
    """Grid-search Dixon-Coles rho minimizing W/D/L log loss on validation pairs."""
    best_rho, best_ll = 0.0, np.inf
    for rho in np.arange(-0.15, 0.16, 0.03):
        ll = _pairs_log_loss(model, val, rho, 1.0)
        if ll < best_ll:
            best_rho, best_ll = float(rho), ll
    return best_rho


def fit_temperature(model: xgb.XGBRegressor, val: pd.DataFrame, rho: float) -> float:
    best_t, best_ll = 1.0, np.inf
    for t in np.arange(0.6, 1.65, 0.05):
        ll = _pairs_log_loss(model, val, rho, float(t))
        if ll < best_ll:
            best_t, best_ll = float(t), ll
    return best_t


def _pairs_log_loss(model, val: pd.DataFrame, rho: float, temperature: float) -> float:
    lls = []
    for _, pair in val.groupby("match_id"):
        pair = pair.sort_values("side")
        if len(pair) != 2:
            continue
        r = match_probs(model, pair, rho, temperature)
        gf, ga = pair.iloc[0]["goals"], pair.iloc[0]["goals_against"]
        p = r["p_win"] if gf > ga else (r["p_draw"] if gf == ga else r["p_loss"])
        lls.append(-np.log(max(p, 1e-9)))
    return float(np.mean(lls))


def elo_baseline_log_loss(val: pd.DataFrame, matrix_draw: float = 0.24) -> float:
    """Elo-only baseline: win prob from elo_diff, fixed draw share."""
    lls = []
    for _, pair in val.groupby("match_id"):
        pair = pair.sort_values("side")
        if len(pair) != 2:
            continue
        e = 1.0 / (1.0 + 10 ** (-pair.iloc[0]["elo_diff"] / 400))
        p_draw = matrix_draw
        p_win = e * (1 - p_draw)
        p_loss = (1 - e) * (1 - p_draw)
        gf, ga = pair.iloc[0]["goals"], pair.iloc[0]["goals_against"]
        p = p_win if gf > ga else (p_draw if gf == ga else p_loss)
        lls.append(-np.log(max(p, 1e-9)))
    return float(np.mean(lls))


def save(model: xgb.XGBRegressor, rho: float, temperature: float, tag: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    model.save_model(MODELS_DIR / f"poisson_{tag}.json")
    (MODELS_DIR / f"params_{tag}.json").write_text(
        json.dumps({"rho": rho, "temperature": temperature})
    )


def load(tag: str) -> tuple[xgb.XGBRegressor, float, float]:
    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / f"poisson_{tag}.json")
    params = json.loads((MODELS_DIR / f"params_{tag}.json").read_text())
    return model, params["rho"], params["temperature"]
