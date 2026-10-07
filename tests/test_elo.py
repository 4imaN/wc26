import pandas as pd
import pytest

from wc26.config import PROCESSED_DIR
from wc26.features.build import FeatureBuilder


@pytest.fixture(scope="module")
def builder():
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    return FeatureBuilder(matches)


def test_top_teams_rated_highly(builder):
    as_of = pd.Timestamp("2026-07-15")
    ratings = {
        t: builder.elo.rating(t, as_of)
        for t in ("Spain", "Argentina", "France", "England", "Brazil", "New Zealand", "Haiti")
    }
    assert ratings["Spain"] > 1900
    assert ratings["Argentina"] > 1900
    for strong in ("Spain", "Argentina", "France", "England"):
        assert ratings[strong] > ratings["New Zealand"]
        assert ratings[strong] > ratings["Haiti"]


def test_semifinalists_beat_field(builder):
    as_of = pd.Timestamp("2026-07-15")
    semis = ["Spain", "France", "England", "Argentina"]
    field = ["Panama", "Curaçao", "Jordan", "New Zealand", "Haiti"]
    min_semi = min(builder.elo.rating(t, as_of) for t in semis)
    max_field = max(builder.elo.rating(t, as_of) for t in field)
    assert min_semi > max_field
