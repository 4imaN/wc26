import pandas as pd
import pytest

from wc26.config import PROCESSED_DIR
from wc26.features.build import FeatureBuilder


@pytest.fixture(scope="module")
def matches():
    return pd.read_parquet(PROCESSED_DIR / "matches.parquet")


def test_truncated_features_match_full_history(matches):
    """Features for a 2022 WC match must be identical whether built from full
    data or from data truncated at the match date (i.e. no future leakage)."""
    target_date = pd.Timestamp("2022-12-18")  # 2022 final
    fb_full = FeatureBuilder(matches)
    fb_trunc = FeatureBuilder(matches, as_of_limit=target_date)

    f_full = fb_full.side_features(
        "Argentina", "France", target_date, False, 2022, None
    )
    f_trunc = fb_trunc.side_features(
        "Argentina", "France", target_date, False, 2022, None
    )
    assert set(f_full) == set(f_trunc)
    for k in f_full:
        a, b = f_full[k], f_trunc[k]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, abs=1e-9), f"leak in feature {k}: {a} != {b}"


def test_elo_lookup_excludes_same_day_result(matches):
    fb = FeatureBuilder(matches)
    semi_date = pd.Timestamp("2022-12-13")  # Argentina 3-0 Croatia
    r_before = fb.elo.rating("Argentina", semi_date)
    r_after = fb.elo.rating("Argentina", semi_date + pd.Timedelta(days=1))
    assert r_after > r_before  # the win must only show up afterwards
