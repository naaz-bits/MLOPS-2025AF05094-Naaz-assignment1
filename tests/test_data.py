import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS, FEATURES, TARGET, clean


@pytest.fixture
def raw_df():
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 0, 0, 2, 129, 1, 2.6, 2, np.nan, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, np.nan, 4],
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],  # duplicate
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].tolist() == [0, 1, 1, 1]


def test_clean_drops_duplicates(raw_df):
    assert len(clean(raw_df)) == 4


def test_clean_nulls_impossible_cholesterol(raw_df):
    out = clean(raw_df)
    assert out["chol"].isna().sum() == 1
    assert (out["chol"].dropna() > 0).all()


def test_clean_keeps_all_features(raw_df):
    out = clean(raw_df)
    assert set(FEATURES) | {TARGET} == set(out.columns)


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
