"""Tests for churn.features: feature lists, new features and the X/y split.

Run from the project root: python -m pytest -v
"""

import pandas as pd
import pytest

from churn.cleaning import clean_data
from churn.config import RAW_DATA_PATH, TARGET_COLUMN
from churn.features import (
    CHANGED_PHONE_COLUMN,
    CITY_CODE_COLUMN,
    EXCLUDED_COLUMNS,
    LEAKAGE_RISK_COLUMNS,
    LIFESTYLE_COLUMNS,
    LIFESTYLE_GROUP,
    MODEL_FEATURES,
    PERSONAL_COLUMNS,
    PERSONAL_GROUP,
    add_features,
    extra_feature_columns,
    split_features_target,
)

DAYS_PER_YEAR = 365


# --- The feature lists ---


def test_no_column_is_both_feature_and_excluded():
    """A column cannot be used AND excluded at the same time."""
    assert set(MODEL_FEATURES) & set(EXCLUDED_COLUMNS) == set()


def test_feature_list_has_no_duplicates():
    """Each feature appears only once."""
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


# parametrize runs this once per excluded group.
@pytest.mark.parametrize(
    "excluded_group", [LEAKAGE_RISK_COLUMNS, PERSONAL_COLUMNS, LIFESTYLE_COLUMNS]
)
def test_excluded_groups_are_not_features(excluded_group: tuple[str, ...]):
    """Leakage-risk, personal and lifestyle columns never reach the model."""
    assert set(excluded_group) & set(MODEL_FEATURES) == set()


# Reads ONLY the header row of the real file (column names, no customer data).
# Skipped if the data file is not on this computer.
@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw data file not available")
def test_every_real_column_is_classified():
    """Every column after cleaning + features is either a feature or excluded."""
    # nrows=0 = read the column names only, no rows.
    empty_raw = pd.read_csv(RAW_DATA_PATH, nrows=0)
    columns = set(add_features(clean_data(empty_raw)).columns)
    classified = set(MODEL_FEATURES) | set(EXCLUDED_COLUMNS)

    assert columns - classified == set(), "columns with no decision"
    assert classified - columns == set(), "decisions for columns that do not exist"


# --- add_features ---


# Fake cleaned data: 3 customers, built to test both new features.
#   Row 0: phone much newer than the account  -> changed phone (1)
#   Row 1: phone as old as the account        -> same phone (0)
#   Row 2: phone age unknown, area unknown    -> both missing
@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    """Three fake cleaned customers."""
    return pd.DataFrame(
        {
            "ServiceArea": ["NYCBRO917", "HOUHOU281", None],
            "CurrentEquipmentDays": [100.0, 2 * DAYS_PER_YEAR, None],
            "MonthsInService": [24, 24, 24],
        }
    )


def test_city_code_is_first_three_letters(cleaned_df: pd.DataFrame):
    """'NYCBRO917' -> 'NYC'; a missing area gives a missing city code."""
    city_codes = add_features(cleaned_df)[CITY_CODE_COLUMN]

    assert city_codes[0] == "NYC"
    assert city_codes[1] == "HOU"
    assert pd.isna(city_codes[2])


def test_changed_phone_flag(cleaned_df: pd.DataFrame):
    """New phone -> 1, original phone -> 0, unknown phone age -> missing."""
    changed_phone = add_features(cleaned_df)[CHANGED_PHONE_COLUMN]

    assert changed_phone[0] == 1
    assert changed_phone[1] == 0
    assert pd.isna(changed_phone[2])


def test_add_features_does_not_change_the_input(cleaned_df: pd.DataFrame):
    """The original data is left untouched (works on a copy)."""
    original = cleaned_df.copy()

    add_features(cleaned_df)

    pd.testing.assert_frame_equal(cleaned_df, original)


def test_add_features_missing_column_raises(cleaned_df: pd.DataFrame):
    """A missing input column gives a clear error."""
    with pytest.raises(ValueError, match="missing columns"):
        add_features(cleaned_df.drop(columns="ServiceArea"))


# --- split_features_target ---


# Fake data with every model feature (all 0) plus Churn for 2 customers.
@pytest.fixture
def model_ready_df() -> pd.DataFrame:
    """Two fake customers with all model features and a Churn label."""
    df = pd.DataFrame(0, index=[0, 1], columns=list(MODEL_FEATURES))
    df[TARGET_COLUMN] = ["Yes", "No"]
    df["CustomerID"] = [1, 2]
    return df


def test_split_returns_model_features_in_fixed_order(model_ready_df: pd.DataFrame):
    """X has exactly the model features, in the same order every time."""
    features, _ = split_features_target(model_ready_df)

    assert list(features.columns) == list(MODEL_FEATURES)


def test_split_turns_churn_into_one_and_zero(model_ready_df: pd.DataFrame):
    """'Yes' -> 1 and 'No' -> 0."""
    _, target = split_features_target(model_ready_df)

    assert list(target) == [1, 0]


def test_split_leaves_out_id_and_target(model_ready_df: pd.DataFrame):
    """The customer ID and the answer are never model inputs."""
    features, _ = split_features_target(model_ready_df)

    assert "CustomerID" not in features
    assert TARGET_COLUMN not in features


def test_split_missing_feature_raises(model_ready_df: pd.DataFrame):
    """A missing model feature gives a clear error."""
    with pytest.raises(ValueError, match="missing columns"):
        split_features_target(model_ready_df.drop(columns=CITY_CODE_COLUMN))


# --- Optional extra feature groups (A4 comparison) ---


def test_personal_group_covers_exactly_the_personal_columns():
    """The personal group = the 6 personal columns, none lost or added."""
    group = extra_feature_columns([PERSONAL_GROUP])
    assert set(group.numeric + group.categorical) == set(PERSONAL_COLUMNS)


def test_lifestyle_group_covers_exactly_the_lifestyle_columns():
    """The lifestyle group = the 10 lifestyle columns, all categorical."""
    group = extra_feature_columns([LIFESTYLE_GROUP])
    assert group.numeric == ()
    assert set(group.categorical) == set(LIFESTYLE_COLUMNS)


def test_no_groups_means_no_extra_columns():
    """The default (no groups) adds nothing: the main model is unchanged."""
    assert extra_feature_columns([]) == ((), ())


def test_unknown_group_raises():
    """A typo in a group name fails loudly instead of being ignored."""
    with pytest.raises(ValueError, match="Unknown feature groups"):
        extra_feature_columns(["persnal"])
