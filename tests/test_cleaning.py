"""Tests for churn.cleaning: the rule-based cleaning steps.

Run from the project root: python -m pytest -v
"""

import pandas as pd
import pytest

from churn.cleaning import (
    CELLPHONE_STATUS_COLUMN,
    NEW_USER_COLUMN,
    NOT_NEW_USER_COLUMN,
    STATUS_NEW,
    STATUS_NOT_NEW,
    STATUS_UNKNOWN,
    clean_data,
)


# Fake raw data: 3 customers, built to hit every cleaning rule.
#   Row 0: normal values
#   Row 1: negative values, "Unknown" price, zero age/income
#   Row 2: missing values that must stay missing
@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Three fake customers covering every cleaning rule."""
    return pd.DataFrame(
        {
            "MonthlyRevenue": [50.0, -6.2, None],
            "TotalRecurringCharge": [45.0, -11.0, 30.0],
            "CurrentEquipmentDays": [300.0, -5.0, 100.0],
            "HandsetPrice": ["30", "Unknown", "150"],
            NEW_USER_COLUMN: ["Yes", "No", "No"],
            NOT_NEW_USER_COLUMN: ["No", "Yes", "No"],
            "AgeHH1": [40.0, 0.0, None],
            "AgeHH2": [0.0, 0.0, 35.0],
            "IncomeGroup": [5, 0, 7],
        }
    )


# --- General behaviour ---


def test_clean_data_does_not_change_the_input(raw_df: pd.DataFrame):
    """The original data is left untouched (cleaning works on a copy)."""
    original = raw_df.copy()

    clean_data(raw_df)

    # assert_frame_equal fails if ANY value, column or type differs.
    pd.testing.assert_frame_equal(raw_df, original)


def test_clean_data_keeps_every_row(raw_df: pd.DataFrame):
    """No customer is dropped by cleaning."""
    assert len(clean_data(raw_df)) == len(raw_df)


def test_clean_data_missing_column_raises(raw_df: pd.DataFrame):
    """A missing column gives a clear error instead of a confusing crash."""
    with pytest.raises(ValueError, match="missing columns"):
        clean_data(raw_df.drop(columns="HandsetPrice"))


# --- Rule 1: negatives become missing ---


@pytest.mark.parametrize(
    "column", ["MonthlyRevenue", "TotalRecurringCharge", "CurrentEquipmentDays"]
)
def test_negative_values_become_missing(raw_df: pd.DataFrame, column: str):
    """Negative money or days (row 1) become missing; positives stay the same."""
    cleaned = clean_data(raw_df)

    assert pd.isna(cleaned.loc[1, column])
    assert cleaned.loc[0, column] == raw_df.loc[0, column]


def test_existing_missing_values_stay_missing(raw_df: pd.DataFrame):
    """A value that was already missing (row 2) stays missing."""
    assert pd.isna(clean_data(raw_df).loc[2, "MonthlyRevenue"])


# --- Rule 2: HandsetPrice becomes a number ---


def test_handset_price_becomes_number(raw_df: pd.DataFrame):
    """'30' and '150' become numbers; 'Unknown' becomes missing."""
    prices = clean_data(raw_df)["HandsetPrice"]

    assert pd.api.types.is_numeric_dtype(prices)
    assert prices[0] == 30
    assert pd.isna(prices[1])
    assert prices[2] == 150


def test_handset_price_bad_text_raises(raw_df: pd.DataFrame):
    """A price that is neither a number nor 'Unknown' fails loudly."""
    raw_df.loc[0, "HandsetPrice"] = "cheap"

    with pytest.raises(ValueError, match="must be a number or 'Unknown'"):
        clean_data(raw_df)


# --- Rule 3: two Yes/No columns become one 3-way column ---


def test_cellphone_columns_are_combined(raw_df: pd.DataFrame):
    """New -> 'New', NotNew -> 'NotNew', both No -> 'Unknown'."""
    status = clean_data(raw_df)[CELLPHONE_STATUS_COLUMN]

    assert list(status) == [STATUS_NEW, STATUS_NOT_NEW, STATUS_UNKNOWN]


def test_original_cellphone_columns_are_dropped(raw_df: pd.DataFrame):
    """The two old columns are removed after combining."""
    cleaned = clean_data(raw_df)

    assert NEW_USER_COLUMN not in cleaned
    assert NOT_NEW_USER_COLUMN not in cleaned


def test_new_and_not_new_at_once_raises(raw_df: pd.DataFrame):
    """A customer marked as BOTH new and not new is a contradiction: fail loudly."""
    raw_df.loc[0, NOT_NEW_USER_COLUMN] = "Yes"

    with pytest.raises(ValueError, match="both a new and a not-new"):
        clean_data(raw_df)


# --- Rule 4: zero means unknown for AgeHH1 and IncomeGroup ---


@pytest.mark.parametrize("column", ["AgeHH1", "IncomeGroup"])
def test_zero_becomes_missing(raw_df: pd.DataFrame, column: str):
    """0 (row 1) becomes missing; real values stay the same."""
    cleaned = clean_data(raw_df)

    assert pd.isna(cleaned.loc[1, column])
    assert cleaned.loc[0, column] == raw_df.loc[0, column]


def test_age_hh2_zero_is_left_alone(raw_df: pd.DataFrame):
    """AgeHH2 = 0 may mean 'no second person', so it is NOT changed."""
    assert clean_data(raw_df).loc[0, "AgeHH2"] == 0
