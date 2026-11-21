"""Tests for churn.data: loading and structure checks.

Run from the project root: python -m pytest -v
"""

from pathlib import Path

import pandas as pd
import pytest

from churn.config import ID_COLUMN, TARGET_COLUMN
from churn.data import load_dataset, validate_dataset


# A fixture is reusable test data. Each test that asks for `valid_df`
# gets a fresh copy, so one test breaking it cannot affect another.
@pytest.fixture
def valid_df() -> pd.DataFrame:
    """A tiny, valid dataset: 4 fake customers, 2 churned."""
    return pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            TARGET_COLUMN: ["Yes", "No", "No", "Yes"],
            "MonthlyRevenue": [24.0, 16.99, 38.0, 82.28],
        }
    )


# `tmp_path` is a temporary folder from pytest, deleted after the test.
# We never touch the real data/ folder in tests.
def test_load_dataset_reads_valid_file(tmp_path: Path, valid_df: pd.DataFrame):
    """A good CSV file loads with the same rows and columns."""
    csv_path = tmp_path / "customers.csv"
    valid_df.to_csv(csv_path, index=False)

    loaded_df = load_dataset(csv_path)

    assert loaded_df.shape == valid_df.shape
    assert list(loaded_df.columns) == list(valid_df.columns)


# `pytest.raises` means: this code MUST fail with this error and message.
def test_load_dataset_missing_file_raises(tmp_path: Path):
    """A missing file gives a clear 'not found' error."""
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_validate_accepts_valid_data(valid_df: pd.DataFrame):
    """Good data passes all checks without an error."""
    validate_dataset(valid_df)


# `parametrize` runs the same test once for each value in the list.
@pytest.mark.parametrize("column", [ID_COLUMN, TARGET_COLUMN])
def test_validate_missing_required_column_raises(valid_df: pd.DataFrame, column: str):
    """Removing CustomerID or Churn is caught."""
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(valid_df.drop(columns=column))


def test_validate_repeated_ids_raises(valid_df: pd.DataFrame):
    """Two rows with the same CustomerID are caught."""
    valid_df.loc[1, ID_COLUMN] = valid_df.loc[0, ID_COLUMN]

    with pytest.raises(ValueError, match="repeated"):
        validate_dataset(valid_df)


def test_validate_missing_target_raises(valid_df: pd.DataFrame):
    """A customer with a blank Churn value is caught."""
    valid_df.loc[0, TARGET_COLUMN] = None

    with pytest.raises(ValueError, match="no Churn value"):
        validate_dataset(valid_df)


@pytest.mark.parametrize("bad_label", ["yes", "Y", "1"])
def test_validate_unexpected_target_raises(valid_df: pd.DataFrame, bad_label: str):
    """Churn values other than exactly 'Yes' or 'No' are caught."""
    valid_df.loc[0, TARGET_COLUMN] = bad_label

    with pytest.raises(ValueError, match="Unexpected Churn values"):
        validate_dataset(valid_df)
