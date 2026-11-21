"""Tests for churn.split: the train/test split and saving it.

Run from the project root: python -m pytest -v
"""

from pathlib import Path

import pandas as pd
import pytest

from churn.config import ID_COLUMN, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET_COLUMN
from churn.split import churn_rate, save_splits, split_train_test

# Fake data size: 100 customers, 30 churned (30% churn rate).
N_CUSTOMERS = 100
N_CHURNED = 30
EXPECTED_CHURN_RATE = N_CHURNED / N_CUSTOMERS
TEST_SHARE = 0.2


# Reusable fake dataset. Each test gets a fresh copy.
@pytest.fixture
def customers_df() -> pd.DataFrame:
    """100 fake customers: the first 30 churned, the other 70 stayed."""
    churn_labels = [POSITIVE_LABEL] * N_CHURNED + [NEGATIVE_LABEL] * (
        N_CUSTOMERS - N_CHURNED
    )
    return pd.DataFrame(
        {
            ID_COLUMN: range(1, N_CUSTOMERS + 1),
            TARGET_COLUMN: churn_labels,
            "MonthsInService": range(N_CUSTOMERS),
        }
    )


# --- split_train_test ---


def test_split_has_no_overlap(customers_df: pd.DataFrame):
    """No customer appears in both train and test (that would be leakage)."""
    train_df, test_df = split_train_test(customers_df, test_size=TEST_SHARE)

    # Sets: "&" gives the IDs found in BOTH. It must be empty.
    shared_ids = set(train_df[ID_COLUMN]) & set(test_df[ID_COLUMN])
    assert shared_ids == set()


def test_split_keeps_every_customer(customers_df: pd.DataFrame):
    """Train + test together contain every customer, none lost."""
    train_df, test_df = split_train_test(customers_df, test_size=TEST_SHARE)

    assert len(train_df) + len(test_df) == N_CUSTOMERS
    assert len(test_df) == N_CUSTOMERS * TEST_SHARE


def test_split_keeps_churn_rate_equal(customers_df: pd.DataFrame):
    """Stratification: train and test both keep the 30% churn rate."""
    train_df, test_df = split_train_test(customers_df, test_size=TEST_SHARE)

    # pytest.approx allows tiny rounding differences in decimal numbers.
    assert churn_rate(train_df) == pytest.approx(EXPECTED_CHURN_RATE)
    assert churn_rate(test_df) == pytest.approx(EXPECTED_CHURN_RATE)


def test_split_is_reproducible_with_same_seed(customers_df: pd.DataFrame):
    """Same seed = exactly the same test customers every time."""
    _, first_test = split_train_test(customers_df, random_state=42)
    _, second_test = split_train_test(customers_df, random_state=42)

    assert list(first_test[ID_COLUMN]) == list(second_test[ID_COLUMN])


# Each bad value below runs as its own test.
@pytest.mark.parametrize("bad_test_size", [0, 1, 1.5, -0.2])
def test_split_rejects_bad_test_size(customers_df: pd.DataFrame, bad_test_size: float):
    """A test size outside 0 to 1 is refused with a clear error."""
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_train_test(customers_df, test_size=bad_test_size)


# --- save_splits ---


def test_save_splits_writes_readable_files(tmp_path: Path, customers_df: pd.DataFrame):
    """Saved files can be read back with the same number of rows."""
    train_df, test_df = split_train_test(customers_df)
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"

    save_splits(train_df, test_df, train_path=train_path, test_path=test_path)

    assert len(pd.read_csv(train_path)) == len(train_df)
    assert len(pd.read_csv(test_path)) == len(test_df)


def test_save_splits_refuses_to_overwrite(tmp_path: Path, customers_df: pd.DataFrame):
    """Saving a second time is blocked, so the test set cannot change by accident."""
    train_df, test_df = split_train_test(customers_df)
    paths = {"train_path": tmp_path / "train.csv", "test_path": tmp_path / "test.csv"}
    save_splits(train_df, test_df, **paths)

    with pytest.raises(FileExistsError, match="Split files already exist"):
        save_splits(train_df, test_df, **paths)


def test_save_splits_overwrites_when_asked(tmp_path: Path, customers_df: pd.DataFrame):
    """With overwrite=True, saving again is allowed."""
    train_df, test_df = split_train_test(customers_df)
    paths = {"train_path": tmp_path / "train.csv", "test_path": tmp_path / "test.csv"}
    save_splits(train_df, test_df, **paths)

    save_splits(train_df, test_df, overwrite=True, **paths)

    assert paths["test_path"].exists()


# --- churn_rate ---


def test_churn_rate_counts_share_of_yes(customers_df: pd.DataFrame):
    """30 churners out of 100 customers gives a churn rate of 0.3."""
    assert churn_rate(customers_df) == pytest.approx(EXPECTED_CHURN_RATE)
