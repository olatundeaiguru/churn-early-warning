"""Load churn datasets and check they have the expected structure."""

from pathlib import Path

import pandas as pd

from churn.config import (
    ID_COLUMN,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TRAIN_PATH,
)

# Columns that must exist in every dataset, or nothing else can work.
REQUIRED_COLUMNS = (ID_COLUMN, TARGET_COLUMN)
# The only allowed Churn values. A frozenset cannot be changed by accident.
VALID_TARGET_VALUES = frozenset({POSITIVE_LABEL, NEGATIVE_LABEL})


# Main loader: open a CSV file, run all checks, give back the table.
# Every other loader below calls this one, so the loading logic exists once.
def load_dataset(path: Path) -> pd.DataFrame:
    """Load a churn CSV file and check its structure.

    Args:
        path: Location of the CSV file.

    Returns:
        The loaded data, one row per customer.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the data fails a structure check (see validate_dataset).
    """
    # Guard clause: stop early with a clear message if the file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    validate_dataset(df)
    return df


# Shortcut: load the original dataset (used only to make the split).
def load_raw_data() -> pd.DataFrame:
    """Load the original, untouched dataset from data/raw/."""
    return load_dataset(RAW_DATA_PATH)


# Shortcut: load the training set (used for EDA and modelling).
def load_train_data() -> pd.DataFrame:
    """Load the training split from data/interim/ (created by churn.split)."""
    return load_dataset(TRAIN_PATH)


# Run all 3 structure checks. Stops with an error at the first problem found.
def validate_dataset(df: pd.DataFrame) -> None:
    """Check that the data has the structure every later step relies on.

    Args:
        df: The data to check.

    Raises:
        ValueError: If required columns are missing, customer IDs repeat,
            or the target has missing or unexpected values.
    """
    _check_required_columns(df)
    _check_unique_ids(df)
    _check_target_values(df)


# Helper functions below start with "_": they are private, used only in this file.


# Check 1: CustomerID and Churn must both exist.
def _check_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# Check 2: each CustomerID appears only once (one row = one customer).
# Repeats could put the same customer in both train and test (leakage).
def _check_unique_ids(df: pd.DataFrame) -> None:
    repeated_ids = df[ID_COLUMN].duplicated().sum()
    if repeated_ids:
        raise ValueError(
            f"Found {repeated_ids} repeated {ID_COLUMN} values. "
            "Expected exactly one row per customer."
        )


# Check 3: Churn has no blanks, and only "Yes" or "No" (catches typos like "yes").
def _check_target_values(df: pd.DataFrame) -> None:
    missing_targets = df[TARGET_COLUMN].isna().sum()
    if missing_targets:
        raise ValueError(f"Found {missing_targets} rows with no {TARGET_COLUMN} value.")

    # Set subtraction: all values found, minus the allowed ones = the bad ones.
    unexpected_values = set(df[TARGET_COLUMN].unique()) - VALID_TARGET_VALUES
    if unexpected_values:
        raise ValueError(
            f"Unexpected {TARGET_COLUMN} values: {sorted(unexpected_values)}. "
            f"Expected only: {sorted(VALID_TARGET_VALUES)}."
        )
