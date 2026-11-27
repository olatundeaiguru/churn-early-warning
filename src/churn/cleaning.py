"""Rule-based cleaning: fixed rules that learn nothing from the data.

Because no rule learns from the data, the same cleaning is safe to apply to
train, test and new customers. It runs as the first step of the model pipeline.
Decisions: see docs/decisions.md (A7).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Labels used in the raw data ---
UNKNOWN_LABEL = "Unknown"
YES_LABEL = "Yes"

# --- Rule 1: values that can never be negative (money and days) ---
NON_NEGATIVE_COLUMNS = (
    "MonthlyRevenue",
    "TotalRecurringCharge",
    "CurrentEquipmentDays",
)

# --- Rule 2: phone price stored as text ("30", "150", "Unknown") ---
HANDSET_PRICE_COLUMN = "HandsetPrice"

# --- Rule 3: two Yes/No columns that are really one 3-way column ---
NEW_USER_COLUMN = "NewCellphoneUser"
NOT_NEW_USER_COLUMN = "NotNewCellphoneUser"
CELLPHONE_STATUS_COLUMN = "CellphoneUserStatus"
STATUS_NEW = "New"
STATUS_NOT_NEW = "NotNew"
STATUS_UNKNOWN = UNKNOWN_LABEL

# --- Rule 4: columns where 0 means "unknown" (nobody is 0 years old) ---
# AgeHH2 is left alone on purpose: 0 there may mean "no second person".
ZERO_MEANS_UNKNOWN_COLUMNS = ("AgeHH1", "IncomeGroup")

# Every column the rules need. Checked first, so a bad input fails clearly.
REQUIRED_COLUMNS = (
    *NON_NEGATIVE_COLUMNS,
    HANDSET_PRICE_COLUMN,
    NEW_USER_COLUMN,
    NOT_NEW_USER_COLUMN,
    *ZERO_MEANS_UNKNOWN_COLUMNS,
)


# Main function: run every cleaning rule in order on a COPY of the data.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all rule-based cleaning steps.

    Args:
        df: Raw customer data (one row per customer).

    Returns:
        A cleaned copy. The input is not changed.

    Raises:
        ValueError: If a column the rules need is missing, a HandsetPrice
            value is not a number or "Unknown", or a customer is marked as
            both a new and a not-new cellphone user.
    """
    _check_required_columns(df)

    # Work on a copy, so the caller's data is never changed by surprise.
    cleaned = df.copy()
    cleaned = _set_negatives_to_missing(cleaned)
    cleaned = _convert_handset_price_to_number(cleaned)
    cleaned = _combine_cellphone_user_columns(cleaned)
    cleaned = _set_zero_to_missing(cleaned)
    return cleaned


# Guard: stop early with a clear message if any needed column is missing.
def _check_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Cannot clean data, missing columns: {missing_columns}")


# Rule 1: negative money or days are impossible, so mark them as missing.
# The pipeline fills missing values later (learned from training data only).
def _set_negatives_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    for column in NON_NEGATIVE_COLUMNS:
        is_negative = df[column] < 0
        logger.debug(
            "Rule 1: %s negatives set to missing: %d", column, is_negative.sum()
        )
        # .mask replaces values where the condition is True (here: with NaN).
        df[column] = df[column].mask(is_negative)
    return df


# Rule 2: turn "30", "150" into numbers 30.0, 150.0, and "Unknown" into missing.
# The "was missing" flag added in the pipeline keeps the "Unknown" information.
def _convert_handset_price_to_number(df: pd.DataFrame) -> pd.DataFrame:
    prices = df[HANDSET_PRICE_COLUMN]
    is_unknown = prices.eq(UNKNOWN_LABEL)
    logger.debug("Rule 2: HandsetPrice 'Unknown' set to missing: %d", is_unknown.sum())

    # .where keeps values where the condition is True, the rest become NaN.
    known_prices = prices.where(~is_unknown)
    try:
        df[HANDSET_PRICE_COLUMN] = pd.to_numeric(known_prices)
    except ValueError as error:
        raise ValueError(
            f"{HANDSET_PRICE_COLUMN} must be a number or '{UNKNOWN_LABEL}'. {error}"
        ) from error
    return df


# Rule 3: combine the two Yes/No columns into ONE column with 3 answers.
#   NewCellphoneUser = Yes     -> "New"
#   NotNewCellphoneUser = Yes  -> "NotNew"
#   both No                    -> "Unknown"
def _combine_cellphone_user_columns(df: pd.DataFrame) -> pd.DataFrame:
    is_new = df[NEW_USER_COLUMN].eq(YES_LABEL)
    is_not_new = df[NOT_NEW_USER_COLUMN].eq(YES_LABEL)

    # Both "Yes" is a contradiction: fail loudly rather than guess.
    contradictions = (is_new & is_not_new).sum()
    if contradictions:
        raise ValueError(
            f"{contradictions} rows are both a new and a not-new cellphone user."
        )

    # np.select picks the first matching choice per row, else the default.
    df[CELLPHONE_STATUS_COLUMN] = np.select(
        [is_new, is_not_new], [STATUS_NEW, STATUS_NOT_NEW], default=STATUS_UNKNOWN
    )
    logger.debug(
        "Rule 3: %s counts: %s",
        CELLPHONE_STATUS_COLUMN,
        df[CELLPHONE_STATUS_COLUMN].value_counts().to_dict(),
    )
    return df.drop(columns=[NEW_USER_COLUMN, NOT_NEW_USER_COLUMN])


# Rule 4: 0 means "unknown" for age and income, so mark it as missing.
def _set_zero_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    for column in ZERO_MEANS_UNKNOWN_COLUMNS:
        is_zero = df[column].eq(0)
        logger.debug("Rule 4: %s zeros set to missing: %d", column, is_zero.sum())
        df[column] = df[column].mask(is_zero)
    return df
