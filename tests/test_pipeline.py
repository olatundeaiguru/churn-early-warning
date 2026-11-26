"""Tests for churn.pipeline: the preprocessing pipeline.

These check the leakage-critical behaviour: what the pipeline learns comes
from the data it is FITTED on, and it applies that unchanged to new data.

Run from the project root: python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import ID_COLUMN, TARGET_COLUMN
from churn.features import (
    CATEGORICAL_FEATURES,
    LEAKAGE_RISK_COLUMNS,
    NUMERIC_FEATURES,
    PERSONAL_COLUMNS,
)
from churn.pipeline import (
    MIN_CATEGORY_COUNT,
    build_model_pipeline,
    build_preprocessor,
    split_raw_target,
)

N_CUSTOMERS = 300
RANDOM_SEED = 0
BIG_CITY_SIZE = 250  # above MIN_CATEGORY_COUNT (100): gets its own column
# 300 - 250 = 50: must be BELOW MIN_CATEGORY_COUNT (100) to count as infrequent.
SMALL_CITY_SIZE = N_CUSTOMERS - BIG_CITY_SIZE
MAX_PROBABILITY = 1

# Raw numeric columns (ChangedPhone is created by the pipeline, not raw).
RAW_NUMERIC_COLUMNS = [c for c in NUMERIC_FEATURES if c not in ("ChangedPhone",)]


# Fake RAW data shaped like the real file: every column the pipeline needs,
# plus excluded ones (ID, retention, personal) that must NOT come out.
# A fixed seed makes the "random" values the same on every run.
@pytest.fixture
def raw_df() -> pd.DataFrame:
    """300 fake raw customers in the same format as cell2celltrain.csv."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.DataFrame(
        {column: rng.uniform(1, 100, N_CUSTOMERS) for column in RAW_NUMERIC_COLUMNS}
    )
    df["MonthsInService"] = rng.integers(6, 61, N_CUSTOMERS)
    df["CurrentEquipmentDays"] = rng.integers(1, 1800, N_CUSTOMERS).astype(float)
    df["HandsetPrice"] = rng.choice(["30", "150", "Unknown"], N_CUSTOMERS)
    df["ServiceArea"] = ["NYCBRO917"] * BIG_CITY_SIZE + ["SLCSLC801"] * SMALL_CITY_SIZE
    df["HandsetRefurbished"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["HandsetWebCapable"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["NewCellphoneUser"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["NotNewCellphoneUser"] = "No"
    df["PrizmCode"] = rng.choice(["Suburban", "Town", "Rural", "Other"], N_CUSTOMERS)
    df["CreditRating"] = rng.choice(["1-Highest", "4-Medium", "7-Lowest"], N_CUSTOMERS)
    # Excluded columns: present in the raw data, must be dropped by the pipeline.
    df[ID_COLUMN] = range(N_CUSTOMERS)
    for column in (*LEAKAGE_RISK_COLUMNS, *PERSONAL_COLUMNS):
        df[column] = 1
    # A few gaps, so the "was missing" flags have something to flag.
    df.loc[:9, "MonthlyRevenue"] = np.nan
    df[TARGET_COLUMN] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    return df


# Shortcut: split off the target and fit an UNSCALED preprocessor.
def fit_preprocessor(raw_df: pd.DataFrame, scale_numeric: bool = False):
    """Fit a preprocessor on the fake data; return it and the transformed X."""
    x_raw, _ = split_raw_target(raw_df)
    preprocessor = build_preprocessor(scale_numeric=scale_numeric)
    return preprocessor, preprocessor.fit_transform(x_raw)


# --- split_raw_target ---


def test_split_raw_target_separates_churn(raw_df: pd.DataFrame):
    """X keeps every raw column except Churn; y is 1/0."""
    x_raw, target = split_raw_target(raw_df)

    assert TARGET_COLUMN not in x_raw
    assert len(x_raw.columns) == len(raw_df.columns) - 1
    assert set(target.unique()) <= {0, 1}


def test_split_raw_target_missing_churn_raises(raw_df: pd.DataFrame):
    """No Churn column gives a clear error."""
    with pytest.raises(ValueError, match="missing column"):
        split_raw_target(raw_df.drop(columns=TARGET_COLUMN))


# --- Output shape and content ---


def test_output_has_no_missing_values(raw_df: pd.DataFrame):
    """Every gap is filled: the model never sees a missing value."""
    _, transformed = fit_preprocessor(raw_df)

    assert not transformed.isna().any().any()


def test_output_is_all_numbers(raw_df: pd.DataFrame):
    """Text categories are turned into numbers the model can use."""
    _, transformed = fit_preprocessor(raw_df)

    # dtype.kind: f = float, i = integer, u = unsigned integer, b = boolean.
    assert all(dtype.kind in "fiub" for dtype in transformed.dtypes)


def test_excluded_columns_never_come_out(raw_df: pd.DataFrame):
    """ID, retention (leakage risk) and personal columns are dropped."""
    _, transformed = fit_preprocessor(raw_df)

    for column in (ID_COLUMN, *LEAKAGE_RISK_COLUMNS, *PERSONAL_COLUMNS):
        assert column not in transformed


def test_missing_flag_is_added(raw_df: pd.DataFrame):
    """A "was missing" flag marks the 10 customers with no MonthlyRevenue."""
    _, transformed = fit_preprocessor(raw_df)

    flag = transformed["missingindicator_MonthlyRevenue"]
    assert flag.sum() == 10


# --- Leakage safety: learn from the FIT data only ---


def test_gaps_are_filled_with_the_training_median(raw_df: pd.DataFrame):
    """A gap in NEW data is filled with the median learned from TRAINING data."""
    preprocessor, _ = fit_preprocessor(raw_df)
    x_raw, _ = split_raw_target(raw_df)
    training_median = x_raw["DroppedCalls"].median()

    # One new customer with DroppedCalls missing.
    new_customer = x_raw.head(1).copy()
    new_customer["DroppedCalls"] = np.nan
    filled_value = preprocessor.transform(new_customer)["DroppedCalls"].iloc[0]

    assert filled_value == pytest.approx(training_median)


def test_new_data_does_not_change_what_was_learned(raw_df: pd.DataFrame):
    """Transforming other data later does not re-learn the medians."""
    preprocessor, first_output = fit_preprocessor(raw_df)
    x_raw, _ = split_raw_target(raw_df)

    # Transform very different data, then the original data again.
    preprocessor.transform(x_raw.assign(DroppedCalls=x_raw["DroppedCalls"] * 1000))
    second_output = preprocessor.transform(x_raw)

    pd.testing.assert_frame_equal(first_output, second_output)


# --- Categories ---


def test_small_city_is_grouped_as_infrequent(raw_df: pd.DataFrame):
    """A city below MIN_CATEGORY_COUNT customers joins the 'infrequent' group."""
    assert SMALL_CITY_SIZE < MIN_CATEGORY_COUNT < BIG_CITY_SIZE
    _, transformed = fit_preprocessor(raw_df)

    assert "CityCode_NYC" in transformed
    assert "CityCode_SLC" not in transformed
    assert "CityCode_infrequent_sklearn" in transformed


def test_unseen_city_does_not_crash(raw_df: pd.DataFrame):
    """A city never seen in training goes to the 'infrequent' group, no error."""
    preprocessor, _ = fit_preprocessor(raw_df)
    x_raw, _ = split_raw_target(raw_df)
    new_customer = x_raw.head(1).assign(ServiceArea="ZZZNEW000")

    transformed = preprocessor.transform(new_customer)

    assert transformed["CityCode_infrequent_sklearn"].iloc[0] == 1


def test_credit_rating_keeps_its_order(raw_df: pd.DataFrame):
    """'1-Highest' -> 0 and '7-Lowest' -> 6 (unscaled ranks)."""
    preprocessor, _ = fit_preprocessor(raw_df)
    x_raw, _ = split_raw_target(raw_df)
    two_customers = x_raw.head(2).assign(CreditRating=["1-Highest", "7-Lowest"])

    ranks = preprocessor.transform(two_customers)["CreditRating"]

    assert list(ranks) == [0, 6]


def test_all_categorical_features_are_encoded(raw_df: pd.DataFrame):
    """Every categorical feature produces at least one 0/1 column."""
    _, transformed = fit_preprocessor(raw_df)

    for feature in CATEGORICAL_FEATURES:
        assert any(column.startswith(f"{feature}_") for column in transformed)


# --- Scaling ---


def test_scaling_centres_numbers_when_asked(raw_df: pd.DataFrame):
    """With scaling on, number columns have an average of about 0."""
    _, transformed = fit_preprocessor(raw_df, scale_numeric=True)

    assert transformed["MonthsInService"].mean() == pytest.approx(0, abs=1e-9)


def test_no_scaling_keeps_original_values(raw_df: pd.DataFrame):
    """With scaling off (tree models), numbers keep their real values."""
    _, transformed = fit_preprocessor(raw_df, scale_numeric=False)
    x_raw, _ = split_raw_target(raw_df)

    assert transformed["MonthsInService"].tolist() == x_raw["MonthsInService"].tolist()


# --- Full pipeline with a model (smoke test) ---


def test_full_pipeline_predicts_probabilities(raw_df: pd.DataFrame):
    """Preprocessing + model fits and returns one churn probability per customer."""
    x_raw, target = split_raw_target(raw_df)
    pipeline = build_model_pipeline(LogisticRegression(), scale_numeric=True)

    pipeline.fit(x_raw, target)
    # predict_proba gives [P(stay), P(churn)] per customer; column 1 = churn.
    churn_probability = pipeline.predict_proba(x_raw)[:, 1]

    assert len(churn_probability) == N_CUSTOMERS
    assert np.all((churn_probability >= 0) & (churn_probability <= MAX_PROBABILITY))
