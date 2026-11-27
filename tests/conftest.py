"""Shared test data. pytest loads this file automatically for every test file.

Any fixture here (e.g. raw_df) can be used by a test just by naming it as an
argument; no import is needed.
"""

import numpy as np
import pandas as pd
import pytest

from churn.config import ID_COLUMN, TARGET_COLUMN
from churn.features import LEAKAGE_RISK_COLUMNS, LIFESTYLE_COLUMNS, NUMERIC_FEATURES

N_CUSTOMERS = 300
RANDOM_SEED = 0
BIG_CITY_SIZE = 250  # above MIN_CATEGORY_COUNT (100): gets its own column
# 300 - 250 = 50: must be BELOW MIN_CATEGORY_COUNT (100) to count as infrequent.
SMALL_CITY_SIZE = N_CUSTOMERS - BIG_CITY_SIZE
BIG_CITY_AREA = "NYCBRO917"
SMALL_CITY_AREA = "SLCSLC801"

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
    big_city = [BIG_CITY_AREA] * BIG_CITY_SIZE
    small_city = [SMALL_CITY_AREA] * SMALL_CITY_SIZE
    df["ServiceArea"] = big_city + small_city
    df["HandsetRefurbished"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["HandsetWebCapable"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["NewCellphoneUser"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["NotNewCellphoneUser"] = "No"
    df["PrizmCode"] = rng.choice(["Suburban", "Town", "Rural", "Other"], N_CUSTOMERS)
    df["CreditRating"] = rng.choice(["1-Highest", "4-Medium", "7-Lowest"], N_CUSTOMERS)
    # Excluded columns: present in the raw data, must be dropped by the pipeline.
    df[ID_COLUMN] = range(N_CUSTOMERS)
    for column in LEAKAGE_RISK_COLUMNS:
        df[column] = 1
    # Personal and lifestyle columns, with the same types as the real file.
    # Left out by default; used only when asked (A4 comparison).
    df["AgeHH1"] = rng.integers(0, 90, N_CUSTOMERS).astype(float)
    df["AgeHH2"] = rng.integers(0, 90, N_CUSTOMERS).astype(float)
    df["IncomeGroup"] = rng.integers(0, 10, N_CUSTOMERS)
    df["ChildrenInHH"] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["MaritalStatus"] = rng.choice(["Yes", "No", "Unknown"], N_CUSTOMERS)
    df["Occupation"] = rng.choice(["Professional", "Other"], N_CUSTOMERS)
    for column in LIFESTYLE_COLUMNS:
        df[column] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    df["Homeownership"] = rng.choice(["Known", "Unknown"], N_CUSTOMERS)
    # A few gaps, so the "was missing" flags have something to flag.
    df.loc[:9, "MonthlyRevenue"] = np.nan
    df[TARGET_COLUMN] = rng.choice(["Yes", "No"], N_CUSTOMERS)
    return df
