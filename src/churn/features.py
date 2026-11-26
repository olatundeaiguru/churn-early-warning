"""Feature selection and creation: which columns the model may use, and why.

Every column of the cleaned data is either a model feature or excluded with a
reason (decision A8). New features are built with fixed rules only, so this
step is safe to apply identically to train, test and new customers.
"""

import pandas as pd

from churn.config import ID_COLUMN, POSITIVE_LABEL, TARGET_COLUMN

# ---------------------------------------------------------------------------
# Excluded columns, grouped by reason
# ---------------------------------------------------------------------------

# Possible leakage: may be recorded after the snapshot (decisions A2, A8).
LEAKAGE_RISK_COLUMNS = (
    "RetentionCalls",
    "RetentionOffersAccepted",
    "MadeCallToRetentionTeam",
)

# Personal data: out of the first model, kept for the fairness check (A4).
PERSONAL_COLUMNS = (
    "AgeHH1",
    "AgeHH2",
    "ChildrenInHH",
    "IncomeGroup",
    "MaritalStatus",
    "Occupation",
)

# Lifestyle data about the customer: treated like personal data (A8).
LIFESTYLE_COLUMNS = (
    "TruckOwner",
    "RVOwner",
    "OwnsMotorcycle",
    "OwnsComputer",
    "HasCreditCard",
    "NonUSTravel",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
    "OptOutMailings",
    "Homeownership",
)

# Repeat information another column already has (EDA Part C).
REDUNDANT_COLUMNS = (
    "DroppedBlockedCalls",  # mostly DroppedCalls + BlockedCalls
    "HandsetModels",  # moves with Handsets (correlation 0.91)
)

# Replaced by a simpler version created in add_features().
SERVICE_AREA_COLUMN = "ServiceArea"  # 730 areas -> 58 city codes

# Not inputs at all: the customer label and the answer we predict.
NOT_INPUT_COLUMNS = (ID_COLUMN, TARGET_COLUMN)

# ---------------------------------------------------------------------------
# New features (built in add_features)
# ---------------------------------------------------------------------------

CITY_CODE_COLUMN = "CityCode"
CITY_CODE_LENGTH = 3  # "NYCBRO917" -> "NYC"

CHANGED_PHONE_COLUMN = "ChangedPhone"
DAYS_PER_MONTH = 365.25 / 12  # average month length, leap years included
CHANGED_PHONE_GAP_MONTHS = 1  # phone at least 1 month newer than the account

# ---------------------------------------------------------------------------
# Model features, grouped by how the pipeline will treat them (Stage 7)
# ---------------------------------------------------------------------------

# Numbers: filled and scaled in the pipeline.
NUMERIC_FEATURES = (
    # Billing and usage
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "DirectorAssistedCalls",
    "OverageMinutes",
    "RoamingCalls",
    "PercChangeMinutes",
    "PercChangeRevenues",
    # Calls and call quality
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "CustomerCareCalls",
    "ThreewayCalls",
    "ReceivedCalls",
    "OutboundCalls",
    "InboundCalls",
    "PeakCallsInOut",
    "OffPeakCallsInOut",
    "CallForwardingCalls",
    "CallWaitingCalls",
    # Account and phone
    "MonthsInService",
    "UniqueSubs",
    "ActiveSubs",
    "Handsets",
    "CurrentEquipmentDays",
    "HandsetPrice",
    "ReferralsMadeBySubscriber",
    "AdjustmentsToCreditRating",
    CHANGED_PHONE_COLUMN,
)

# Categories with no natural order: one-hot encoded in the pipeline.
CATEGORICAL_FEATURES = (
    "HandsetRefurbished",
    "HandsetWebCapable",
    "CellphoneUserStatus",
    "PrizmCode",
    CITY_CODE_COLUMN,
)

# Categories WITH a natural order: encoded as ranks in the pipeline.
CREDIT_RATING_COLUMN = "CreditRating"
CREDIT_RATING_ORDER = (
    "1-Highest",
    "2-High",
    "3-Good",
    "4-Medium",
    "5-Low",
    "6-VeryLow",
    "7-Lowest",
)
ORDINAL_FEATURES = (CREDIT_RATING_COLUMN,)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES

# Everything left out of the model, for checks and documentation.
EXCLUDED_COLUMNS = (
    *NOT_INPUT_COLUMNS,
    *LEAKAGE_RISK_COLUMNS,
    *PERSONAL_COLUMNS,
    *LIFESTYLE_COLUMNS,
    *REDUNDANT_COLUMNS,
    SERVICE_AREA_COLUMN,
)


# Build the 2 new features from columns the customer data already has.
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CityCode and ChangedPhone to a cleaned dataset.

    Args:
        df: Cleaned customer data (output of churn.cleaning.clean_data).

    Returns:
        A copy with the two new columns added. The input is not changed.

    Raises:
        ValueError: If a column needed to build the features is missing.
    """
    needed_columns = [SERVICE_AREA_COLUMN, "CurrentEquipmentDays", "MonthsInService"]
    missing_columns = [column for column in needed_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Cannot build features, missing columns: {missing_columns}")

    featured = df.copy()
    featured[CITY_CODE_COLUMN] = _city_code(featured[SERVICE_AREA_COLUMN])
    featured[CHANGED_PHONE_COLUMN] = _changed_phone(
        featured["CurrentEquipmentDays"], featured["MonthsInService"]
    )
    return featured


# "NYCBRO917" -> "NYC". A missing area gives a missing city code.
def _city_code(service_area: pd.Series) -> pd.Series:
    return service_area.str[:CITY_CODE_LENGTH]


# 1 = phone at least 1 month newer than the account (customer changed phone),
# 0 = same phone since joining, missing = phone age unknown.
def _changed_phone(
    equipment_days: pd.Series, months_in_service: pd.Series
) -> pd.Series:
    phone_age_months = equipment_days / DAYS_PER_MONTH
    changed = phone_age_months < (months_in_service - CHANGED_PHONE_GAP_MONTHS)
    # Keep "unknown" honest: no phone age means we cannot say 0 or 1.
    return changed.astype(float).mask(equipment_days.isna())


# Split data into X (model inputs) and y (1 = churned, 0 = stayed).
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model features and turn Churn into 1/0.

    Args:
        df: Cleaned data with features added (output of add_features).

    Returns:
        X: only the MODEL_FEATURES columns, in a fixed order.
        y: 1 if the customer churned, else 0.

    Raises:
        ValueError: If the target or any model feature is missing.
    """
    missing_columns = [
        column for column in (*MODEL_FEATURES, TARGET_COLUMN) if column not in df
    ]
    if missing_columns:
        raise ValueError(f"Cannot split X and y, missing columns: {missing_columns}")

    features = df[list(MODEL_FEATURES)]
    return features, encode_target(df[TARGET_COLUMN])


# Turn the Churn labels into numbers: "Yes" -> 1, "No" -> 0.
# Shared by this module and the pipeline, so the rule is written once.
def encode_target(churn_labels: pd.Series) -> pd.Series:
    """Convert Churn labels to 1 (churned) and 0 (stayed).

    Args:
        churn_labels: The Churn column ("Yes"/"No"), already checked by
            churn.data.validate_dataset.

    Returns:
        1 where the label is "Yes", else 0.
    """
    return churn_labels.eq(POSITIVE_LABEL).astype(int)
