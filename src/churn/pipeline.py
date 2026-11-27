"""The preprocessing pipeline: raw customer data in, model-ready numbers out.

    raw data -> clean_data -> add_features -> fill gaps + "was missing" flags
             -> encode categories -> (scale numbers) -> model

Every step that LEARNS from data (medians, category lists, scaling) learns
from the data the pipeline is fitted on: training data only. The same fitted
pipeline is then applied unchanged to test data and new customers.
Decisions: see docs/decisions.md (A9).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from churn.cleaning import clean_data
from churn.config import TARGET_COLUMN
from churn.features import (
    CATEGORICAL_FEATURES,
    CREDIT_RATING_ORDER,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    add_features,
    encode_target,
    extra_feature_columns,
)

# Categories seen fewer times than this in training are grouped together
# (e.g. small cities). Categories never seen in training join the same group.
MIN_CATEGORY_COUNT = 100
# Label used to fill missing text values before encoding.
MISSING_CATEGORY = "Missing"


# Split raw data into X (everything except the answer) and y (1/0).
# X keeps ALL raw columns: the pipeline itself cleans and selects them.
def split_raw_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target from raw customer data.

    Args:
        df: Raw data as loaded by churn.data (includes Churn).

    Returns:
        X_raw: all columns except Churn (the pipeline picks the features).
        y: 1 if the customer churned, else 0.

    Raises:
        ValueError: If the Churn column is missing.
    """
    if TARGET_COLUMN not in df:
        raise ValueError(f"Cannot split target, missing column: {TARGET_COLUMN}")
    return df.drop(columns=TARGET_COLUMN), encode_target(df[TARGET_COLUMN])


# Build the preprocessing steps. scale_numeric=True for logistic regression,
# False for tree models (they do not need scaling).
# extra_groups adds optional columns (e.g. "personal") for the A4 comparison;
# by default the main 35 features are used, as decided in A8.
def build_preprocessor(
    scale_numeric: bool, extra_groups: Sequence[str] = ()
) -> Pipeline:
    """Build the preprocessing pipeline (not yet fitted).

    Args:
        scale_numeric: Scale numbers to a similar range. Needed for
            logistic regression; not needed for tree models.
        extra_groups: Optional extra feature groups from
            churn.features.EXTRA_FEATURE_GROUPS. Empty = main features only.

    Returns:
        An unfitted Pipeline: clean -> add features -> encode columns.

    Raises:
        ValueError: If a group name in extra_groups is unknown.
    """
    extra = extra_feature_columns(extra_groups)
    numeric_columns = [*NUMERIC_FEATURES, *extra.numeric]
    categorical_columns = [*CATEGORICAL_FEATURES, *extra.categorical]
    column_encoder = ColumnTransformer(
        [
            ("numeric", _numeric_steps(scale_numeric), numeric_columns),
            ("categorical", _categorical_steps(), categorical_columns),
            ("ordinal", _ordinal_steps(scale_numeric), list(ORDINAL_FEATURES)),
        ],
        # Any column not listed above (ID, excluded columns) is dropped here.
        remainder="drop",
        # Keep plain column names (e.g. "MonthsInService", not "numeric__...").
        verbose_feature_names_out=False,
    )
    preprocessor = Pipeline(
        [
            # Fixed rules (learn nothing): safe to apply to any data.
            ("clean", FunctionTransformer(clean_data)),
            ("features", FunctionTransformer(add_features)),
            # Learned steps: fitted on training data only.
            ("encode", column_encoder),
        ]
    )
    # Output a pandas table with named columns (needed for SHAP explanations).
    return preprocessor.set_output(transform="pandas")


# Full pipeline = preprocessing + model, fitted and saved as ONE object.
def build_model_pipeline(
    model: BaseEstimator, scale_numeric: bool, extra_groups: Sequence[str] = ()
) -> Pipeline:
    """Combine the preprocessor and a model into one pipeline.

    Args:
        model: An unfitted scikit-learn compatible classifier.
        scale_numeric: Passed to build_preprocessor.
        extra_groups: Passed to build_preprocessor (default: none).

    Returns:
        An unfitted Pipeline: preprocess -> model.
    """
    preprocessor = build_preprocessor(scale_numeric, extra_groups)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


# Numbers: fill gaps with the TRAINING median and add a "was missing" flag
# (missing usage data was a strong churn signal in EDA). Then scale if asked.
def _numeric_steps(scale_numeric: bool) -> Pipeline:
    steps = [("fill", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# Categories: fill gaps with "Missing", then one 0/1 column per category.
# Rare categories (< 100 customers) and unseen ones share one "infrequent" column.
def _categorical_steps() -> Pipeline:
    return Pipeline(
        [
            (
                "fill",
                SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_COUNT,
                    sparse_output=False,
                ),
            ),
        ]
    )


# CreditRating: turn "1-Highest" ... "7-Lowest" into ranks 0 ... 6.
# An unexpected rating becomes missing, then gets the training median.
def _ordinal_steps(scale_numeric: bool) -> Pipeline:
    steps = [
        (
            "rank",
            OrdinalEncoder(
                categories=[list(CREDIT_RATING_ORDER)],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
        ),
        ("fill", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
