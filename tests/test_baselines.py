"""Tests for churn.baselines: the customer-care rule and the baseline set.

Run from the project root: python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.baselines import CUSTOMER_CARE_COLUMN, CustomerCareRule, build_baselines

# 4 fake preprocessed customers with different numbers of care calls.
CARE_CALLS = [0.0, 5.0, 1.0, 10.0]


@pytest.fixture
def care_df() -> pd.DataFrame:
    """Four fake customers: only the CustomerCareCalls column matters."""
    return pd.DataFrame({CUSTOMER_CARE_COLUMN: CARE_CALLS, "Other": [1, 2, 3, 4]})


def test_rule_ranks_by_care_calls(care_df: pd.DataFrame):
    """More care calls = higher churn score; the order follows the calls."""
    rule = CustomerCareRule().fit(care_df, pd.Series([0, 1, 0, 1]))
    churn_score = rule.predict_proba(care_df)[:, 1]

    # argsort gives positions from lowest to highest score.
    assert list(np.argsort(churn_score)) == list(np.argsort(CARE_CALLS))


def test_rule_returns_valid_probabilities(care_df: pd.DataFrame):
    """Each row: [P(stay), P(churn)], both 0-1, adding up to 1."""
    rule = CustomerCareRule().fit(care_df, pd.Series([0, 1, 0, 1]))
    probabilities = rule.predict_proba(care_df)

    assert probabilities.shape == (len(CARE_CALLS), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_rule_without_care_column_raises(care_df: pd.DataFrame):
    """The rule cannot work without CustomerCareCalls: clear error."""
    with pytest.raises(ValueError, match=CUSTOMER_CARE_COLUMN):
        CustomerCareRule().fit(care_df.drop(columns=CUSTOMER_CARE_COLUMN), None)


def test_three_baselines_are_built():
    """Random calling, the care rule and logistic regression, all as pipelines."""
    baselines = build_baselines()

    assert set(baselines) == {
        "random_calling",
        "customer_care_rule",
        "logistic_regression",
    }
    assert all(isinstance(pipeline, Pipeline) for pipeline in baselines.values())
