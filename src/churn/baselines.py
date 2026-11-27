"""Stage 8 baselines: simple benchmarks every real model must beat (A5).

1. Random calling: everyone gets the same score, so the call list is random.
2. Customer-care rule: call the customers with the most customer care calls.
3. Logistic regression: the first real model, through the full pipeline.

Run from the project root (takes about a minute):
    python -m churn.baselines
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import CV_FOLDS, RANDOM_STATE, REPORTS_DIR
from churn.data import load_train_data
from churn.evaluation import SCORING
from churn.pipeline import build_model_pipeline, split_raw_target

logger = logging.getLogger(__name__)

CUSTOMER_CARE_COLUMN = "CustomerCareCalls"
LOGREG_MAX_ITER = 1000  # enough steps for logistic regression to finish
RESULTS_PATH = REPORTS_DIR / "baseline_results.csv"


# The "common sense" rule as a tiny model, so it can be measured exactly like
# a real model. It learns nothing: risk score = number of customer care calls.
class CustomerCareRule(ClassifierMixin, BaseEstimator):
    """Rule baseline: more customer care calls = higher churn risk."""

    # Nothing to learn. Only check the column exists and record the classes.
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CustomerCareRule":
        """Check the input has the CustomerCareCalls column.

        Args:
            X: Preprocessed features (must include CustomerCareCalls).
            y: Target (1 = churned). Not used by the rule.

        Returns:
            The rule itself.

        Raises:
            ValueError: If CustomerCareCalls is not in X.
        """
        if CUSTOMER_CARE_COLUMN not in X:
            raise ValueError(f"The rule needs the column {CUSTOMER_CARE_COLUMN}.")
        self.classes_ = np.array([0, 1])
        return self

    # Turn care calls into a 0-1 "probability" by rank: the customer with the
    # most calls gets 1. Only the ORDER matters for our measures.
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return [P(stay), P(churn)] per customer, ranked by care calls."""
        churn_score = X[CUSTOMER_CARE_COLUMN].rank(pct=True).to_numpy()
        return np.column_stack([1 - churn_score, churn_score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn (1) when the rank score is above one half."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# The 3 baselines, each as a full pipeline (same preprocessing for all).
def build_baselines() -> dict[str, Pipeline]:
    """Return the 3 unfitted baseline pipelines, by name."""
    return {
        # strategy="prior": the same score for everyone -> random call list
        # (ties are broken randomly by the measures).
        "random_calling": build_model_pipeline(
            DummyClassifier(strategy="prior"), scale_numeric=False
        ),
        "customer_care_rule": build_model_pipeline(
            CustomerCareRule(), scale_numeric=False
        ),
        "logistic_regression": build_model_pipeline(
            LogisticRegression(max_iter=LOGREG_MAX_ITER), scale_numeric=True
        ),
    }


# Run 5-fold cross-validation for each baseline and summarise the results.
def evaluate_baselines(x_raw: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validate every baseline with the same folds and measures.

    Args:
        x_raw: Raw training features (from split_raw_target).
        y: Training target (1 = churned).

    Returns:
        One row per baseline: mean and standard deviation of each measure.
    """
    # Same folds for every baseline, each keeping the 28.8% churn rate.
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, pipeline in build_baselines().items():
        logger.info("Cross-validating %s ...", name)
        scores = cross_validate(pipeline, x_raw, y, cv=folds, scoring=SCORING)
        rows[name] = _summarise_scores(scores)
    return pd.DataFrame(rows).T


# Mean and spread (std) across the 5 folds, for each measure.
def _summarise_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {}
    for measure in SCORING:
        fold_values = scores[f"test_{measure}"]
        summary[f"{measure}_mean"] = round(fold_values.mean(), 4)
        summary[f"{measure}_std"] = round(fold_values.std(), 4)
    return summary


# Load training data, evaluate, print and save the results table.
def main() -> None:
    """Run all baselines on the training set and save the results."""
    x_raw, y = split_raw_target(load_train_data())
    results = evaluate_baselines(x_raw, y)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH)
    logger.info("Results (mean and std over %d folds):\n%s", CV_FOLDS, results)
    logger.info("Saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
