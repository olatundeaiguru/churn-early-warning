"""How we measure a churn model for a team that can only call 20% of customers.

Main measure (decision A3): recall at the call budget = of ALL churners, what
share lands in the top-20% call list. Also: precision at the budget, ROC-AUC
and PR-AUC. Decisions: see docs/decisions.md (A10).
"""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import CALL_BUDGET_SHARE, CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


# Pick the customers the team would call: the top share by risk score.
# Ties (equal scores) are broken randomly but repeatably (fixed seed), so a
# score with many ties is neither helped nor hurt by the row order.
def top_share_mask(
    y_score: np.ndarray,
    share: float = CALL_BUDGET_SHARE,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Mark the customers inside the call list (highest scores first).

    Args:
        y_score: Risk score per customer (higher = more likely to churn).
        share: Share of customers the team can call (0 to 1).
        random_state: Seed for breaking ties between equal scores.

    Returns:
        True for customers in the call list, False for the rest.

    Raises:
        ValueError: If share is not between 0 and 1, or there are no scores.
    """
    if not 0 < share <= 1:
        raise ValueError(f"share must be between 0 and 1, got {share}.")
    scores = np.asarray(y_score, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot pick a call list from zero customers.")

    # Number of calls: share of customers, rounded up so the budget is used.
    n_calls = math.ceil(share * scores.size)
    # lexsort sorts by the LAST key first: highest score first, then a random
    # tie-breaker. Result = customer positions in calling order.
    tie_breaker = np.random.default_rng(random_state).random(scores.size)
    calling_order = np.lexsort((tie_breaker, -scores))

    in_call_list = np.zeros(scores.size, dtype=bool)
    in_call_list[calling_order[:n_calls]] = True
    return in_call_list


# Of ALL churners, what share is in the call list? (The main measure.)
# Example: 288 churners, 100 of them in the top 20% -> 100 / 288 = 0.35.
def recall_at_budget(
    y_true: np.ndarray, y_score: np.ndarray, share: float = CALL_BUDGET_SHARE
) -> float:
    """Share of all churners found inside the top-share call list.

    Args:
        y_true: 1 if the customer churned, else 0.
        y_score: Risk score per customer.
        share: Share of customers the team can call.

    Returns:
        Churners in the call list divided by all churners (0 to 1).

    Raises:
        ValueError: If there are no churners (recall is undefined).
    """
    churned = np.asarray(y_true) == 1
    total_churners = churned.sum()
    if total_churners == 0:
        raise ValueError("Recall is undefined: there are no churners.")
    in_call_list = top_share_mask(y_score, share)
    return float((churned & in_call_list).sum() / total_churners)


# Of the customers we call, what share are real churners? (Calls well spent.)
def precision_at_budget(
    y_true: np.ndarray, y_score: np.ndarray, share: float = CALL_BUDGET_SHARE
) -> float:
    """Share of the call list who really churned.

    Args:
        y_true: 1 if the customer churned, else 0.
        y_score: Risk score per customer.
        share: Share of customers the team can call.

    Returns:
        Churners in the call list divided by the size of the call list.
    """
    churned = np.asarray(y_true) == 1
    in_call_list = top_share_mask(y_score, share)
    return float((churned & in_call_list).sum() / in_call_list.sum())


# Scorers = the measures in a form cross_validate understands.
# response_method="predict_proba" passes the churn probability as y_score.
SCORING = {
    "recall_at_20": make_scorer(recall_at_budget, response_method="predict_proba"),
    "precision_at_20": make_scorer(
        precision_at_budget, response_method="predict_proba"
    ),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


# Shared by every stage, so all models are measured in exactly the same way:
# the same 5 folds (fixed seed), the same measures, the same summary.
def cross_validate_models(
    models: dict[str, Pipeline],
    x_raw: pd.DataFrame,
    y: pd.Series,
    include_train_scores: bool = False,
) -> pd.DataFrame:
    """Cross-validate each model pipeline with the same folds and measures.

    Args:
        models: Unfitted model pipelines, by name.
        x_raw: Raw training features (from split_raw_target).
        y: Training target (1 = churned).
        include_train_scores: Also report scores on the training folds. A big
            gap between train and check scores means overfitting.

    Returns:
        One row per model: mean and standard deviation of each measure.

    Raises:
        ValueError: If no models are given.
    """
    if not models:
        raise ValueError("No models to cross-validate.")
    # Same folds for every model, each keeping the 28.8% churn rate.
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, pipeline in models.items():
        logger.info("Cross-validating %s ...", name)
        scores = cross_validate(
            pipeline,
            x_raw,
            y,
            cv=folds,
            scoring=SCORING,
            return_train_score=include_train_scores,
        )
        rows[name] = _summarise_scores(scores, include_train_scores)
    return pd.DataFrame(rows).T


# Mean and spread (std) across the 5 folds, for each measure.
# Check-fold columns keep their Stage 8 names (e.g. recall_at_20_mean);
# training-fold columns get a "train_" prefix.
def _summarise_scores(
    scores: dict[str, np.ndarray], include_train_scores: bool
) -> dict[str, float]:
    summary = {}
    for measure in SCORING:
        fold_values = scores[f"test_{measure}"]
        summary[f"{measure}_mean"] = round(fold_values.mean(), 4)
        summary[f"{measure}_std"] = round(fold_values.std(), 4)
        if include_train_scores:
            train_values = scores[f"train_{measure}"]
            summary[f"train_{measure}_mean"] = round(train_values.mean(), 4)
    return summary
