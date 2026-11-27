"""How we measure a churn model for a team that can only call 20% of customers.

Main measure (decision A3): recall at the call budget = of ALL churners, what
share lands in the top-20% call list. Also: precision at the budget, ROC-AUC
and PR-AUC. Decisions: see docs/decisions.md (A10).
"""

import math

import numpy as np
from sklearn.metrics import make_scorer

from churn.config import CALL_BUDGET_SHARE, RANDOM_STATE


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
