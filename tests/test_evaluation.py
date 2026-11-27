"""Tests for churn.evaluation: the call-budget measures.

Run from the project root: python -m pytest -v
"""

import numpy as np
import pytest

from churn.evaluation import precision_at_budget, recall_at_budget, top_share_mask

# 10 fake customers: the first 2 churned. Score = 9, 8, ..., 0 (perfect order).
Y_TRUE = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
PERFECT_SCORES = np.arange(10)[::-1].astype(float)
WORST_SCORES = np.arange(10).astype(float)  # churners ranked last
TWENTY_PERCENT = 0.2


# --- top_share_mask ---


def test_call_list_has_the_right_size():
    """20% of 10 customers = 2 calls."""
    assert top_share_mask(PERFECT_SCORES, TWENTY_PERCENT).sum() == 2


def test_call_list_rounds_up():
    """20% of 11 customers = 2.2, rounded UP to 3 calls (use the full budget)."""
    assert top_share_mask(np.arange(11), TWENTY_PERCENT).sum() == 3


def test_call_list_takes_highest_scores():
    """The customers with the highest scores are called first."""
    mask = top_share_mask(PERFECT_SCORES, TWENTY_PERCENT)
    assert list(np.flatnonzero(mask)) == [0, 1]


def test_ties_are_broken_the_same_way_every_time():
    """Equal scores: the random tie-break is repeatable (fixed seed)."""
    tied_scores = np.zeros(100)
    first = top_share_mask(tied_scores, TWENTY_PERCENT)
    second = top_share_mask(tied_scores, TWENTY_PERCENT)

    assert np.array_equal(first, second)


def test_ties_are_not_broken_by_row_order():
    """With all scores tied, the call list is NOT simply the first rows."""
    tied_scores = np.zeros(100)
    mask = top_share_mask(tied_scores, TWENTY_PERCENT)

    assert not mask[:20].all()


@pytest.mark.parametrize("bad_share", [0, -0.1, 1.5])
def test_bad_share_raises(bad_share: float):
    """A call share outside (0, 1] is refused."""
    with pytest.raises(ValueError, match="share must be between 0 and 1"):
        top_share_mask(PERFECT_SCORES, bad_share)


def test_no_customers_raises():
    """An empty list of scores gives a clear error."""
    with pytest.raises(ValueError, match="zero customers"):
        top_share_mask(np.array([]), TWENTY_PERCENT)


# --- recall_at_budget and precision_at_budget ---


def test_perfect_scores_catch_every_churner():
    """Perfect ranking: both churners are in the 2 calls."""
    assert recall_at_budget(Y_TRUE, PERFECT_SCORES, TWENTY_PERCENT) == 1.0
    assert precision_at_budget(Y_TRUE, PERFECT_SCORES, TWENTY_PERCENT) == 1.0


def test_worst_scores_catch_no_churner():
    """Churners ranked last: none are called."""
    assert recall_at_budget(Y_TRUE, WORST_SCORES, TWENTY_PERCENT) == 0.0
    assert precision_at_budget(Y_TRUE, WORST_SCORES, TWENTY_PERCENT) == 0.0


def test_recall_with_one_of_two_churners_called():
    """1 of 2 churners in the call list -> recall 0.5, precision 0.5."""
    scores = np.array([9, 0, 8, 1, 2, 3, 4, 5, 6, 7], dtype=float)

    assert recall_at_budget(Y_TRUE, scores, TWENTY_PERCENT) == 0.5
    assert precision_at_budget(Y_TRUE, scores, TWENTY_PERCENT) == 0.5


def test_recall_without_churners_raises():
    """Recall is undefined when nobody churned."""
    with pytest.raises(ValueError, match="no churners"):
        recall_at_budget(np.zeros(10), PERFECT_SCORES, TWENTY_PERCENT)


def test_random_scores_catch_about_the_call_share():
    """Random scores catch about 20% of churners: the random-calling floor."""
    rng = np.random.default_rng(0)
    y_true = rng.random(50_000) < 0.3
    random_scores = rng.random(50_000)

    recall = recall_at_budget(y_true, random_scores, TWENTY_PERCENT)

    assert recall == pytest.approx(TWENTY_PERCENT, abs=0.01)
