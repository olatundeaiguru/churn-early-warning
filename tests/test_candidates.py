"""Tests for churn.candidates: the Stage 9 models are set up correctly.

Most tests check the set-up (names, scaling, seeds). The smoke test fits each
model on 300 FAKE customers (from conftest.py), never on the real data.

Run from the project root: python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.baselines import build_baselines
from churn.candidates import build_candidates
from churn.config import RANDOM_STATE
from churn.pipeline import split_raw_target

TREE_MODELS = ("random_forest", "lightgbm")


# Shortcut: the step names of a pipeline's numeric part (e.g. ["fill", "scale"]).
def numeric_step_names(pipeline: Pipeline) -> list[str]:
    """Return the step names used for numeric columns in a model pipeline."""
    encoder = pipeline.named_steps["preprocess"].named_steps["encode"]
    parts = {name: steps for name, steps, _columns in encoder.transformers}
    return list(parts["numeric"].named_steps)


# All 3 models exist, so the comparison table has a row for each.
def test_three_candidates_are_built():
    assert list(build_candidates()) == [
        "logistic_regression",
        "random_forest",
        "lightgbm",
    ]


# Logistic regression needs scaled numbers; trees do not (decision A9).
def test_only_logistic_regression_is_scaled():
    candidates = build_candidates()
    assert "scale" in numeric_step_names(candidates["logistic_regression"])
    for name in TREE_MODELS:
        assert "scale" not in numeric_step_names(candidates[name])


# The baseline must be EXACTLY the Stage 8 one, or the comparison is unfair.
def test_logistic_regression_matches_baseline():
    candidate = build_candidates()["logistic_regression"].named_steps["model"]
    baseline = build_baselines()["logistic_regression"].named_steps["model"]
    assert candidate.get_params() == baseline.get_params()


# A fixed seed means the same results every time the owner runs it.
@pytest.mark.parametrize("name", TREE_MODELS)
def test_tree_models_use_fixed_seed(name: str):
    model = build_candidates()[name].named_steps["model"]
    assert model.random_state == RANDOM_STATE


# Smoke test: each model runs end to end through the real pipeline, so a
# problem (e.g. LightGBM rejecting our column names) shows up here, not in the
# owner's real run.
@pytest.mark.parametrize("name", list(build_candidates()))
def test_candidate_fits_and_gives_probabilities(name: str, raw_df: pd.DataFrame):
    x_raw, target = split_raw_target(raw_df)
    pipeline = build_candidates()[name]

    pipeline.fit(x_raw, target)
    # predict_proba gives [P(stay), P(churn)] per customer; column 1 = churn.
    churn_probability = pipeline.predict_proba(x_raw)[:, 1]

    assert len(churn_probability) == len(raw_df)
    assert np.all((churn_probability >= 0) & (churn_probability <= 1))
