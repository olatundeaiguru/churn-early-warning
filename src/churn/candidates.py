"""Stage 9 candidates: can stronger models clearly beat logistic regression?

Models (default settings, no tuning yet; tuning is Stage 11):
1. Logistic regression: the Stage 8 baseline, reused unchanged.
2. Random Forest: many full-depth trees that vote.
3. LightGBM: trees built one after another, each fixing the last one's errors.

Success target (decision A10): recall at 20% >= 0.31 and ROC-AUC >= 0.64.

Run from the project root (the owner runs this, see CLAUDE.md section 6):
    python -m churn.candidates
"""

import logging
from collections.abc import Sequence

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn.baselines import build_baselines
from churn.config import CV_FOLDS, RANDOM_STATE, REPORTS_DIR
from churn.data import load_train_data
from churn.evaluation import cross_validate_models
from churn.pipeline import build_model_pipeline, split_raw_target

logger = logging.getLogger(__name__)

USE_ALL_CPU_CORES = -1  # n_jobs=-1: only changes speed, not results
LIGHTGBM_HIDE_INFO = -1  # verbose=-1: hide info messages; errors still show
RESULTS_PATH = REPORTS_DIR / "candidate_results.csv"


# The shortlisted model (decision A11), in one place so every stage uses the
# exact same settings. extra_groups is only used for the A4 comparison.
def build_lightgbm_pipeline(extra_groups: Sequence[str] = ()) -> Pipeline:
    """Return an unfitted LightGBM pipeline (default settings, fixed seed).

    Args:
        extra_groups: Optional extra feature groups (see build_preprocessor).

    Returns:
        Preprocessing (unscaled) + LightGBM, as one pipeline.
    """
    return build_model_pipeline(
        LGBMClassifier(random_state=RANDOM_STATE, verbose=LIGHTGBM_HIDE_INFO),
        scale_numeric=False,
        extra_groups=extra_groups,
    )


# The 3 models to compare. Trees split on thresholds, so they do not need
# scaling; only logistic regression is scaled (decided in A9).
def build_candidates() -> dict[str, Pipeline]:
    """Return the unfitted candidate pipelines, by name.

    Returns:
        Logistic regression (the baseline), Random Forest and LightGBM, each
        as a full pipeline with the same preprocessing.
    """
    return {
        # Reused from Stage 8, so it is exactly the same baseline.
        "logistic_regression": build_baselines()["logistic_regression"],
        "random_forest": build_model_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=USE_ALL_CPU_CORES),
            scale_numeric=False,
        ),
        "lightgbm": build_lightgbm_pipeline(),
    }


# Load training data, cross-validate, print and save the results table.
def main() -> None:
    """Run all candidates on the training set and save the results."""
    x_raw, y = split_raw_target(load_train_data())
    # Training scores too, so we can spot overfitting (train >> check score).
    results = cross_validate_models(
        build_candidates(), x_raw, y, include_train_scores=True
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH)
    logger.info("Results (mean and std over %d folds):\n%s", CV_FOLDS, results)
    logger.info("Saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
