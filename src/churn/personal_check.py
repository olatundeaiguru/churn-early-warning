"""A4 comparison: do personal or lifestyle columns help the shortlisted model?

Same LightGBM (decision A11), same 5 folds, 4 versions:
1. main_features: the 35 features (decision A8).
2. plus_personal: + 6 personal columns (age, income, children, ...).
3. plus_lifestyle: + 10 lifestyle columns (owns a truck, credit card, ...).
4. plus_both: + all 16.

Rule set BEFORE looking at results (data minimisation): a group is kept only
if it raises recall at 20% by at least MIN_RECALL_GAIN over main_features.

Run from the project root (the owner runs this, see CLAUDE.md section 6):
    python -m churn.personal_check
"""

import logging

import pandas as pd

from churn.candidates import build_lightgbm_pipeline
from churn.config import CV_FOLDS, REPORTS_DIR
from churn.data import load_train_data
from churn.evaluation import cross_validate_models
from churn.features import LIFESTYLE_GROUP, PERSONAL_GROUP
from churn.pipeline import split_raw_target

logger = logging.getLogger(__name__)

# About 3x LightGBM's fold-to-fold wobble (std 0.006): a real gain, not luck.
MIN_RECALL_GAIN = 0.02
REFERENCE_VERSION = "main_features"
RESULTS_PATH = REPORTS_DIR / "personal_check_results.csv"

# Version name -> extra feature groups added on top of the main features.
VERSIONS = {
    REFERENCE_VERSION: (),
    "plus_personal": (PERSONAL_GROUP,),
    "plus_lifestyle": (LIFESTYLE_GROUP,),
    "plus_both": (PERSONAL_GROUP, LIFESTYLE_GROUP),
}


# Compare each version with main_features and apply the rule.
def add_gain_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Add the recall gain over main_features and whether it passes the rule.

    Args:
        results: Output of cross_validate_models, one row per version,
            including a REFERENCE_VERSION row.

    Returns:
        A copy with recall_gain and passes_rule columns added.

    Raises:
        ValueError: If the reference version is missing.
    """
    if REFERENCE_VERSION not in results.index:
        raise ValueError(f"Results need a '{REFERENCE_VERSION}' row to compare to.")
    with_gain = results.copy()
    reference_recall = with_gain.loc[REFERENCE_VERSION, "recall_at_20_mean"]
    with_gain["recall_gain"] = (
        with_gain["recall_at_20_mean"] - reference_recall
    ).round(4)
    with_gain["passes_rule"] = with_gain["recall_gain"] >= MIN_RECALL_GAIN
    # The reference is what we compare against, so it cannot "pass" itself.
    with_gain.loc[REFERENCE_VERSION, "passes_rule"] = False
    return with_gain


# Load training data, cross-validate the 4 versions, save the table.
def main() -> None:
    """Run the A4 comparison on the training set and save the results."""
    x_raw, y = split_raw_target(load_train_data())
    models = {
        name: build_lightgbm_pipeline(extra_groups=groups)
        for name, groups in VERSIONS.items()
    }
    results = add_gain_columns(cross_validate_models(models, x_raw, y))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH)
    summary_columns = [
        "recall_at_20_mean",
        "recall_at_20_std",
        "roc_auc_mean",
        "recall_gain",
        "passes_rule",
    ]
    summary = results[summary_columns]
    logger.info("Results (mean over %d folds):\n%s", CV_FOLDS, summary)
    logger.info("Rule: keep a group only if recall_gain >= %.2f", MIN_RECALL_GAIN)
    logger.info("Saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
