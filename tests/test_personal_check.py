"""Tests for churn.personal_check: the A4 "+0.02 recall" rule.

Uses a small hand-made results table; trains no model.

Run from the project root: python -m pytest -v
"""

import pandas as pd
import pytest

from churn.personal_check import (
    MIN_RECALL_GAIN,
    REFERENCE_VERSION,
    VERSIONS,
    add_gain_columns,
)


# A fake results table: one version clearly better, one barely better.
@pytest.fixture
def fake_results() -> pd.DataFrame:
    """Recall per version, shaped like cross_validate_models output."""
    return pd.DataFrame(
        {"recall_at_20_mean": [0.33, 0.36, 0.34]},
        index=[REFERENCE_VERSION, "plus_personal", "plus_lifestyle"],
    )


def test_gain_is_measured_against_main_features(fake_results: pd.DataFrame):
    """Gain = version recall - main_features recall."""
    gains = add_gain_columns(fake_results)["recall_gain"]
    assert gains["plus_personal"] == pytest.approx(0.03)
    assert gains["plus_lifestyle"] == pytest.approx(0.01)


def test_only_a_clear_gain_passes_the_rule(fake_results: pd.DataFrame):
    """+0.03 passes (>= 0.02); +0.01 does not; the reference never passes."""
    passes = add_gain_columns(fake_results)["passes_rule"]
    assert MIN_RECALL_GAIN == 0.02
    assert bool(passes["plus_personal"]) is True
    assert bool(passes["plus_lifestyle"]) is False
    assert bool(passes[REFERENCE_VERSION]) is False


def test_missing_reference_raises(fake_results: pd.DataFrame):
    """Without main_features there is nothing to compare against."""
    with pytest.raises(ValueError, match=REFERENCE_VERSION):
        add_gain_columns(fake_results.drop(index=REFERENCE_VERSION))


def test_main_features_version_adds_nothing():
    """The reference version must be the plain A8 feature set."""
    assert VERSIONS[REFERENCE_VERSION] == ()
