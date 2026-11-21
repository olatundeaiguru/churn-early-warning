"""Project-wide paths and settings, kept in one place."""

from pathlib import Path

# Find the project folder from THIS file's location, not from where the code is run.
# config.py is in src/churn/, so we go up 2 folders: churn -> src -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- File locations ---
# The original dataset. Read-only: we never change or overwrite it.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "cell2celltrain.csv"
# The train and test sets created by churn.split.
INTERIM_DIR = DATA_DIR / "interim"
TRAIN_PATH = INTERIM_DIR / "train.csv"
TEST_PATH = INTERIM_DIR / "test.csv"

# --- Column names ---
# The unique customer number. Used for checks only, never as a model input.
ID_COLUMN = "CustomerID"
# The column we want to predict, and its two possible values.
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"  # the customer left (the case we want to catch)
NEGATIVE_LABEL = "No"  # the customer stayed

# --- Split settings ---
# 20% of customers go to the test set, kept untouched until the final check.
TEST_SIZE = 0.2
# A fixed seed, so the random split is exactly the same every time it runs.
RANDOM_STATE = 42
