# Project Status: Churn Early-Warning System

_Last updated: 2026-11-26_

## Current stage

**Stage 7: Build preprocessing (complete).**
Stage 6 (split) was done early (A6). Next stage: **Stage 8: Establish a baseline** (not started).

## Next single step

Start Stage 8: run the 3 baselines (random calling, customer-care rule, logistic regression) with cross-validation on the training set, then set the success target number (A3).

## Problem summary

| Item | Answer |
|---|---|
| Client | Head of Data Science, Cell2Cell Telecom (scenario project) |
| Problem | About 29% of customers churn. The retention team only learns about it after the customer has cancelled. |
| Goal | Score every active customer on churn risk, so the team can call the right people. |
| Users | The retention team. They can call about **20% of customers per month**. |
| Task type | Binary classification, used to rank customers by risk. |
| Unit of observation | One row = one customer. |
| Target | `Churn` (Yes/No): the customer left **31 to 60 days after** their data was recorded. |
| Prediction time | The moment the customer's data was recorded (the snapshot). Only information known at that moment may be used. |
| Most costly error | Missing a churner (false negative). A wasted call (false positive) costs less. |
| Main metric | Recall and precision within the **top 20%** of risk scores. ROC-AUC is reported too. |
| Data | `data/raw/cell2celltrain.csv` only. 51,047 rows and 58 columns (confirmed in Stage 2). |
| Split | Train 40,837 rows (28.82% churn), test 10,210 rows (28.81% churn), in `data/interim/`. Test set locked until Stage 12. |
| Timeline | 3 weeks. EDA report at the end of week 1, Streamlit demo at the end of week 3. |
| Project type | Prototype / portfolio scenario. Not a production system. |

## Deliverables (from the brief)

- [ ] EDA notebook with 3 to 5 plain-English business insights
- [ ] Reusable scikit-learn pipeline (missing values, encoding, scaling) with no leakage
- [ ] Model comparison: logistic regression baseline plus at least 2 stronger models
- [ ] Tuned final model, with a threshold that fits the 20% call budget
- [ ] SHAP explanations: overall and for one customer
- [ ] Business impact estimate (monthly revenue saved)
- [ ] Streamlit demo app (single customer input and CSV upload)
- [ ] Clean GitHub repo, `requirements.txt`, README

## Decisions

| # | Decision | Status |
|---|---|---|
| A1 | Use the `Churn` label as given: churn 31 to 60 days after the snapshot. | Decided |
| A2 | `RetentionCalls`, `RetentionOffersAccepted` and `MadeCallToRetentionTeam` are possible leakage. Check them in EDA. If we cannot show they were known at snapshot time, leave them out. | Decided in A8: **excluded** |
| A3 | Success rule: within the 20% call budget, the final model must catch clearly more churners than the logistic regression baseline and far more than random calling. The exact target number is set right after the baseline run, before looking at stronger models. | Rule decided, number pending |
| A4 | Build the model **without** `AgeHH1`, `AgeHH2`, `ChildrenInHH`, `MaritalStatus`, `Occupation` and `IncomeGroup` first. Keep them for a fairness check. Then build **with** them, compare on cross-validation (never the test set), and choose. `CreditRating` and `PrizmCode` stay in; `PrizmCode` gets a fairness check. | Decided |
| A5 | Baselines: (1) random calling, (2) rule "call the 20% with the most `CustomerCareCalls`", (3) logistic regression. | Decided |
| A6 | Split before EDA: 80/20, stratified on `Churn`, `random_state=42`. EDA and modelling use the training set only. | Done |
| A7 | Rule-based cleaning (`src/churn/cleaning.py`), run as the first pipeline step: negatives → missing, `HandsetPrice` → number, combined `CellphoneUserStatus`, `AgeHH1`/`IncomeGroup` 0 → missing. | Done |
| A8 | Feature set: **35 features** (29 numeric, 5 categorical, 1 ordinal) in `src/churn/features.py`. Excluded: retention columns (leakage risk), 6 personal + 10 lifestyle columns (for the later comparison, A4), `DroppedBlockedCalls` and `HandsetModels` (redundant), `ServiceArea` (replaced by `CityCode`). New: `CityCode`, `ChangedPhone`. | Done |
| A9 | Pipeline (`src/churn/pipeline.py`): clean → add features → median fill + "was missing" flags → one-hot (rare/unseen → "infrequent") → `CreditRating` ranks → scale only for logistic regression. 35 features → 93 columns. | Done |

## Stage 2 findings (data inspection)

Source: `notebooks/01_data_inspection.ipynb`.

**Structure (good)**
- 51,047 rows × 58 columns. Rows loaded match lines in the file, so no broken rows.
- No duplicate rows and no repeated `CustomerID`: one row = one customer.
- Churn: 14,711 Yes (28.82%) and 36,336 No (71.18%). Matches the brief and the documentation.
- Column types: 26 float, 9 int, 23 text (pandas 3 `str`).
- Call counts have decimals because they are monthly averages (the documentation says averages over 4 months).

**Issues to handle later (Stage 4 cleaning / Stage 7 pipeline)**
- Missing values: 156 rows missing 6 usage columns (`MonthlyRevenue`, `MonthlyMinutes`, `TotalRecurringCharge`, `DirectorAssistedCalls`, `OverageMinutes`, `RoamingCalls`); 367 missing `PercChangeMinutes` and `PercChangeRevenues`; 909 missing `AgeHH1` and `AgeHH2`; 24 missing `ServiceArea`; 1 missing `Handsets`, `HandsetModels`, `CurrentEquipmentDays`.
- Hidden missing values: `HandsetPrice` 57% `Unknown`; `MaritalStatus` 39% `Unknown`; `Homeownership` is only `Known`/`Unknown`; many 0 values in `AgeHH1`, `AgeHH2` and `IncomeGroup` (probably unknown); `Occupation` 74% `Other`.
- Impossible values: negative minimums in `MonthlyRevenue` (-6.17), `TotalRecurringCharge` (-11) and `CurrentEquipmentDays` (-5).
- `HandsetPrice` is stored as text because of the `Unknown` values.
- Very large maximums in `UniqueSubs` (196) and `ActiveSubs` (53). Possibly business accounts: investigate, do not delete.
- `NewCellphoneUser` and `NotNewCellphoneUser` overlap (both often "No"): really one 3-way column.
- `ServiceArea` has 747 categories: needs a careful encoding choice.
- `CreditRating` is ordered (1-Highest to 7-Lowest).
- `MadeCallToRetentionTeam` = Yes for 1,745 customers (see decision A2).

## Stage 3 findings (EDA, training set)

Source: `notebooks/02_eda.ipynb`. Full detail in the notebook's C4 and "EDA summary" cells.

**Business insights (draft for the manager)**
1. **1-year danger zone:** 11-12 months in service churn 42.5% vs 15.9% at 6-8 months. Same with or without a phone change (timing, likely contract end).
2. **Long-standing customers on their original phone:** 19+ months, same phone 33.8% vs changed phone 25.9%. Non-web phones 37.2% vs 27.9%.
3. **Low or falling usage:** lowest minutes 37.8% vs heaviest 25.0%. Biggest drop in minutes 34.3%. Missing usage data 46-57%.
4. **Customer care calls are not a churn signal alone:** most calls 24.5% vs almost none 30.5%. Our rule baseline (A5) will probably do worse than random. Kept as decided.
5. **Churn differs by city:** 22.0% (STL) to 41.7% (SLC) among cities with 100+ customers.

**Data findings**
- Missing usage data is a strong churn signal (Part A). "Unknown" values are weak signals (+1 to +1.5 points).
- Retention columns: strong link (about 45% churn) but not near 100%. Timing still unconfirmed (A2, decide in Stage 5).
- Honest ranking of real values (spread without "Missing"): `MonthsInService` 26.6, `CurrentEquipmentDays` 19.3, `MonthlyMinutes` 12.8, `PercChangeMinutes` 9.4, `CreditRating` 9.3, `HandsetWebCapable` 9.3.
- Overlaps: `MadeCallToRetentionTeam` = `RetentionCalls` > 0 exactly. `DroppedBlockedCalls` = Dropped + Blocked in 88.9% of rows. `Handsets`/`HandsetModels` correlation 0.91. `NewCellphoneUser`/`NotNewCellphoneUser` = one 3-way column.
- `ServiceArea`: 730 areas in train, only 112 with 100+ customers. 58 city codes (first 3 letters).
- Proposed actions for Stages 4, 5 and 7 are listed in the notebook's "EDA summary" cell.

## Assumptions to keep in mind

- The 31 to 60 day churn window comes from third-party copies of the original Teradata/Duke documentation. Our file has 51,047 rows; the documentation says 71,047. The gap is exactly 20,000, and the documentation says its validation sample has no churn label. So our file is very likely the labelled part. Strong guess, not confirmed.
- All feature values were recorded at or before the snapshot.
- Each row is a different customer, recorded once (confirmed in Stage 2).
- We do not know whether churn means the customer cancelled or was cut off for not paying.
- The customers in the data represent today's customers. The data is old, so this may not hold.

## Known risks

- Churn window not 100% confirmed for this exact file.
- Possible leakage from the retention-call columns (A2).
- Possible bias through `PrizmCode` and other proxy columns.
- The data may be out of date compared with today's customers.
- **No customers under 6 months** (`MonthsInService` is 6 to 61). The model must not be used for new customers.
- "Unknown" values may themselves be linked to churn. Check in EDA before deciding how to handle them.

## Completed work

- Stage 1: problem defined, decisions A1 to A5 recorded in `docs/decisions.md`.
- Environment: `.venv` (Python 3.13.15) with pinned packages in `requirements.txt`.
- Stage 2: data inspected in `notebooks/01_data_inspection.ipynb`. Findings above.
- Stage 3 setup:
  - `pyproject.toml` with the package installed in editable mode (`pip install -e .`).
  - `src/churn/` package: `config.py` (paths and settings), `data.py` (loading and checks), `split.py` (train/test split).
  - 22 tests passing (`tests/test_data.py`, `tests/test_split.py`).
  - Split run with `python -m churn.split` (decision A6).
  - Notebook 01 uses the shared loader.
  - Learning comments added to all code (to be cleaned up at the end of the project).
- Stage 3 EDA: Parts A (data quality, leakage), B (churn by column, charts), C (overlaps, areas, insights).
- `docs/data_dictionary.md`: all 58 columns (5 confirmed by sources, the rest inferred and marked).
- Stage 4: `src/churn/cleaning.py` with 4 rules (decision A7) and 15 tests. 37 tests passing in total.
- Stage 5: `src/churn/features.py` (feature lists, `add_features`, `split_features_target`, decision A8) and 14 tests. 51 tests passing in total.
- Stage 7: `src/churn/pipeline.py` (decision A9) and 15 tests. 66 tests passing in total. `encode_target()` added to `features.py`.
- Branches: `feat/stage-4-cleaning` renamed to `feat/stage-4-5-cleaning-features` (Stage 5 was committed there). Stage 7 on `feat/stage-7-pipeline`.
- `DAYS_PER_MONTH` changed from 30.4 to 365.25 / 12. 198 customers moved category in the EDA C3 table; conclusions unchanged, numbers updated.
