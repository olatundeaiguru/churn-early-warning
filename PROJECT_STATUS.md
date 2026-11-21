# Project Status: Churn Early-Warning System

_Last updated: 2026-11-21_

## Current stage

**Stage 2: Understand and load the data (complete).**
Next stage: **Stage 3: Exploratory Data Analysis** (not started).

## Next single step

Walk through the code in `notebooks/01_data_inspection.ipynb` section by section, so the owner understands it. Then start Stage 3.

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
| A2 | `RetentionCalls`, `RetentionOffersAccepted` and `MadeCallToRetentionTeam` are possible leakage. Check them in EDA. If we cannot show they were known at snapshot time, leave them out. | Decide in Stage 5 |
| A3 | Success rule: within the 20% call budget, the final model must catch clearly more churners than the logistic regression baseline and far more than random calling. The exact target number is set right after the baseline run, before looking at stronger models. | Rule decided, number pending |
| A4 | Build the model **without** `AgeHH1`, `AgeHH2`, `ChildrenInHH`, `MaritalStatus`, `Occupation` and `IncomeGroup` first. Keep them for a fairness check. Then build **with** them, compare on cross-validation (never the test set), and choose. `CreditRating` and `PrizmCode` stay in; `PrizmCode` gets a fairness check. | Decided |
| A5 | Baselines: (1) random calling, (2) rule "call the 20% with the most `CustomerCareCalls`", (3) logistic regression. | Decided |

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
