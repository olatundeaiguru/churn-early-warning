# Project Status: Churn Early-Warning System

_Last updated: 2026-11-20_

## Current stage

**Stage 1: Define the problem (complete).**
Next stage: **Stage 2: Understand and load the data** (not started).

## Next single step

Start Stage 2: move `cell2celltrain.csv` into `data/raw/` (with approval), then inspect its structure through code.

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
| Data | `cell2celltrain.csv` only. About 51,047 rows and 58 columns (row count to be confirmed in Stage 2). |
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

## Assumptions to keep in mind

- The 31 to 60 day churn window comes from third-party copies of the original Teradata/Duke documentation. Our file (about 51,047 rows) differs from the documented size (71,047), so it is probably a re-packaged version.
- All feature values were recorded at or before the snapshot.
- Each row is a different customer, recorded once.
- We do not know whether churn means the customer cancelled or was cut off for not paying.
- The customers in the data represent today's customers. The data is old, so this may not hold.

## Known risks

- Churn window not 100% confirmed for this exact file.
- Possible leakage from the retention-call columns (A2).
- Possible bias through `PrizmCode` and other proxy columns.
- The data may be out of date compared with today's customers.

## Completed work

- Stage 1: problem defined, decisions A1 to A5 recorded in `docs/decisions.md`.
