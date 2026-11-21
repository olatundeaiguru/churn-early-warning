# Decision Log

Each decision records: what we decided, why, the evidence, other options we considered, and when to look at it again.

---

## A1: Use the `Churn` label as given

- **Date:** 2026-11-20
- **Stage:** 1 (Define the problem)
- **Decision:** Use the `Churn` column as the target. `Churn` = Yes means the customer left **31 to 60 days after** their data was recorded.
- **Why:** It is the only label we have, and the brief says to use `cell2celltrain.csv` only.
- **Evidence:**
  - Third-party copies of the original Teradata/Duke documentation define churn as "churn between 31-60 days after obs_date", with a mean of 0.29.
  - The brief also says about 29% of customers churn.
  - The CSV has no date columns, so the window cannot be checked from the data itself.
- **Limits:**
  - The documented dataset has 71,047 records. Our file has about 51,047 rows, so it is probably a re-packaged version.
  - We do not know if churn means "cancelled" or "cut off for not paying".
  - We cannot check the labels for errors.
- **What it means for the model:** the model predicts "likely to leave in about 1 to 2 months", not "will leave at some point".
- **Alternatives:** None realistic. There is no other label source.
- **Revisit if:** we find the original documentation for this exact file, or the EDA shows a churn rate far from 29%.

---

## A2: Retention-call columns are possible leakage

- **Date:** 2026-11-20
- **Stage:** 1 (flagged), final decision in Stage 5
- **Columns:** `RetentionCalls`, `RetentionOffersAccepted`, `MadeCallToRetentionTeam`
- **Decision:** Flag these as possible leakage. Check them in EDA. If we cannot show they were known at the snapshot, **leave them out**.
- **Why:** A customer may call the retention team *because* they are already about to leave. If that call happened after the snapshot, the column gives away the answer. The model would look great in testing but fail in real use.
- **Evidence to collect in EDA:**
  - How strongly each column relates to churn. A near-perfect link is a warning sign.
  - Whatever the documentation says about when these values were recorded.
- **Alternatives:**
  - Keep them in without checking. Rejected, because of the leakage risk.
  - Drop them right away. Possible, but we would rather decide with evidence.
- **Trade-off:** If we leave them out, the model may score a little lower, but the score will be honest.
- **Revisit:** Stage 5 (Separate features and target).

---

## A3: How we define success

- **Date:** 2026-11-20
- **Stage:** 1 (rule), Stage 8 (number)
- **Decision:** Within the 20% call budget, the final model must catch **clearly more churners** than the logistic regression baseline, and far more than random calling. Random calling catches about 20% of churners.
- **When the number is set:** right after the baseline run, and **before** we look at the stronger models.
- **Why:** If we set the target after seeing all the results, we could move the goalposts to make any model look good.
- **Why "within 20%":** The team can only call 20% of customers. A model that is good overall but weak in its top 20% does not help them.
- **Alternatives:** Use only ROC-AUC or only recall. Rejected as the main measure:
  - ROC-AUC looks at all customers, not just the top 20% the team will call.
  - Recall alone can be pushed to 100% by flagging everyone, which the budget does not allow.
  - Both are still reported, as the brief asks.
- **Revisit:** Stage 8 (Baseline), to set the number.

---

## A4: Personal demographic columns

- **Date:** 2026-11-20
- **Stage:** 1
- **Columns:** `AgeHH1`, `AgeHH2`, `ChildrenInHH`, `MaritalStatus`, `Occupation`, `IncomeGroup`
- **Decision:**
  1. Build the model **without** these columns first. Keep them aside for a fairness check: does the model work equally well for different groups?
  2. Then build the same model **with** them.
  3. Compare both using cross-validation on the training data only, **never the test set**.
  4. Choose which version to keep.
- **Why:**
  - Data minimisation: only use personal data you really need.
  - Choosing who to call based on age or family situation is hard to defend.
  - Comparing both versions shows how much these columns really add, so the final choice is based on evidence.
- **Other columns:**
  - `CreditRating` and `PrizmCode` stay in. They count as business and segment data.
  - `PrizmCode` (a neighbourhood lifestyle segment) may hide income or ethnicity information, so it also gets the fairness check.
- **Alternatives:**
  - Include all columns from the start. Rejected as the first step: more ethical risk, harder to explain.
  - Never use them. Rejected: we want evidence before deciding.
- **Trade-off:** The model without demographics may be slightly less accurate.
- **Revisit:** after the model without demographics is built (around Stage 9 or 10).

---

## A5: Baselines

- **Date:** 2026-11-20
- **Stage:** 1 (chosen), Stage 8 (run)
- **Decision:** Compare every model against three baselines:
  1. **Random calling:** call a random 20% of customers. This catches about 20% of churners.
  2. **Simple rule:** call the 20% of customers with the most `CustomerCareCalls`.
  3. **Logistic regression:** the baseline the brief asks for.
- **Why:**
  - If a simple rule works almost as well as a model, the company may not need the model.
  - It answers the brief's question: "Does the model beat the baseline, and why?"
- **Why this rule:** It was chosen **before** looking at the data, so we could not pick a rule that just happens to look good.
- **Alternatives:** Only logistic regression, as in the brief. We kept it, and added the rule and random calling for a fairer picture.
- **Revisit:** Stage 8 (Baseline).

---

## A6: Split the data before EDA

- **Date:** 2026-11-21
- **Stage:** 3 (done before EDA, instead of in Stage 6)
- **Decision:** Hold out the test set **before** EDA. All EDA, cleaning decisions and modelling use the training set only.
- **Settings:**
  - 80% train / 20% test
  - Stratified on `Churn`
  - `random_state=42`
  - Saved as CSV in `data/interim/` (not committed to git)
- **How:** `python -m churn.split` (code in `src/churn/split.py`). The script refuses to overwrite existing files, so the test set cannot change by accident.
- **Result (actual run):**

  | Set | Rows | Churn rate |
  |---|---|---|
  | Raw | 51,047 | 28.82% |
  | Train | 40,837 | 28.82% |
  | Test | 10,210 | 28.81% |

- **Why:**
  - What we learn in EDA shapes our cleaning and feature choices.
  - If the test customers were included, that knowledge would leak into our decisions ("human leakage").
  - Splitting first keeps the test set truly untouched, as the brief requires.
- **Why these settings:**
  - Stratified, so both sets keep the same churn rate.
  - 20% gives about 2,940 churners in the test set, enough for a reliable final check.
  - The seed is fixed, so the split can be reproduced.
- **Alternatives:**
  - EDA on all the data, splitting in Stage 6. Simpler and common, but with a small leakage risk. Rejected.
  - A time-based split. Better for real deployment, but **not possible**: the data has no dates.
- **Limitation:** A random split cannot show how the model handles behaviour changing over time. The test score may be a little optimistic compared with real use on future customers.
- **Revisit if:** the dataset changes, or a dataset with dates becomes available.
