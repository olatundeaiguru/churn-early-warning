# Experiment Log

One entry per meaningful experiment. All results are on the **training set** with cross-validation unless stated. The test set stays locked until Stage 12.

---

## EXP-001: Baselines

- **Date:** 2026-11-26
- **Stage:** 8 (Baseline)
- **Git branch:** `feat/stage-8-baseline`
- **Command:** `python -m churn.baselines` (about 18 seconds)
- **Data:** `data/interim/train.csv`, 40,837 customers, 28.82% churn
- **Validation:** 5-fold stratified cross-validation, `random_state=42`, same folds for every model
- **Pipeline:** decision A9 (35 features → 93 columns)
- **Measures:** recall and precision at 20% call budget (ties broken randomly, fixed seed), ROC-AUC, PR-AUC
- **Results file:** `reports/baseline_results.csv`

| Model | Recall at 20% | Precision at 20% | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Random calling (`DummyClassifier(strategy="prior")`) | 0.197 ± 0.003 | 0.284 ± 0.004 | 0.500 ± 0.000 | 0.288 ± 0.000 |
| Customer-care rule (most care calls first) | 0.171 ± 0.004 | 0.246 ± 0.005 | 0.470 ± 0.002 | 0.273 ± 0.002 |
| Logistic regression (default, `max_iter=1000`, scaled) | **0.278 ± 0.011** | **0.400 ± 0.016** | **0.611 ± 0.010** | **0.377 ± 0.013** |

(± = standard deviation across the 5 folds)

- **Findings:**
  - Random calling catches ~20% of churners, as expected: the measure works.
  - The customer-care rule is **worse than random**, as EDA predicted (heavy care callers churn less).
  - Logistic regression reaches ~40% more churners than random for the same calls.
  - ROC-AUC 0.61 is modest: churn is hard to predict from this data (retention columns excluded on purpose).
- **Warnings:** none.
- **Notes:** the cleaning step's row-count logs were moved from INFO to DEBUG after this run (they repeated on every pipeline run).
- **Decision taken from this experiment:** success target for stronger models = recall at 20% ≥ 0.31 and ROC-AUC ≥ 0.64 (A10).

---

## EXP-002: Candidate models

- **Date:** 2026-11-27
- **Stage:** 9 (Train and compare candidates)
- **Git branch:** `feat/stage-9-candidates`
- **Command:** `python -m churn.candidates` (run by the owner)
- **Data:** `data/interim/train.csv`, 40,837 customers, 28.82% churn
- **Validation:** same 5-fold stratified CV as EXP-001 (`random_state=42`), with training-fold scores recorded too (overfitting check)
- **Pipeline:** A9 (35 features). Scaling only for logistic regression.
- **Models:** logistic regression (the EXP-001 baseline, reused unchanged); `RandomForestClassifier(random_state=42, n_jobs=-1)`; `LGBMClassifier(random_state=42, verbose=-1)`. All other settings default.
- **Results file:** `reports/candidate_results.csv`

| Model | Recall at 20% | Precision at 20% | ROC-AUC | PR-AUC | Train recall / ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.278 ± 0.011 | 0.400 ± 0.016 | 0.611 ± 0.010 | 0.377 ± 0.013 | 0.283 / 0.620 |
| Random Forest | 0.314 ± 0.008 | 0.453 ± 0.011 | 0.657 ± 0.009 | 0.429 ± 0.009 | 0.694 / 1.000 |
| **LightGBM** | **0.332 ± 0.006** | **0.478 ± 0.009** | **0.674 ± 0.008** | **0.456 ± 0.011** | 0.471 / 0.808 |

- **Findings:**
  - Both tree models pass the A10 target. LightGBM is best on every measure and the most stable.
  - Logistic regression matches EXP-001 exactly (same code path, after moving the CV code into `cross_validate_models`).
  - Random Forest memorises the training folds (train ROC-AUC 1.000); LightGBM has a smaller gap.
- **Warnings:** none.
- **Decision taken from this experiment:** shortlist LightGBM (A11).

---

## EXP-003: Personal and lifestyle columns (A4 check)

- **Date:** 2026-11-27
- **Stage:** 9
- **Git branch:** `feat/stage-9-candidates`
- **Command:** `python -m churn.personal_check` (run by the owner)
- **Data and validation:** as EXP-002
- **Model:** the shortlisted LightGBM (`build_lightgbm_pipeline`), unchanged
- **Versions:** main 35 features; + 6 personal; + 10 lifestyle; + both (16)
- **Rule (set before the run):** keep a group only if recall at 20% gains ≥ 0.02
- **Results file:** `reports/personal_check_results.csv`

| Version | Recall at 20% | ROC-AUC | Gain | Passes |
|---|---|---|---|---|
| Main features | 0.332 ± 0.006 | 0.674 | — | — |
| + personal | 0.337 ± 0.009 | 0.678 | +0.005 | No |
| + lifestyle | 0.331 ± 0.006 | 0.675 | −0.001 | No |
| + both | 0.333 ± 0.005 | 0.677 | +0.001 | No |

- **Findings:** no group helps beyond noise. "Main features" reproduces EXP-002's LightGBM score exactly (0.3321).
- **Warnings:** none.
- **Decision taken from this experiment:** personal and lifestyle columns stay out (A12).
