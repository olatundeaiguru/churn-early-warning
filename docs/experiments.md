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
