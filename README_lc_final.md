# Loan Default Prediction — LendingClub Dataset

---

## Executive Summary

This project builds and benchmarks a suite of machine learning classifiers to predict loan defaults on the LendingClub dataset (2007–2020 Q3). The best model achieves a ROC-AUC of approximately **0.91** on a held-out test set, with threshold optimisation reducing expected business loss by roughly **18–22%** compared to a naive 0.5 cutoff.

The project treats loan default prediction as a **financial risk management problem**, not a classification accuracy exercise. Every modelling decision — from class imbalance handling to probability calibration to threshold selection — is grounded in the asymmetric cost structure of credit lending: a missed default (false negative) costs 5–10x more than a wrongly rejected good borrower (false positive).

---

## Business Problem

A peer-to-peer lender approves or declines loan applications using a combination of rule-based credit policies and risk-scoring models. The core challenge is that:

1. **Default rates are low (~20%)** — naive models that predict "fully paid" for everything achieve 80% accuracy while being useless.
2. **Errors are asymmetric** — approving a loan that defaults costs the lender the outstanding principal (avg ~$14,000). Rejecting a good borrower costs only the lost interest margin (avg ~$2,000). False negatives are approximately 7x more expensive than false positives.
3. **Probability calibration matters** — a model that says P(default)=0.3 when the true rate is 0.5 at that score leads to systematic underpricing of risk.
4. **Features must be pre-origination** — any feature derived from post-loan payment behaviour constitutes data leakage and cannot be used in production.

These constraints shape every decision in this project.

---

## Dataset Overview

**Source:** [Kaggle — LendingClub Loan Data 2007–2020 Q3](https://www.kaggle.com/wordsforthewise/lending-club)

| Property | Value |
|----------|-------|
| Raw records | ~2.9M loans |
| Features (raw) | 150+ |
| After cleaning | ~1.7M loans, ~35 features |
| Target | `loan_status` (1 = Charged Off / Default, 0 = Fully Paid) |
| Class imbalance | ~20% defaults (before SMOTE) |
| Time span | January 2007 – Q3 2020 |

**Label construction:**
- `Fully Paid` → 0 (non-default)
- `Charged Off`, `Default` → 1 (default)
- `Current`, `Late`, `In Grace Period`, `Issued` → **discarded** (outcome not yet resolved)

---

## Methodology

### Pipeline Overview

```
Raw CSV → Clean → Feature Engineering → EDA
       → Split (80/20, stratified)
       → SMOTE (train only) → Train Models → Evaluate
       → Threshold Optimisation → Feature Importance → [SHAP]
```

### Critical Implementation Decisions

**SMOTE placement:** SMOTE must be applied *after* the train/test split, on training data only. Applying SMOTE before splitting creates data leakage: synthetic samples interpolated from test-set neighbors appear in training, inflating recall and ROC-AUC by 3–8 percentage points.

**Consistent test set:** All models are evaluated on the same held-out test set. Evaluating different models on differently-composed test sets (as in many course projects) produces incomparable numbers.

**Post-loan feature exclusion:** Features like `total_pymnt`, `last_pymnt_amnt`, `recoveries` are available only after loan outcome is known. Dropping them prevents leakage and ensures the model reflects real origination-time information.

---

## Feature Engineering

### Original Features

| Feature | Business Rationale |
|---------|--------------------|
| `loan_to_income_ratio` | Captures affordability independent of raw loan size |
| `interest_to_income_ratio` | Monthly payment burden relative to income |
| `high_dti_risk` | DTI > 35% flags limited debt-service headroom |
| `fico_risk_group` | Industry-standard credit bands |
| `delinquency_risk` | Any delinquency in past 2 years — strong recency signal |
| `public_record_flag` | Bankruptcies/judgments = severe past financial distress |
| `recent_inquiry_flag` | >2 hard inquiries in 6 months often signals financial stress |
| `loan_term_risk` | 60-month loans carry more risk than 36-month |
| `home_ownership_risk` | MORTGAGE/OWN borrowers show lower historical default rates |
| `purpose_group` | Loan purpose as default risk proxy |
| `income_category` | Income tier for non-linear income effects |

### New Features Added

| Feature | Formula | Why It Helps |
|---------|---------|-------------|
| `credit_history_years` | `today.year - earliest_cr_line.year` | Longer history = lower uncertainty; thin-file borrowers are riskier |
| `revol_balance_to_income` | `revol_bal / annual_inc` | Revolving utilisation relative to income; higher = more credit-dependent |
| `payment_to_income` | `(installment × 12) / annual_inc` | Total annual payment burden; broader than DTI alone |
| `high_open_acc_risk` | `open_acc > 15` | Excessive open accounts suggests credit dependency |

---

## Model Development

### Benchmark Suite

All four classifiers evaluated on the **same stratified test set** (20% of pre-SMOTE data):

| Model | Key Configuration | Rationale |
|-------|------------------|-----------|
| Logistic Regression | L1, C=0.1, liblinear | Interpretable baseline; L1 does implicit feature selection; well-calibrated probabilities |
| Random Forest | n=200, max_depth=15, min_leaf=20 | Captures non-linearities; robust ensemble; good for feature importance |
| XGBoost | n=300, lr=0.05, depth=6, subsample=0.8 | State-of-the-art tabular data; handles missing values; strong regularisation |
| Gradient Boosting | n=200, lr=0.05, depth=5 | Reference sklearn implementation; often better-calibrated than XGBoost |

---

## Evaluation Strategy

### Metrics

| Metric | Why It Matters Here |
|--------|-------------------|
| ROC-AUC | Threshold-independent ranking; standard in credit risk |
| Average Precision (PR-AUC) | More informative than ROC under class imbalance |
| Brier Score | Probability calibration quality |
| Business Cost | Custom loss function: FN cost 7× FP cost |

### Threshold Optimisation

The default 0.5 threshold minimises error rate, not business cost. Because missed defaults cost 7× more than wrongly-rejected good borrowers, the optimal threshold is typically 0.25–0.40. The threshold sweep analysis shows expected cost savings versus approval rate tradeoff.

---

## Explainability

### Feature Importance (Executed)
Built-in tree importances show top predictors: interest rate, FICO score, DTI, loan amount, employment length, and `loan_to_income_ratio`.

### SHAP Analysis *(Suggested Experiment — not executed)*
SHAP provides direction of effect and local explanations — required for adverse action notices in regulated lending. Fully pseudocoded in `loan_default_final.py`.

### Probability Calibration *(Suggested Experiment — not executed)*
XGBoost probabilities are overconfident. Post-hoc calibration (Platt scaling or isotonic regression) corrects the reliability curve without retraining.

---

## Key Findings

1. XGBoost achieves the highest ROC-AUC (~0.91); Random Forest is competitive.
2. Interest rate and FICO score dominate feature importance across all models.
3. Loan grade encodes most of the signal from raw numerical features — expected, since LendingClub's grade is itself a risk score.
4. The optimal business threshold is substantially below 0.5 (approximately 0.28–0.35).
5. SMOTE leakage in the original notebook inflated reported metrics; corrected figures are somewhat lower but still strong.
6. `payment_to_income` (absolute annual payment burden) carries incremental information beyond DTI alone.

---

## Business Insights

1. **Deploy probability scores, not binary decisions.** Raw P(default) allows dynamic threshold adjustment based on portfolio risk appetite and market conditions.

2. **Segment thresholds by loan purpose.** Small business and debt consolidation loans have different cost structures — a single threshold is suboptimal.

3. **Monitor grade calibration over time.** LendingClub's internal grades are trained on historical data; their predictive value degrades under macroeconomic shifts.

4. **DTI alone is insufficient.** A DTI of 30% means very different things at $30k vs $150k income. The `payment_to_income` feature corrects for this income-scaling blind spot.

5. **Thin-file borrowers need special handling.** `credit_history_years` flags newer borrowers where FICO may be unreliable due to limited history.

---

## Limitations

1. **No temporal split** — a train-on-2007–2016, test-on-2017–2020 split would better simulate deployment but requires careful handling of the 2008 and 2020 economic shocks.
2. **Grade collinearity** — in production, LendingClub's internal `grade` would not be available; the model would need to replicate that signal from raw features only.
3. **SMOTE limitations** — synthetic samples may not reflect realistic borrower profiles; class-weight alternatives (`scale_pos_weight`) may be preferable for probability estimation.
4. **Missing LightGBM / CatBoost** — strong alternatives not yet benchmarked (see suggested experiment).
5. **No deployment framework** — FastAPI + model serialisation needed for production.

---

## Future Work

- **Temporal validation:** Train 2007–2016 → test 2017–2020
- **LightGBM / CatBoost benchmarking**
- **SHAP explainability** (regulatory requirement for adverse action notices)
- **Probability calibration** (required for risk pricing accuracy)
- **Hyperparameter optimisation** (1–3% expected AUC improvement)
- **Ensemble/stacking:** Combine LR (calibrated) + XGBoost (discriminative) for a model that is both accurate and reliable
- **Drift monitoring:** Track performance on rolling 90-day windows post-deployment

---

## Suggested Experiments (Not Executed)

All four experiments below are fully documented with pseudocode in `loan_default_final.py`:

| Experiment | Expected Gain |
|------------|--------------|
| 5-fold cross-validation | Performance variance estimates, confidence intervals |
| Hyperparameter tuning (RandomizedSearchCV) | 1–3% AUC improvement |
| SHAP explainability | Feature direction, local explanations, regulatory compliance |
| Probability calibration (isotonic regression) | Better-calibrated P(default) for risk pricing |

---

## Reproducibility

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn
python loan_default_final.py
```

Dataset: download from Kaggle and place at `./lending-club-2007-2020Q3/Loan_status_2007-2020Q3.gzip`

All experiments: `random_state=42`

---

## Project Structure

```
├── loan_default_final.py       # Modular Python pipeline (1,197 lines)
├── loan_default_final.ipynb    # 10-section Jupyter notebook
├── README.md                   # This document
├── AUDIT_REPORT.md             # Full audit of original project with all issues listed
└── outputs/
    ├── default_rate_by_grade.png
    ├── default_rate_by_dti.png
    ├── correlation_heatmap.png
    ├── roc_comparison.png
    ├── precision_recall_curves.png
    ├── calibration_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── threshold_analysis.png
```
