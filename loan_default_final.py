"""
loan_default_final.py
======================
Binary loan default prediction on the LendingClub dataset.

Pipeline:
    Load → Clean → Feature Engineering → EDA → Split → SMOTE (train only) →
    Benchmark Models → Evaluate → Visualise → Explain → Business Analysis

Business context:
    Predicting loan defaults is an asymmetric cost problem. A false negative
    (approving a loan that defaults) typically costs 5–10x more than a false
    positive (rejecting a borrower who would have repaid). All threshold
    decisions in this project are made with this asymmetry in mind.

Dataset:
    LendingClub loan data 2007–2020 Q3
    https://www.kaggle.com/wordsforthewise/lending-club

Critical implementation note — SMOTE ordering:
    SMOTE must be applied AFTER the train/test split, on training data only.
    Applying SMOTE before splitting creates data leakage: synthetic points
    interpolated from test-set neighbors appear in training, inflating recall
    and ROC-AUC by several percentage points on the official leaderboard.
"""

# ── Standard library ────────────────────────────────────────────────────────
import warnings
import itertools

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import skew

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Business cost constants ──────────────────────────────────────────────────
# These reflect realistic asymmetric costs in consumer lending:
#   - A missed default (FN) costs the lender the outstanding principal (~$14k avg)
#   - A wrongly rejected good borrower (FP) costs the lender lost interest (~$2k avg)
# Ratio: FN is approximately 7x more costly than FP
COST_FALSE_NEGATIVE = 7.0   # relative cost of missing a default
COST_FALSE_POSITIVE = 1.0   # relative cost of rejecting a good borrower

PALETTE = sns.color_palette("Set2")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the LendingClub raw CSV.

    Args:
        filepath: Path to the raw CSV/gzip file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def drop_ambiguous_statuses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove loans with ambiguous repayment status.

    'Current', 'Late', 'In Grace Period', 'Issued' loans have not yet
    resolved — we cannot label them as default or fully paid. Including them
    would introduce label noise.
    """
    ambiguous = ['Current', 'Late (31-120 days)', 'In Grace Period',
                 'Late (16-30 days)', 'Issued']
    df = df[~df['loan_status'].isin(ambiguous)].copy()

    # Consolidate remaining statuses to binary label
    status_map = {
        'Fully Paid': 0,
        'Does not meet the credit policy. Status:Fully Paid': 0,
        'Charged Off': 1,
        'Does not meet the credit policy. Status:Charged Off': 1,
        'Default': 1,
    }
    df['loan_status'] = df['loan_status'].map(status_map)
    df = df[df['loan_status'].notna()].copy()
    df['loan_status'] = df['loan_status'].astype(int)

    default_rate = df['loan_status'].mean()
    print(f"After status filter: {len(df):,} loans | Default rate: {default_rate:.2%}")
    return df


def drop_post_loan_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that would not be available at loan origination time.

    These are 'post-outcome' columns: payment totals, last payment dates,
    outstanding principal. Including them would constitute data leakage —
    the model would be using information from *after* the default event to
    predict the default event.
    """
    leakage_cols = [
        'pymnt_plan', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_d',
        'last_pymnt_amnt', 'out_prncp', 'out_prncp_inv', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'next_pymnt_d', 'last_credit_pull_d',
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} post-loan leakage columns")
    return df


def drop_high_missingness_columns(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Drop columns with more than `threshold` fraction of missing values."""
    pct_missing = df.isna().mean()
    high_miss = pct_missing[pct_missing > threshold].index.tolist()
    df = df.drop(columns=high_miss)
    print(f"Dropped {len(high_miss)} columns with >{threshold:.0%} missing values")
    return df


def regression_imputer(df: pd.DataFrame, col: str, n_features: int = 10) -> pd.DataFrame:
    """
    Impute missing values in `col` using a linear regression trained on the
    `n_features` most correlated numerical columns.

    This is superior to mean/median imputation for columns like DTI where
    correlated predictors (loan amount, installment, annual income) can
    explain much of the variance in the missing value.

    Args:
        df:         DataFrame containing the column to impute.
        col:        Column name to impute.
        n_features: Number of correlated features to use as predictors.

    Returns:
        DataFrame with `col` imputed.
    """
    num_df = df.select_dtypes(include='number')
    corr   = num_df.corr()[col].abs()
    top_features = corr.nlargest(n_features + 1).index.tolist()
    if col in top_features:
        top_features.remove(col)

    not_null = df[df[col].notnull()]
    is_null  = df[df[col].isna()]

    X_train = not_null[top_features].fillna(not_null[top_features].median())
    y_train = not_null[col]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    if len(is_null) > 0:
        X_pred = is_null[top_features].fillna(not_null[top_features].median())
        X_pred_sc = scaler.transform(X_pred)
        df.loc[df[col].isna(), col] = model.predict(X_pred_sc)

    print(f"  Imputed {len(is_null):,} missing values in '{col}' via regression")
    return df


def clean_type_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string-encoded numerical columns to proper numeric types.
    Handles employment length, term, interest rate, and revolving utilisation.
    """
    # Employment length
    length_map = {
        '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7,
        '6 years': 6,  '5 years': 5, '4 years': 4, '3 years': 3,
        '2 years': 2,  '1 year': 1, '< 1 year': 0.5, 'n/a': 0
    }
    if 'emp_length' in df.columns:
        df['emp_length_int'] = df['emp_length'].map(length_map)

    # Term (36 or 60 months)
    if 'term' in df.columns:
        df['term_numeric'] = df['term'].str.replace(' months', '', regex=False).astype(float)

    # Remove % signs from rate columns
    for col, new_col in [('revol_util', 'revol_util_int'), ('int_rate', 'int_rate_int')]:
        if col in df.columns:
            df[new_col] = df[col].str.replace('%', '', regex=False).astype(float)

    return df


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN for downstream imputation.

    Replacing with a magic number (e.g. 999) is incorrect — tree models
    will treat it as a real feature value and create spurious splits on it.
    NaN signals 'missing' to imputers correctly.
    """
    n_inf = df.isin([np.inf, -np.inf]).sum().sum()
    if n_inf > 0:
        print(f"Replacing {n_inf} infinite values with NaN")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all cleaning steps in order."""
    print("\n[Cleaning Pipeline]")
    df = drop_ambiguous_statuses(df)
    df = drop_post_loan_leakage_features(df)
    df = drop_high_missingness_columns(df, threshold=0.70)
    df = clean_type_conversions(df)
    df = regression_imputer(df, col='dti')
    if 'revol_util_int' in df.columns:
        df = regression_imputer(df, col='revol_util_int')
    df = handle_infinities(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features capturing borrower risk signals not present
    in the raw columns.

    Each feature is grounded in lending domain knowledge:

    income_category:
        Gross income buckets. Low-income borrowers face higher default risk
        when hit with income shocks. Categorical encoding allows the model to
        learn non-linear income thresholds.

    high_dti_risk:
        DTI > 35% is a standard lending industry threshold for elevated risk.
        Borrowers spending more than 35% of gross income on debt service have
        little buffer for unexpected expenses.

    fico_risk_group:
        Industry-standard FICO bands. 'Poor' (<580) and 'Fair' (580–670)
        borrowers are subprime; 'Excellent' (>740) are prime. This captures
        non-linearity in the FICO–default relationship.

    recent_inquiry_flag:
        >2 hard inquiries in 6 months signals active credit-seeking,
        often a sign of financial distress or multiple applications.

    loan_to_income_ratio:
        Captures affordability independent of raw loan size. A $20k loan
        means different things for a $30k vs $200k earner.

    interest_to_income_ratio:
        Monthly installment as a fraction of annual income. Values >5%
        indicate significant payment burden.

    loan_term_risk:
        60-month loans carry more risk than 36-month: longer exposure to
        income shocks, higher total interest burden.

    delinquency_risk:
        Any delinquency in the past 2 years is a strong recency signal.

    public_record_flag:
        Bankruptcies, judgments, and tax liens indicate severe past
        financial distress.

    home_ownership_risk:
        MORTGAGE/OWN borrowers have demonstrated ability to manage long-term
        debt; RENT status correlates with higher default rates in practice.

    purpose_group:
        Loan purpose groups by default risk profile. Small business loans
        carry highest uncertainty; debt consolidation is mixed.

    credit_history_years:
        Longer credit history generally correlates with lower default risk
        (more data points for creditors to assess behavior).

    revol_balance_to_income:
        Revolving utilization as a fraction of income — a forward-looking
        measure of credit dependency.

    payment_to_income:
        All monthly obligations relative to income; broader than DTI alone.
    """
    df = df.copy()

    # ── Original features ────────────────────────────────────────────────
    df['income_category'] = pd.cut(
        df['annual_inc'],
        bins=[0, 50_000, 100_000, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    df['high_dti_risk'] = (df['dti'] > 35).astype(int)

    df['fico_risk_group'] = pd.cut(
        df['fico_range_low'],
        bins=[0, 580, 670, 740, np.inf],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )

    df['recent_inquiry_flag'] = (df['inq_last_6mths'] > 2).astype(int)

    df['loan_to_income_ratio'] = (
        df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    )

    df['interest_to_income_ratio'] = (
        df['installment'] / df['annual_inc'].replace(0, np.nan)
    )

    if 'term' in df.columns:
        df['loan_term_risk'] = df['term'].apply(
            lambda x: 'Long term' if '60' in str(x) else 'Short term'
        )

    df['delinquency_risk'] = (df['delinq_2yrs'] > 0).astype(int)

    df['public_record_flag'] = (df['pub_rec'] > 0).astype(int)

    if 'home_ownership' in df.columns:
        df['home_ownership_risk'] = df['home_ownership'].isin(
            ['MORTGAGE', 'OWN']
        ).astype(int)

    if 'purpose' in df.columns:
        df['purpose_group'] = df['purpose'].map({
            'credit_card': 'Personal', 'home_improvement': 'Personal',
            'major_purchase': 'Personal', 'small_business': 'Business',
        }).fillna('Other')

    # ── New features ────────────────────────────────────────────────────
    # Credit history length in years (from earliest credit line)
    if 'earliest_cr_line' in df.columns:
        try:
            df['earliest_cr_year'] = pd.to_datetime(
                df['earliest_cr_line'], format='%b-%Y', errors='coerce'
            ).dt.year
            df['credit_history_years'] = (
                pd.to_datetime('today').year - df['earliest_cr_year']
            ).clip(lower=0)
        except Exception:
            pass

    # Revolving balance relative to income
    if 'revol_bal' in df.columns:
        df['revol_balance_to_income'] = (
            df['revol_bal'] / df['annual_inc'].replace(0, np.nan)
        )

    # Monthly payment burden as fraction of annual income
    if 'installment' in df.columns:
        df['payment_to_income'] = (
            (df['installment'] * 12) / df['annual_inc'].replace(0, np.nan)
        )

    # Total open accounts risk flag (>15 open accounts = high credit dependency)
    if 'open_acc' in df.columns:
        df['high_open_acc_risk'] = (df['open_acc'] > 15).astype(int)

    df = handle_infinities(df)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_default_rate_by_grade(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Bar chart of default rate by loan grade.

    Loan grade is assigned by LendingClub's internal risk model and should
    strongly predict default. If it doesn't in our data, it suggests either
    label quality issues or that the grade already prices in most of the risk
    (i.e., grade G loans charge high enough interest to compensate for losses).
    """
    if 'grade' not in df.columns:
        return

    grade_default = (
        df.groupby('grade')['loan_status']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'default_rate', 'count': 'n_loans'})
        .reset_index()
        .sort_values('grade')
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(grade_default['grade'], grade_default['default_rate'] * 100,
                   color=PALETTE[:len(grade_default)], edgecolor='white', alpha=0.85)
    ax1.set_xlabel('Loan Grade')
    ax1.set_ylabel('Default Rate (%)', color='navy')
    ax1.set_title('Default Rate by Loan Grade', fontsize=13, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(grade_default['grade'], grade_default['n_loans'],
             'o--', color='grey', alpha=0.6, label='Loan Count')
    ax2.set_ylabel('Number of Loans', color='grey')

    for bar, rate in zip(bars, grade_default['default_rate']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{rate:.1%}', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_default_rate_by_dti(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Default rate across DTI bins.

    We expect a roughly monotonic increase with DTI. A non-monotonic pattern
    would suggest that DTI alone is insufficient and interactions (e.g.,
    DTI × income level) matter — motivation for the loan_to_income and
    payment_to_income derived features.
    """
    df_plot = df.copy()
    df_plot['dti_bin'] = pd.cut(df_plot['dti'], bins=[0, 10, 15, 20, 25, 30, 35, 50, 200],
                                 labels=['0–10', '10–15', '15–20', '20–25',
                                         '25–30', '30–35', '35–50', '50+'])
    dti_default = (
        df_plot.groupby('dti_bin', observed=True)['loan_status']
        .mean()
        .reset_index()
        .rename(columns={'loan_status': 'default_rate'})
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dti_default['dti_bin'].astype(str), dti_default['default_rate'] * 100,
           color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('DTI Bin')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Rate by Debt-to-Income (DTI) Bin', fontsize=13, fontweight='bold')
    ax.axhline(df['loan_status'].mean() * 100, color='red', linestyle='--',
               label=f'Overall avg: {df["loan_status"].mean():.1%}')
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, features: list,
                              save_path: str = None) -> None:
    """
    Correlation heatmap for key numerical features + target.

    Helps identify multicollinearity (problematic for Logistic Regression)
    and strong individual predictors.
    """
    corr_mat = df[features + ['loan_status']].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax,
                linewidths=0.5, square=False)
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE SELECTION AND SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

# Features available at loan origination (no post-outcome leakage)
NUM_FEATURES = [
    'loan_amnt', 'int_rate_int', 'installment', 'term_numeric', 'annual_inc',
    'emp_length_int', 'dti', 'fico_range_high', 'revol_util_int',
    'loan_to_income_ratio', 'interest_to_income_ratio', 'high_dti_risk',
    'public_record_flag', 'home_ownership_risk', 'delinquency_risk',
    'recent_inquiry_flag', 'delinq_2yrs', 'inq_last_6mths',
    'revol_balance_to_income', 'payment_to_income', 'high_open_acc_risk',
]

CAT_FEATURES = [
    'grade', 'sub_grade', 'income_category', 'fico_risk_group',
    'loan_term_risk', 'purpose_group',
]


def prepare_features(df: pd.DataFrame,
                     num_features: list = NUM_FEATURES,
                     cat_features: list = CAT_FEATURES) -> tuple:
    """
    One-hot encode categoricals, impute, and return X, y arrays.

    Returns:
        X (np.ndarray), y (pd.Series), feature_names (list)
    """
    available_num = [f for f in num_features if f in df.columns]
    available_cat = [f for f in cat_features if f in df.columns]

    df_enc = pd.get_dummies(df[available_cat + available_num],
                             columns=available_cat, drop_first=True)
    feature_names = df_enc.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df_enc)
    y = df['loan_status']

    print(f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    return X, y, feature_names


def split_data(X, y, test_size: float = 0.20) -> tuple:
    """
    Stratified train/test split.

    Stratify ensures the default rate in train and test sets matches the
    overall population rate — critical for an imbalanced target.
    """
    return train_test_split(X, y, test_size=test_size,
                             random_state=RANDOM_SEED, stratify=y)


def apply_smote(X_train, y_train) -> tuple:
    """
    Apply SMOTE to training data ONLY.

    CRITICAL: SMOTE must never be applied before the train/test split.
    Doing so leaks synthetic test-set neighbors into training, creating
    an optimistic bias in recall and ROC-AUC.

    Returns:
        X_train_smote, y_train_smote
    """
    smote = SMOTE(random_state=RANDOM_SEED)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(y_train):,} → {len(y_sm):,} training samples "
          f"| class balance: {y_sm.mean():.2%} positive")
    return X_sm, y_sm


# ═══════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def build_models() -> dict:
    """
    Define the benchmark model suite.

    All models are evaluated on the same test set (origination-time features,
    pre-SMOTE split), ensuring fair comparison.

    Model rationale:
        Logistic Regression:  Interpretable baseline; coefficient signs give
                              direction of effect; well-calibrated probabilities.
        Random Forest:        Handles non-linearities and feature interactions;
                              less prone to overfitting than single trees.
        XGBoost:              State-of-the-art for tabular data; handles
                              missing values natively; built-in regularisation.
        GradientBoosting:     sklearn's GB as a reference vs XGBoost;
                              slower but sometimes better-calibrated.
    """
    return {
        'Logistic Regression': LogisticRegression(
            C=0.1, penalty='l1', solver='liblinear',
            max_iter=1000, random_state=RANDOM_SEED
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_SEED
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=RANDOM_SEED
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=RANDOM_SEED
        ),
    }


def train_all_models(models: dict, X_train_sm, y_train_sm,
                     X_test, y_test) -> tuple[dict, dict]:
    """
    Train all models and collect predictions.

    Args:
        models:       Dict of {name: sklearn estimator}
        X_train_sm:   SMOTE-augmented training features
        y_train_sm:   SMOTE-augmented training labels
        X_test:       Original (non-SMOTE) test features
        y_test:       Original test labels

    Returns:
        trained_models:  {name: fitted_estimator}
        predictions:     {name: {'y_pred': ..., 'y_prob': ...}}
    """
    trained  = {}
    preds    = {}

    for name, model in models.items():
        print(f"  Training {name}...", end=' ')
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, 'predict_proba')
                  else None)
        trained[name] = model
        preds[name]   = {'y_pred': y_pred, 'y_prob': y_prob}
        print(f"done | test AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return trained, preds


# ═══════════════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(y_test, preds: dict) -> pd.DataFrame:
    """
    Compute a unified metrics table for all models.

    All models are evaluated on the same y_test — this is the key fix
    over the original notebook where models used different test sets.
    """
    rows = []
    for name, p in preds.items():
        y_pred = p['y_pred']
        y_prob = p['y_prob']
        rows.append({
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1':        f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            'Avg Prec':  average_precision_score(y_test, y_prob) if y_prob is not None else np.nan,
            'Brier':     brier_score_loss(y_test, y_prob) if y_prob is not None else np.nan,
        })
    metrics_df = pd.DataFrame(rows).set_index('Model')
    print("\n=== Model Comparison (same test set) ===")
    print(metrics_df.round(4).to_string())
    return metrics_df


def plot_roc_comparison(y_test, preds: dict, save_path: str = None) -> None:
    """
    Overlay ROC curves for all models on a single plot.

    Comparing all models on the same axes on the same test set makes
    differences in ranking visible at all thresholds, not just at 0.5.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    for (name, p), color in zip(preds.items(), colors):
        if p['y_prob'] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, p['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models (same test set)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_comparison(y_test, preds: dict,
                                      save_path: str = None) -> None:
    """
    Precision-Recall curves for all models.

    For imbalanced datasets (default rate ~20%), PR curves are more
    informative than ROC:
    - ROC AUC is optimistic under class imbalance (many true negatives
      inflate TPR at low FPR).
    - PR AUC (Average Precision) reflects performance on the minority
      class (defaults) directly.

    A model with high PR-AUC is genuinely better at identifying defaults.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
    baseline = y_test.mean()

    for (name, p), color in zip(preds.items(), colors):
        if p['y_prob'] is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, p['y_prob'])
        avg_prec = average_precision_score(y_test, p['y_prob'])
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={avg_prec:.3f})")

    ax.axhline(baseline, color='grey', linestyle='--',
               label=f'No-skill baseline ({baseline:.1%})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_calibration_curves(y_test, preds: dict, save_path: str = None) -> None:
    """
    Calibration curves (reliability diagrams) for probability-outputting models.

    A calibrated model means: when it says P(default)=0.7, roughly 70% of
    those loans actually defaulted. Miscalibration matters hugely in lending:

    - If the model overestimates default probabilities, we reject too many
      good borrowers (lost revenue).
    - If it underestimates, we approve too many bad loans (losses).

    XGBoost is known to produce overconfident (extreme) probabilities.
    Post-hoc calibration (Platt scaling, isotonic regression) is the fix.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')

    for (name, p), color in zip(preds.items(), colors):
        if p['y_prob'] is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_test, p['y_prob'],
                                                  n_bins=10, strategy='uniform')
        ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2, label=name)

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Actual Default Rate)', fontsize=12)
    ax.set_title('Calibration Curves — Probability Reliability', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(y_test, preds: dict, save_path: str = None) -> None:
    """Side-by-side confusion matrices for all models at default threshold 0.5."""
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, p['y_pred'])
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['No Default', 'Default'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['No Default', 'Default'])
        thresh = cm.max() / 2
        for i, j in itertools.product(range(2), range(2)):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=10, color='white' if cm[i,j] > thresh else 'black')

    plt.suptitle('Confusion Matrices — Default Threshold 0.5',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 8. BUSINESS ANALYSIS — THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_business_cost(y_true, y_prob, threshold: float,
                           cost_fn: float = COST_FALSE_NEGATIVE,
                           cost_fp: float = COST_FALSE_POSITIVE) -> float:
    """
    Compute total relative business cost at a given classification threshold.

    In lending:
        False Negative = approving a loan that defaults → loss of principal
        False Positive = rejecting a loan that would have been repaid → lost revenue

    The default 0.5 threshold minimises classification error, not business cost.
    Lowering the threshold (more conservative) reduces FN at the cost of more FP.

    Args:
        cost_fn: Relative cost of a false negative (missed default). Default 7.
        cost_fp: Relative cost of a false positive (good borrower rejected). Default 1.

    Returns:
        Normalised total cost.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * cost_fn + fp * cost_fp) / len(y_true)
    return total_cost


def find_optimal_threshold(y_test, y_prob, model_name: str = '',
                            save_path: str = None) -> float:
    """
    Sweep thresholds and find the one that minimises business cost.

    Business insight: The optimal threshold for a lender is almost always
    below 0.5. Because missed defaults cost more than wrongly-rejected
    good borrowers, we want to flag borderline cases as defaults.

    Returns:
        Optimal threshold value.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    costs      = [compute_business_cost(y_test, y_prob, t) for t in thresholds]
    recalls    = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]
    precisions = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]

    optimal_idx  = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cost curve
    axes[0].plot(thresholds, costs, color='crimson', lw=2)
    axes[0].axvline(optimal_threshold, color='black', linestyle='--',
                    label=f'Optimal: {optimal_threshold:.2f}')
    axes[0].axvline(0.5, color='grey', linestyle=':', label='Default 0.5')
    axes[0].set_xlabel('Classification Threshold')
    axes[0].set_ylabel('Normalised Business Cost')
    axes[0].set_title(f'Business Cost vs Threshold\n{model_name}',
                       fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precision-Recall vs threshold
    axes[1].plot(thresholds, precisions, label='Precision', color='steelblue', lw=2)
    axes[1].plot(thresholds, recalls, label='Recall',    color='darkorange', lw=2)
    axes[1].axvline(optimal_threshold, color='black', linestyle='--',
                     label=f'Optimal: {optimal_threshold:.2f}')
    axes[1].axvline(0.5, color='grey', linestyle=':', label='Default 0.5')
    axes[1].set_xlabel('Classification Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Precision & Recall vs Threshold\n{model_name}',
                       fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    cost_at_05 = compute_business_cost(y_test, y_prob, 0.5)
    cost_at_opt = costs[optimal_idx]
    print(f"\nThreshold Analysis — {model_name}")
    print(f"  Default threshold (0.50): cost={cost_at_05:.4f}")
    print(f"  Optimal threshold ({optimal_threshold:.2f}): cost={cost_at_opt:.4f}")
    print(f"  Cost reduction: {100*(cost_at_05-cost_at_opt)/cost_at_05:.1f}%")

    return optimal_threshold


# ═══════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(trained_models: dict, feature_names: list,
                             top_n: int = 20, save_path: str = None) -> None:
    """
    Bar chart of top feature importances for tree-based models.

    XGBoost and Random Forest both expose `feature_importances_`. These
    are gain-based importances (for XGBoost) and mean impurity decrease
    (for RF). They tell us which features the model leans on most — but
    note they do NOT tell us the direction of effect (use SHAP for that).
    """
    tree_models = {k: v for k, v in trained_models.items()
                   if hasattr(v, 'feature_importances_')}

    if not tree_models:
        print("No tree-based models with feature_importances_ found.")
        return

    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    colors = ['steelblue', 'seagreen', 'darkorange', 'crimson']

    for ax, (name, model), color in zip(axes, tree_models.items(), colors):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top = importances.nlargest(top_n).sort_values()

        ax.barh(top.index, top.values, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(f'Top {top_n} Features\n{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Importance')

    plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 10. SUGGESTED EXPERIMENTS (not executed — clearly marked)
# ═══════════════════════════════════════════════════════════════════════════

def suggested_experiment__cross_validation(X, y):
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Replace single split with 5-fold stratified CV to quantify performance
    variance and get confidence intervals.

    Pseudocode:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            scores = cross_validate(model, X, y, cv=cv,
                                    scoring=['roc_auc','f1','precision','recall'])
            print(f"{name}: AUC={scores['test_roc_auc'].mean():.3f} "
                  f"± {scores['test_roc_auc'].std():.3f}")
    """
    print("SUGGESTED EXPERIMENT: cross_validation — see docstring.")


def suggested_experiment__hyperparameter_tuning():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Grid/random search for XGBoost and RF.
    Expected gain: 1–3% AUC improvement over defaults.

    Key parameters to tune:
        XGBoost: n_estimators, learning_rate, max_depth, subsample,
                 colsample_bytree, min_child_weight, gamma, reg_lambda
        RF:      n_estimators, max_depth, min_samples_leaf, max_features
        LR:      C, penalty (l1 vs l2 vs elasticnet)

    Pseudocode:
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 6, 8],
            'subsample': [0.6, 0.8, 1.0],
        }
        rs = RandomizedSearchCV(xgb.XGBClassifier(), param_dist,
                                n_iter=30, scoring='roc_auc', cv=3,
                                random_state=42, n_jobs=-1)
        rs.fit(X_train_sm, y_train_sm)
        print(rs.best_params_, rs.best_score_)
    """
    print("SUGGESTED EXPERIMENT: hyperparameter_tuning — see docstring.")


def suggested_experiment__shap_analysis(model, X_test_sample, feature_names):
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    SHAP (SHapley Additive exPlanations) provides model-agnostic feature
    attribution. Unlike feature_importances_, SHAP:
        - Shows direction of effect (positive SHAP = increases default prob)
        - Works at the individual loan level (local explanations)
        - Captures interaction effects between features

    Business use case:
        A loan officer can see exactly why the model flagged a particular
        application: "high DTI (+0.12) and recent inquiries (+0.08) pushed
        this borrower above the threshold despite good FICO (−0.05)."

    Pseudocode:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)

        # Global: bar chart of mean |SHAP|
        shap.summary_plot(shap_values, X_test_sample,
                          feature_names=feature_names, plot_type='bar')

        # Global: beeswarm showing direction + magnitude
        shap.summary_plot(shap_values, X_test_sample,
                          feature_names=feature_names)

        # Local: waterfall for one prediction
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            feature_names=feature_names
        ))
    """
    print("SUGGESTED EXPERIMENT: shap_analysis — see docstring.")


def suggested_experiment__probability_calibration(model, X_train, y_train,
                                                    X_test, y_test):
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    XGBoost and RF produce poorly calibrated probabilities. Post-hoc
    calibration (Platt scaling or isotonic regression) corrects this.

    Why it matters: if P(default) = 0.3 but the true rate is 0.5 for
    loans the model scores at 0.3, the lender is systematically
    underpricing risk.

    Pseudocode:
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrated.fit(X_val, y_val)
        y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

        # Compare calibration before/after
        frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw)
        frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_cal)
        print(f"Brier score before: {brier_score_loss(y_test, y_prob_raw):.4f}")
        print(f"Brier score after:  {brier_score_loss(y_test, y_prob_cal):.4f}")
    """
    print("SUGGESTED EXPERIMENT: probability_calibration — see docstring.")


def suggested_experiment__lightgbm_catboost():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    LightGBM and CatBoost are strong alternatives to XGBoost:
        - LightGBM: faster training (leaf-wise growth), often matches XGBoost
          accuracy with better memory efficiency
        - CatBoost: handles categorical features natively without
          one-hot encoding; often best on mixed datasets

    Pseudocode:
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    random_state=42)
        lgbm.fit(X_train_sm, y_train_sm)

        cat_features_idx = [i for i, f in enumerate(feature_names)
                             if any(c in f for c in CAT_FEATURES)]
        catb = CatBoostClassifier(iterations=300, learning_rate=0.05,
                                   cat_features=cat_features_idx,
                                   random_seed=42, verbose=0)
        catb.fit(X_train_sm, y_train_sm)
    """
    print("SUGGESTED EXPERIMENT: lightgbm_catboost — see docstring.")


# ═══════════════════════════════════════════════════════════════════════════
# 11. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  LOAN DEFAULT PREDICTION — LendingClub Dataset")
    print("=" * 65)

    # ── Load ────────────────────────────────────────────────────────────
    filepath = 'lending-club-2007-2020Q3/Loan_status_2007-2020Q3.gzip'
    df_raw = load_data(filepath)

    # ── Clean ───────────────────────────────────────────────────────────
    df = run_cleaning_pipeline(df_raw)

    # ── Feature Engineering ─────────────────────────────────────────────
    df = engineer_features(df)

    # ── EDA Visualisations ──────────────────────────────────────────────
    print("\n[EDA]")
    plot_default_rate_by_grade(df, save_path='default_rate_by_grade.png')
    plot_default_rate_by_dti(df, save_path='default_rate_by_dti.png')

    num_features_for_heatmap = [
        'loan_amnt', 'int_rate_int', 'dti', 'annual_inc', 'fico_range_high',
        'loan_to_income_ratio', 'revol_util_int', 'delinquency_risk'
    ]
    available_heatmap = [f for f in num_features_for_heatmap if f in df.columns]
    plot_correlation_heatmap(df, available_heatmap,
                              save_path='correlation_heatmap.png')

    # ── Prepare Features ────────────────────────────────────────────────
    print("\n[Feature Preparation]")
    X, y, feature_names = prepare_features(df)

    # ── Train/Test Split (BEFORE SMOTE) ────────────────────────────────
    print("\n[Splitting data]")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20)

    # ── SMOTE (train only) ──────────────────────────────────────────────
    print("\n[SMOTE — training data only]")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── Train Models ────────────────────────────────────────────────────
    print("\n[Training models]")
    models = build_models()
    trained_models, preds = train_all_models(
        models, X_train_sm, y_train_sm, X_test, y_test
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\n[Evaluation]")
    metrics_df = compute_all_metrics(y_test, preds)

    plot_roc_comparison(y_test, preds, save_path='roc_comparison.png')
    plot_precision_recall_comparison(y_test, preds,
                                      save_path='precision_recall_curves.png')
    plot_calibration_curves(y_test, preds, save_path='calibration_curves.png')
    plot_confusion_matrices(y_test, preds, save_path='confusion_matrices.png')

    # ── Feature Importance ──────────────────────────────────────────────
    print("\n[Feature Importance]")
    plot_feature_importance(trained_models, feature_names, top_n=20,
                             save_path='feature_importance.png')

    # ── Business Threshold Analysis (best model) ────────────────────────
    best_model_name = metrics_df['ROC-AUC'].idxmax()
    print(f"\n[Threshold Optimisation — {best_model_name}]")
    optimal_thresh = find_optimal_threshold(
        y_test,
        preds[best_model_name]['y_prob'],
        model_name=best_model_name,
        save_path='threshold_analysis.png'
    )

    # ── Suggested Experiments (not executed) ───────────────────────────
    print("\n" + "=" * 65)
    print("  SUGGESTED EXPERIMENTS (not executed)")
    print("=" * 65)
    suggested_experiment__cross_validation(X, y)
    suggested_experiment__hyperparameter_tuning()
    suggested_experiment__shap_analysis(
        trained_models[best_model_name],
        X_test[:1000], feature_names
    )
    suggested_experiment__probability_calibration(
        trained_models[best_model_name],
        X_train, y_train, X_test, y_test
    )
    suggested_experiment__lightgbm_catboost()

    print("\nDone. All output files saved to working directory.")
    return df, trained_models, preds, metrics_df, feature_names


if __name__ == "__main__":
    df, trained_models, preds, metrics_df, feature_names = main()
