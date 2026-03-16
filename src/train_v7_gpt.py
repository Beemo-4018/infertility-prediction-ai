import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# =====================================================
# PATH
# =====================================================

DATA_PATH = "../data/"
SAVE_PATH = "../data/submissions/"
os.makedirs(SAVE_PATH, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=================================================")
print("데이터 로드")
print("=================================================")

train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

target_col = "임신 성공 여부"

y = train[target_col]

train = train.drop(columns=[target_col])

# ID 제거
if "ID" in train.columns:
    train = train.drop(columns=["ID"])
    test = test.drop(columns=["ID"])

X = train.copy()
X_test = test.copy()

print("train shape :", X.shape)
print("test shape  :", X_test.shape)

# =====================================================
# PREPROCESS
# =====================================================

cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:

    le = LabelEncoder()

    combined = pd.concat([X[col], X_test[col]]).astype(str)

    le.fit(combined)

    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X = X.fillna(-999)
X_test = X_test.fillna(-999)

# =====================================================
# PARAMS
# =====================================================

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1
}

cat_params = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 6,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 0
}

meta_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 200,
    "verbose": -1
}

# =====================================================
# SEEDS
# =====================================================

SEEDS = [42, 2024, 777]

final_preds = []

# =====================================================
# TRAIN LOOP
# =====================================================

for seed in SEEDS:

    print("\n===================================")
    print("Seed:", seed)
    print("===================================")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    lgb_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))

    lgb_test = np.zeros(len(X_test))
    xgb_test = np.zeros(len(X_test))
    cat_test = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        print("Fold", fold + 1)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        # ----------------------
        # LightGBM
        # ----------------------

        lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=8000)

        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )

        lgb_oof[valid_idx] = lgb_model.predict_proba(X_valid)[:,1]
        lgb_test += lgb_model.predict_proba(X_test)[:,1] / skf.n_splits

        # ----------------------
        # XGBoost
        # ----------------------

        xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=5000)

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        xgb_oof[valid_idx] = xgb_model.predict_proba(X_valid)[:,1]
        xgb_test += xgb_model.predict_proba(X_test)[:,1] / skf.n_splits

        # ----------------------
        # CatBoost
        # ----------------------

        cat_model = CatBoostClassifier(**cat_params)

        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            verbose=False
        )

        cat_oof[valid_idx] = cat_model.predict_proba(X_valid)[:,1]
        cat_test += cat_model.predict_proba(X_test)[:,1] / skf.n_splits

    print("LightGBM CV :", roc_auc_score(y, lgb_oof))
    print("XGBoost CV  :", roc_auc_score(y, xgb_oof))
    print("CatBoost CV :", roc_auc_score(y, cat_oof))

    # =====================================================
    # STACKING
    # =====================================================

    meta_train = pd.DataFrame({
        "lgb": lgb_oof,
        "xgb": xgb_oof,
        "cat": cat_oof
    })

    meta_test = pd.DataFrame({
        "lgb": lgb_test,
        "xgb": xgb_test,
        "cat": cat_test
    })

    meta_model = lgb.LGBMClassifier(**meta_params, n_estimators=2000)

    meta_model.fit(meta_train, y)

    meta_pred = meta_model.predict_proba(meta_test)[:,1]

    final_preds.append(meta_pred)

# =====================================================
# SEED ENSEMBLE
# =====================================================

print("\nSeed Ensemble")

final_prediction = np.mean(final_preds, axis=0)

# =====================================================
# SUBMISSION
# =====================================================

submission = pd.read_csv(DATA_PATH + "sample_submission.csv")

target_name = submission.columns[-1]

submission[target_name] = final_prediction

save_name = SAVE_PATH + "submission_v7_fixed.csv"

submission.to_csv(save_name, index=False)

print("\nSubmission Saved:", save_name)