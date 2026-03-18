# -*- coding: utf-8 -*-
"""
Fast LightGBM-only screening for v8plus_v5 feature ablation/addition.

Purpose
- Quickly estimate which v5b-added features actually matter
- Quickly test whether any v5a-only features help when added on top of v5

Strategy
- Reuse train_v8plus_v5 preprocess as the base feature set
- Evaluate with LightGBM only
- Optional 3-fold screening for speed
- Compare remove/add candidates in one run
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import train_v8plus_v5 as v5


REMOVE_CANDIDATES = [
    "난자정자출처조합_te",
    "총배아대비_해동비율",
    "IVF_실패부담",
]

ADD_CANDIDATES = [
    "출산전환율",
    "IVF_출산전환율",
    "기증자정자_비율",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fast v5 feature ablation screening")
    parser.add_argument("--folds", type=int, default=3, help="CV folds for screening")
    parser.add_argument("--n-estimators", type=int, default=3000, help="LightGBM boosting rounds")
    parser.add_argument("--early-stopping", type=int, default=150, help="Early stopping rounds")
    parser.add_argument("--output", type=Path, default=Path("submissions/v5_screen_results.csv"))
    return parser.parse_args()


def train_lgb_cv(X, y, n_folds=3, n_estimators=3000, early_stopping=150):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": v5.SEED,
        "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=v5.SEED)
    oof = np.zeros(len(X))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = pred
        fold_auc = roc_auc_score(y_val, pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold}/{n_folds} AUC: {fold_auc:.5f} (best_iter: {model.best_iteration_})")

    cv = roc_auc_score(y, oof)
    return cv, fold_scores


def evaluate_case(name, X, y, feature_drop=None, feature_add=None, args=None):
    X_case = X.copy()

    if feature_drop is not None and feature_drop in X_case.columns:
        X_case = X_case.drop(columns=[feature_drop])
    if feature_add is not None and feature_add not in X_case.columns:
        raise ValueError(f"Feature to add not found in base set: {feature_add}")

    print("\n" + "=" * 70)
    print(f"[{name}] features={X_case.shape[1]}")
    if feature_drop:
        print(f"  remove: {feature_drop}")
    if feature_add:
        print(f"  add   : {feature_add}")
    print("=" * 70)

    cv, fold_scores = train_lgb_cv(
        X_case,
        y,
        n_folds=args.folds,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
    )
    print(f"  => CV AUC: {cv:.5f}")
    return {
        "case": name,
        "feature_count": X_case.shape[1],
        "removed": feature_drop or "",
        "added": feature_add or "",
        "cv_auc": cv,
        "fold_scores": ",".join(f"{s:.5f}" for s in fold_scores),
    }


def main():
    args = parse_args()

    print("=" * 70)
    print("v5 Feature Screening (LightGBM only)")
    print("=" * 70)

    train, test, _ = v5.load_data()
    X_train, y_train, _, feature_cols = v5.preprocess(train, test)

    base_feature_set = set(feature_cols)
    missing_remove = [f for f in REMOVE_CANDIDATES if f not in base_feature_set]
    missing_add = [f for f in ADD_CANDIDATES if f not in base_feature_set]
    if missing_remove:
        print(f"warning: remove candidates not found: {missing_remove}")
    if missing_add:
        print(f"warning: add candidates not found: {missing_add}")

    results = []
    results.append(evaluate_case("base_v5", X_train, y_train, args=args))

    for feature in REMOVE_CANDIDATES:
        if feature in X_train.columns:
            results.append(
                evaluate_case(f"remove_{feature}", X_train, y_train, feature_drop=feature, args=args)
            )

    # For add-candidate screening, use a v5b-like base:
    # remove the v5a-only features from the current v5 set, then add one back at a time.
    v5b_base = X_train.drop(columns=[f for f in ADD_CANDIDATES if f in X_train.columns]).copy()
    print("\n" + "=" * 70)
    print(f"[base_v5b_like] features={v5b_base.shape[1]}")
    print("=" * 70)
    base_v5b_cv, base_v5b_folds = train_lgb_cv(
        v5b_base,
        y_train,
        n_folds=args.folds,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
    )
    print(f"  => CV AUC: {base_v5b_cv:.5f}")
    results.append(
        {
            "case": "base_v5b_like",
            "feature_count": v5b_base.shape[1],
            "removed": ",".join([f for f in ADD_CANDIDATES if f in X_train.columns]),
            "added": "",
            "cv_auc": base_v5b_cv,
            "fold_scores": ",".join(f"{s:.5f}" for s in base_v5b_folds),
        }
    )

    for feature in ADD_CANDIDATES:
        if feature in X_train.columns:
            X_case = v5b_base.copy()
            X_case[feature] = X_train[feature]
            print("\n" + "=" * 70)
            print(f"[add_to_v5b_like_{feature}] features={X_case.shape[1]}")
            print(f"  add: {feature}")
            print("=" * 70)
            cv, fold_scores = train_lgb_cv(
                X_case,
                y_train,
                n_folds=args.folds,
                n_estimators=args.n_estimators,
                early_stopping=args.early_stopping,
            )
            print(f"  => CV AUC: {cv:.5f}")
            results.append(
                {
                    "case": f"add_to_v5b_like_{feature}",
                    "feature_count": X_case.shape[1],
                    "removed": "",
                    "added": feature,
                    "cv_auc": cv,
                    "fold_scores": ",".join(f"{s:.5f}" for s in fold_scores),
                }
            )

    result_df = pd.DataFrame(results).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Screening complete")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
