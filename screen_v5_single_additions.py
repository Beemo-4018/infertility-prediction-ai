# -*- coding: utf-8 -*-
"""
Fast LightGBM-only screening for single-feature additions on top of v8plus_v5.

Candidates
- 출산전환율
- 기증자정자_비율
- IVF_출산전환율
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import train_v8plus_v5 as v5


ADD_CANDIDATES = [
    "출산전환율",
    "기증자정자_비율",
    "IVF_출산전환율",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fast single-add screening on top of v5")
    parser.add_argument("--folds", type=int, default=3, help="CV folds for screening")
    parser.add_argument("--n-estimators", type=int, default=3000, help="LightGBM boosting rounds")
    parser.add_argument("--early-stopping", type=int, default=150, help="Early stopping rounds")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/v5_single_add_results.csv"),
        help="CSV path to save screening results",
    )
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


def build_candidate_series(name, X):
    if name == "출산전환율":
        return X["총 출산 횟수_num"] / (X["총 임신 횟수_num"] + 1e-6)
    if name == "기증자정자_비율":
        return X["기증자 정자와 혼합된 난자 수"] / (X["혼합된 난자 수"] + 1e-6)
    if name == "IVF_출산전환율":
        return X["IVF 출산 횟수_num"] / (X["IVF 임신 횟수_num"] + 1e-6)
    raise ValueError(f"Unknown candidate: {name}")


def evaluate_case(case_name, X, y, args):
    print("\n" + "=" * 70)
    print(f"[{case_name}] features={X.shape[1]}")
    print("=" * 70)
    cv, fold_scores = train_lgb_cv(
        X,
        y,
        n_folds=args.folds,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
    )
    print(f"  => CV AUC: {cv:.5f}")
    return cv, fold_scores


def main():
    args = parse_args()

    print("=" * 70)
    print("v5 Single-Addition Screening (LightGBM only)")
    print("=" * 70)

    train, test, _ = v5.load_data()
    X_train, y_train, _, feature_cols = v5.preprocess(train, test)

    results = []

    base_cv, base_folds = evaluate_case("base_v5", X_train, y_train, args)
    results.append(
        {
            "case": "base_v5",
            "feature_count": X_train.shape[1],
            "added": "",
            "cv_auc": base_cv,
            "delta_vs_base": 0.0,
            "fold_scores": ",".join(f"{s:.5f}" for s in base_folds),
        }
    )

    for feature in ADD_CANDIDATES:
        X_case = X_train.copy()
        X_case[feature] = build_candidate_series(feature, X_train)
        cv, fold_scores = evaluate_case(f"add_{feature}", X_case, y_train, args)
        results.append(
            {
                "case": f"add_{feature}",
                "feature_count": X_case.shape[1],
                "added": feature,
                "cv_auc": cv,
                "delta_vs_base": cv - base_cv,
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
