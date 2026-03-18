# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8plus_v6a
평가 지표 : ROC-AUC

[v8plus_v6a 변경사항 - v8plus_v5 대비]
  단일 추가 검증 피처
  - IVF_출산전환율
====================================================
"""

import train_v8plus_v5 as base


base.VERSION = "v8plus_v6a"


def preprocess(train, test):
    X_train, y_train, X_test, feature_cols = base.preprocess(train, test)

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["IVF_출산전환율"] = X_train["IVF 출산 횟수_num"] / (X_train["IVF 임신 횟수_num"] + 1e-6)
    X_test["IVF_출산전환율"] = X_test["IVF 출산 횟수_num"] / (X_test["IVF 임신 횟수_num"] + 1e-6)

    feature_cols = list(X_train.columns)
    print("  v6a 추가 피처  : ['IVF_출산전환율']")
    print(f"  추가 후 피처 수: {len(feature_cols)}개")

    return X_train, y_train, X_test, feature_cols


if __name__ == "__main__":
    print("=" * 55)
    print(f"  난임 환자 임신 성공 여부 예측 ({base.VERSION} / CPU)")
    print("=" * 55)

    train, test, sub = base.load_data()
    X_train, y_train, X_test, feature_cols = preprocess(train, test)

    best_params = None
    if base.USE_OPTUNA:
        best_params = base.optuna_lgb(X_train, y_train, n_trials=base.OPTUNA_TRIALS)

    oof_lgb, pred_lgb, cv_lgb, importance = base.train_lgb(
        X_train, y_train, X_test, feature_cols, best_params
    )
    oof_xgb, pred_xgb, cv_xgb = base.train_xgb(X_train, y_train, X_test)
    oof_cat, pred_cat, cv_cat = base.train_cat(X_train, y_train, X_test)

    base.ensemble_and_save(
        y_train.values, sub,
        oof_lgb, oof_xgb, oof_cat,
        pred_lgb, pred_xgb, pred_cat,
        cv_lgb, cv_xgb, cv_cat
    )

    base.print_feature_importance(importance)
