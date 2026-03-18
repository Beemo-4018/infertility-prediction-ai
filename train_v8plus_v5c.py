# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8plus_v5c
평가 지표 : ROC-AUC

[v8plus_v5c 변경사항 - v8plus_v5 대비]
  v5 핵심 피처 중
  - 난자정자출처조합_te
  만 제거하여 검증
====================================================
"""

import train_v8plus_v5 as base


base.VERSION = "v8plus_v5c"


def preprocess(train, test):
    X_train, y_train, X_test, feature_cols = base.preprocess(train, test)

    drop_feature = "난자정자출처조합_te"
    if drop_feature in X_train.columns:
        X_train = X_train.drop(columns=[drop_feature])
        X_test = X_test.drop(columns=[drop_feature])

    feature_cols = list(X_train.columns)
    print(f"  v5c 제거 피처  : ['{drop_feature}']")
    print(f"  제거 후 피처 수: {len(feature_cols)}개")

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
