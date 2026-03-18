# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8plus_v4a
평가 지표 : ROC-AUC

[v8plus_v4a 변경사항 - v8plus_v4 대비]
  v4에서 제거했던 저중요도 피처 중
  - 고령_여부
  만 다시 복구
====================================================
"""

import train_v8plus_v3 as base


V4A_DROPS = [
    "난자기증자_젊음",
    "반복시술_여부",
    "남성 부 불임 원인",
    "난자기증자_있음",
    "해동난자_있음",
    "임신_경험",
    "동결 배아 사용 여부",
    "최적연령_여부",
]


base.VERSION = "v8plus_v4a"
base.ALL_DROPS = list(dict.fromkeys(base.V2_DROPS + base.V3_DROPS + V4A_DROPS))


if __name__ == "__main__":
    print("=" * 55)
    print(f"  난임 환자 임신 성공 여부 예측 ({base.VERSION} / CPU)")
    print("=" * 55)

    train, test, sub = base.load_data()
    X_train, y_train, X_test, feature_cols = base.preprocess(train, test)

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
