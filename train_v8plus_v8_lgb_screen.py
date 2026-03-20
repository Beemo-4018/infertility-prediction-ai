# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8plus_v8_lgb_screen
평가 지표: ROC-AUC

[용도]
  v8_final 풀체인(v6e + v7b + v8) 기준
  LightGBM 빠른 스크리닝 베이스라인
====================================================
"""

import train_v8plus_v8_final as base


VERSION = "v8plus_v8_lgb_screen"


def main():
    print("=" * 55)
    print(f"  난임 환자 임신 성공 여부 예측 ({VERSION})")
    print("=" * 55)

    train, test, _ = base.load_data()
    X_train, y_train, X_test, feature_cols = base.build_v8_final_features(train, test)

    _, _, cv_lgb, importance = base.train_lgb(
        X_train, y_train, X_test, feature_cols, best_params=None
    )

    print("\n" + "=" * 55)
    print(f"[Screen] {VERSION} LightGBM CV AUC : {cv_lgb:.5f}")
    print("=" * 55)
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
