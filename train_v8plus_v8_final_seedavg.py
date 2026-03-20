# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8plus_v8_final_seedavg
평가 지표: ROC-AUC

[용도]
  v8_final 기준 시드 평균 앙상블 구조
  - 기본 seeds: 42, 2024, 777
  - 각 seed에서 LGB+CAT(0.6/0.4) 조합 생성
  - 최종 예측은 seed 간 단순 평균
====================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score

import train_v8plus_v8_final as base


VERSION = "v8plus_v8_final_seedavg"
SEEDS = [42, 2024, 777]
RUN_OPTUNA_EACH_SEED = True


def run_single_seed(seed):
    old_seed = base.SEED
    old_version = base.VERSION
    try:
        base.SEED = seed
        base.VERSION = f"v8plus_v8_final_seed{seed}"

        train, test, _ = base.load_data()
        X_train, y_train, X_test, feature_cols = base.build_v8_final_features(train, test)

        best_params = None
        if RUN_OPTUNA_EACH_SEED and base.USE_OPTUNA:
            best_params = base.optuna_lgb(X_train, y_train, n_trials=base.OPTUNA_TRIALS)

        oof_lgb, pred_lgb, cv_lgb, _ = base.train_lgb(X_train, y_train, X_test, feature_cols, best_params)
        oof_cat, pred_cat, cv_cat = base.train_cat(X_train, y_train, X_test)

        blend_oof = 0.6 * oof_lgb + 0.4 * oof_cat
        blend_pred = 0.6 * pred_lgb + 0.4 * pred_cat
        blend_cv = roc_auc_score(y_train, blend_oof)

        return {
            "seed": seed,
            "y_train": y_train.values,
            "blend_oof": blend_oof,
            "blend_pred": blend_pred,
            "blend_cv": blend_cv,
        }
    finally:
        base.SEED = old_seed
        base.VERSION = old_version


if __name__ == "__main__":
    print("=" * 55)
    print(f"  난임 환자 임신 성공 여부 예측 ({VERSION} / CPU)")
    print("=" * 55)
    print(f"  사용 시드: {SEEDS}")
    print(f"  시드별 Optuna: {RUN_OPTUNA_EACH_SEED}")

    seed_runs = []
    for seed in SEEDS:
        print("\n" + "=" * 55)
        print(f"[Seed Run] {seed}")
        print("=" * 55)
        result = run_single_seed(seed)
        seed_runs.append(result)
        print(f"  seed {seed} blend CV : {result['blend_cv']:.5f}")

    avg_oof = np.mean([r["blend_oof"] for r in seed_runs], axis=0)
    avg_pred = np.mean([r["blend_pred"] for r in seed_runs], axis=0)
    y_train = seed_runs[0]["y_train"]
    final_cv = roc_auc_score(y_train, avg_oof)

    print("\n" + "=" * 55)
    print("[Final] Seed Average")
    print("=" * 55)
    for r in seed_runs:
        print(f"  seed {r['seed']} : {r['blend_cv']:.5f}")
    print(f"  평균 앙상블 CV : {final_cv:.5f}")

    _, _, sub = base.load_data()
    os.makedirs(base.SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_path = f"{base.SAVE_PATH}sub_{VERSION}_cpu_{timestamp}_cv{final_cv:.5f}.csv"
    sub["probability"] = avg_pred
    sub.to_csv(out_path, index=False)
    print(f"\n  저장: {out_path}")
