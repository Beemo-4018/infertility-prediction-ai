# -*- coding: utf-8 -*-
"""
Project entry point.

- `python main.py`: run the current final baseline (`v8_final`)
- `python train_v8plus_v8_lgb_screen.py`: run quick LightGBM screening
- `python train_v8plus_v8_final_seedavg.py`: run seed averaging ensemble
"""

import train_v8plus_v8_final as run


if __name__ == "__main__":
    print("=" * 55)
    print("  Main entry -> v8plus_v8_final")
    print("=" * 55)

    train, test, sub = run.load_data()
    X_train, y_train, X_test, feature_cols = run.build_v8_final_features(train, test)

    best_params = None
    if run.USE_OPTUNA:
        best_params = run.optuna_lgb(
            X_train, y_train, n_trials=run.OPTUNA_TRIALS
        )

    oof_lgb, pred_lgb, cv_lgb, importance = run.train_lgb(
        X_train, y_train, X_test, feature_cols, best_params
    )
    oof_xgb, pred_xgb, cv_xgb = run.train_xgb(X_train, y_train, X_test)
    oof_cat, pred_cat, cv_cat = run.train_cat(X_train, y_train, X_test)

    run.ensemble_and_save(
        y_train.values,
        sub,
        oof_lgb,
        oof_xgb,
        oof_cat,
        pred_lgb,
        pred_xgb,
        pred_cat,
        cv_lgb,
        cv_xgb,
        cv_cat,
    )

    run.print_feature_importance(importance)
