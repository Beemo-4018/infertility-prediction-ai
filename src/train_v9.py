# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v9 (Seed Ensemble)
평가 지표  : ROC-AUC
실행 방법  : python src/train_v9.py

[v9 변경사항 - v5 대비]
    - SEED 42 / 2024 / 777 세 개로 각각 학습 후 평균
    - 피처/모델/파라미터는 v5와 완전 동일
    - 랜덤성에 의한 분산 감소 → 안정적인 LB 기대

[Data Leakage 방지 원칙]
    1. LabelEncoder는 train 데이터로만 fit, test는 transform만 수행
    2. 결측치 보간 통계값(median)은 train 기준으로만 계산 후 test에 적용
    3. 파생 변수는 각 행(row) 내 연산만 수행
    4. Target Encoding은 K-Fold OOF 방식으로 train 내부 leakage 차단
====================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ====================================================
# 설정값
# ====================================================
SEEDS         = [42, 2024, 777]   # 세 가지 시드로 앙상블
N_FOLDS       = 5
TARGET        = '임신 성공 여부'
DATA_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
USE_OPTUNA    = True
OPTUNA_TRIALS = 50


# ====================================================
# 1. 데이터 로드
# ====================================================
def load_data():
    print('=' * 55)
    print('[1] 데이터 로드')
    print('=' * 55)
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    sub   = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    print(f'  train shape  : {train.shape}')
    print(f'  test  shape  : {test.shape}')
    success = train[TARGET].mean()
    print(f'  임신 성공 비율 : {success:.4f} ({success*100:.2f}%)')
    return train, test, sub


# ====================================================
# 2. Target Encoding (K-Fold OOF - leakage 없음)
# ====================================================
def target_encode(train, test, col, target, seed, n_splits=5, smooth=20):
    global_mean = train[target].mean()
    train_enc   = np.zeros(len(train))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in skf.split(train, train[target]):
        stats    = train.iloc[tr_idx].groupby(col)[target].agg(['mean', 'count'])
        smoothed = (
            (stats['mean'] * stats['count'] + global_mean * smooth)
            / (stats['count'] + smooth)
        )
        train_enc[val_idx] = (
            train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
        )

    full_stats  = train.groupby(col)[target].agg(['mean', 'count'])
    full_smooth = (
        (full_stats['mean'] * full_stats['count'] + global_mean * smooth)
        / (full_stats['count'] + smooth)
    )
    test_enc = test[col].map(full_smooth).fillna(global_mean).values
    return train_enc, test_enc


# ====================================================
# 3. 전처리 & 피처 엔지니어링 (v5와 동일, seed 적용)
# ====================================================
def preprocess(train, test, seed):
    print(f'\n  [SEED={seed}] 전처리 & 피처 엔지니어링')

    train = train.copy()
    test  = test.copy()

    # ── 나이 수치화 ─────────────────────────────────────────────
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치']     = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

    # ── 횟수 컬럼 수치화 ────────────────────────────────────────
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수',
        'IVF 시술 횟수', 'DI 시술 횟수',
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]

    def parse_count(val):
        if pd.isna(val):
            return np.nan
        s = str(val).replace('회', '').replace(' 이상', '').strip()
        try:
            return float(s)
        except Exception:
            return np.nan

    for col in count_cols:
        for df in [train, test]:
            df[col + '_num'] = df[col].apply(parse_count)

    # ── 기본 파생 피처 ──────────────────────────────────────────
    for df in [train, test]:
        df['과거_임신성공률'] = df['총 임신 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['과거_출산성공률'] = df['총 출산 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['IVF_임신성공률'] = df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['IVF_출산성공률'] = df['IVF 출산 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['DI_임신성공률']  = df['DI 임신 횟수_num']  / (df['DI 시술 횟수_num'] + 1e-6)

        df['IVF_경험']      = (df['IVF 시술 횟수_num'] > 0).astype(int)
        df['DI_경험']       = (df['DI 시술 횟수_num']  > 0).astype(int)
        df['임신_경험']     = (df['총 임신 횟수_num']   > 0).astype(int)
        df['출산_경험']     = (df['총 출산 횟수_num']   > 0).astype(int)
        df['반복시술_여부'] = (df['총 시술 횟수_num']   >= 3).astype(int)
        df['클리닉_집중도'] = (
            df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )

        df['배아_이식비율']   = df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율']   = df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        df['배아_활용률']     = (
            (df['이식된 배아 수'] + df['저장된 배아 수']) / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        )
        df['미세주입_이식률'] = (
            df['미세주입 배아 이식 수'] / (df['미세주입에서 생성된 배아 수'] + 1e-6)
        )
        df['난자_수정률']     = df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1e-6)
        df['파트너정자_비율'] = (
            df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        )

        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        df['배반포_이식추정'] = (df['혼합_이식_간격'] >= 5).astype(int)

        male_cause_cols = [
            '불임 원인 - 남성 요인', '불임 원인 - 정자 농도',
            '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'
        ]
        female_cause_cols = [
            '불임 원인 - 난관 질환', '불임 원인 - 배란 장애',
            '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증'
        ]
        all_cause_cols = male_cause_cols + female_cause_cols + [
            '남성 주 불임 원인', '남성 부 불임 원인',
            '여성 주 불임 원인', '여성 부 불임 원인',
            '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인'
        ]
        df['남성_불임원인_수'] = df[male_cause_cols].sum(axis=1)
        df['여성_불임원인_수'] = df[female_cause_cols].sum(axis=1)
        df['총_불임원인_수']   = df[all_cause_cols].sum(axis=1)
        df['복합_불임원인']    = (df['총_불임원인_수'] >= 2).astype(int)
        df['불명확_단독원인']  = (
            (df['불명확 불임 원인'] == 1) & (df['총_불임원인_수'] == 1)
        ).astype(int)

        for col in [
            '착상 전 유전 검사 사용 여부', 'PGD 시술 여부',
            'PGS 시술 여부', '난자 해동 경과일', '배아 해동 경과일',
            '임신 시도 또는 마지막 임신 경과 연수'
        ]:
            df[col + '_결측'] = df[col].isnull().astype(int)

        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

        df['시술유형_나이조합'] = (
            df['특정 시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        )
        df['시술유형_불임주원인조합'] = (
            df['특정 시술 유형'].astype(str)
            + '_male' + df['남성 주 불임 원인'].astype(str)
            + '_female' + df['여성 주 불임 원인'].astype(str)
        )

        df['남성요인_ICSI매칭'] = (
            (df['불임 원인 - 남성 요인'] == 1) &
            (df['특정 시술 유형'].astype(str).str.contains('ICSI'))
        ).astype(int)
        df['배란장애_자극매칭'] = (
            (df['불임 원인 - 배란 장애'] == 1) &
            (df['배란 자극 여부'] == 1)
        ).astype(int)
        df['고령_동결배아조합'] = (
            (df['고령_여부'] == 1) & (df['동결배아_시술'] == 1)
        ).astype(int)

        sperm_issues = df[
            ['불임 원인 - 정자 농도', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']
        ].sum(axis=1)
        df['정자문제_파트너정자'] = (
            (sperm_issues > 0) & (df['파트너정자_비율'] > 0.5)
        ).astype(int)
        df['초고령_반복시술'] = (
            (df['초고령_여부'] == 1) & (df['반복시술_여부'] == 1)
        ).astype(int)

    # ── 피처 컬럼 확정 ──────────────────────────────────────────
    drop_cols    = ['ID', TARGET] + count_cols + ['클리닉_나이조합', '클리닉_시술유형조합']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train[feature_cols].select_dtypes(include='object').columns.tolist()

    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)
    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # ── 클리닉 집계 피처 (seed 적용) ────────────────────────────
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    tr_clinic_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        stats    = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (
            (stats['mean'] * stats['count'] + global_mean * 20)
            / (stats['count'] + 20)
        )
        tr_clinic_rate[val_idx] = (
            train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
        )
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (
        (full_stats['mean'] * full_stats['count'] + global_mean * 20)
        / (full_stats['count'] + 20)
    )
    train['시술시기코드_성공률'] = tr_clinic_rate
    test['시술시기코드_성공률']  = test[clinic_col].map(full_smooth).fillna(global_mean).values

    tr_clinic_cnt = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        cnt_map = train.iloc[tr_idx].groupby(clinic_col).size()
        tr_clinic_cnt[val_idx] = (
            train.iloc[val_idx][clinic_col].map(cnt_map).fillna(1).values
        )
    full_cnt_map = train.groupby(clinic_col).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_clinic_cnt)
    test['시술시기코드_시술건수']  = np.log1p(
        test[clinic_col].map(full_cnt_map).fillna(1).values
    )

    train['시술시기코드_성공률편차'] = train['시술시기코드_성공률'] - global_mean
    test['시술시기코드_성공률편차']  = test['시술시기코드_성공률']  - global_mean

    train['클리닉_나이조합'] = (
        train[clinic_col].astype(str) + '_' + train['시술 당시 나이'].astype(str)
    )
    test['클리닉_나이조합'] = (
        test[clinic_col].astype(str) + '_' + test['시술 당시 나이'].astype(str)
    )
    tr_enc2, te_enc2 = target_encode(train, test, '클리닉_나이조합', TARGET, seed)
    train['클리닉_나이별성공률'] = tr_enc2
    test['클리닉_나이별성공률']  = te_enc2

    train['클리닉_시술유형조합'] = (
        train[clinic_col].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    )
    test['클리닉_시술유형조합'] = (
        test[clinic_col].astype(str) + '_' + test['특정 시술 유형'].astype(str)
    )
    tr_enc3, te_enc3 = target_encode(train, test, '클리닉_시술유형조합', TARGET, seed)
    train['클리닉_시술유형별성공률'] = tr_enc3
    test['클리닉_시술유형별성공률']  = te_enc3

    emb_global_mean = train['이식된 배아 수'].mean()
    tr_emb_mean = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        emb_map = train.iloc[tr_idx].groupby(clinic_col)['이식된 배아 수'].mean()
        tr_emb_mean[val_idx] = (
            train.iloc[val_idx][clinic_col].map(emb_map).fillna(emb_global_mean).values
        )
    full_emb_map = train.groupby(clinic_col)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb_mean
    test['시술시기코드_배아이식수평균']  = (
        test[clinic_col].map(full_emb_map).fillna(emb_global_mean).values
    )

    # ── Target Encoding (seed 적용) ──────────────────────────────
    te_cols = [
        '시술 시기 코드', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유',
        '난자 출처', '정자 출처'
    ]
    te_interaction_cols = ['시술유형_나이조합', '시술유형_불임주원인조합']

    for col in te_cols + te_interaction_cols:
        if col in train.columns and col in test.columns:
            tr_enc, te_enc = target_encode(train, test, col, TARGET, seed)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    # ── Label Encoding (train만 fit) ─────────────────────────────
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col].astype(str).tolist() + ['Unknown'])
        known     = set(le.classes_)
        test[col] = test[col].astype(str).apply(
            lambda x: x if x in known else 'Unknown'
        )
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    feature_cols = [c for c in train.columns if c not in drop_cols]
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    return X_train, y_train, X_test, feature_cols


# ====================================================
# 4. Optuna (SEED=42 고정으로 탐색)
# ====================================================
def optuna_lgb(X_train, y_train, n_trials=50):
    print('\n' + '=' * 55)
    print(f'[3] Optuna 튜닝 (LightGBM, {n_trials} trials)')
    print('=' * 55)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'objective'        : 'binary',
            'metric'           : 'auc',
            'verbose'          : -1,
            'random_state'     : 42,
            'n_jobs'           : -1,
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.05),
            'num_leaves'       : trial.suggest_int('num_leaves', 31, 255),
            'max_depth'        : trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr  = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]
            m = lgb.LGBMClassifier(**params, n_estimators=2000)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            aucs.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n  최적 CV AUC  : {study.best_value:.5f}')
    print(f'  최적 파라미터 : {study.best_params}')
    return study.best_params


# ====================================================
# 5. 단일 시드 학습 (LGBm + XGB + CatBoost)
# ====================================================
def train_one_seed(X_train, y_train, X_test, seed, best_params=None):
    print('\n' + '=' * 55)
    print(f'[SEED={seed}] 베이스 모델 학습')
    print('=' * 55)

    # LightGBM
    if best_params:
        lgb_params = {
            **best_params,
            'objective'   : 'binary',
            'metric'      : 'auc',
            'verbose'     : -1,
            'random_state': seed,
            'n_jobs'      : -1
        }
    else:
        lgb_params = {
            'objective'        : 'binary',
            'metric'           : 'auc',
            'learning_rate'    : 0.02,
            'num_leaves'       : 127,
            'max_depth'        : -1,
            'min_child_samples': 20,
            'feature_fraction' : 0.8,
            'bagging_fraction' : 0.8,
            'bagging_freq'     : 5,
            'reg_alpha'        : 0.1,
            'reg_lambda'       : 0.1,
            'verbose'          : -1,
            'random_state'     : seed,
            'n_jobs'           : -1
        }

    xgb_params = {
        'objective'       : 'binary:logistic',
        'eval_metric'     : 'auc',
        'learning_rate'   : 0.02,
        'max_depth'       : 7,
        'subsample'       : 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha'       : 0.1,
        'reg_lambda'      : 1.0,
        'random_state'    : seed,
        'n_jobs'          : -1,
        'verbosity'       : 0,
        'tree_method'     : 'hist'
    }

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_lgb    = np.zeros(len(X_train))
    oof_xgb    = np.zeros(len(X_train))
    oof_cat    = np.zeros(len(X_train))
    pred_lgb   = np.zeros(len(X_test))
    pred_xgb   = np.zeros(len(X_test))
    pred_cat   = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=5000)
        m_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        oof_lgb[val_idx]  = m_lgb.predict_proba(X_val)[:, 1]
        pred_lgb          += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

        # XGBoost
        m_xgb = xgb.XGBClassifier(
            **xgb_params, n_estimators=5000, early_stopping_rounds=200
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx]  = m_xgb.predict_proba(X_val)[:, 1]
        pred_xgb          += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost
        m_cat = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.02,
            depth=7,
            eval_metric='AUC',
            random_seed=seed,
            verbose=False,
            early_stopping_rounds=200,
            task_type='CPU'
        )
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx]  = m_cat.predict_proba(X_val)[:, 1]
        pred_cat          += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

        print(
            f'  Fold {fold+1}'
            f'  LGB: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}'
            f'  XGB: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}'
            f'  CAT: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}'
        )

    cv_lgb = roc_auc_score(y_train, oof_lgb)
    cv_xgb = roc_auc_score(y_train, oof_xgb)
    cv_cat = roc_auc_score(y_train, oof_cat)
    print(f'  LGB CV: {cv_lgb:.5f}  XGB CV: {cv_xgb:.5f}  CAT CV: {cv_cat:.5f}')

    # CV AUC 기반 가중 앙상블
    cvs     = np.array([cv_lgb, cv_xgb, cv_cat])
    weights = cvs / cvs.sum()
    pred    = (
        weights[0] * pred_lgb
        + weights[1] * pred_xgb
        + weights[2] * pred_cat
    )
    oof     = (
        weights[0] * oof_lgb
        + weights[1] * oof_xgb
        + weights[2] * oof_cat
    )
    cv_ens  = roc_auc_score(y_train, oof)
    print(f'  앙상블 CV: {cv_ens:.5f}')
    return pred, cv_ens


# ====================================================
# 6. 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v9 - Seed Ensemble)')
    print(f'  SEEDS: {SEEDS}')
    print('=' * 55)

    train_raw, test_raw, sub = load_data()

    # Optuna는 SEED=42 기준으로 한 번만 실행
    best_params = None
    if USE_OPTUNA:
        print('\n' + '=' * 55)
        print('[2] Optuna 튜닝 (SEED=42 기준 1회만 실행)')
        print('=' * 55)
        X_tmp, y_tmp, _, _ = preprocess(train_raw, test_raw, seed=42)
        best_params = optuna_lgb(X_tmp, y_tmp, n_trials=OPTUNA_TRIALS)

    # 각 시드별 학습
    all_preds = []
    all_cvs   = []

    for seed in SEEDS:
        X_train, y_train, X_test, _ = preprocess(train_raw, test_raw, seed=seed)
        pred, cv = train_one_seed(X_train, y_train, X_test, seed, best_params)
        all_preds.append(pred)
        all_cvs.append(cv)
        print(f'\n  ✅ SEED={seed} 완료  CV AUC: {cv:.5f}')

    # 시드 평균 앙상블
    final_pred = np.mean(all_preds, axis=0)

    print('\n' + '=' * 55)
    print('[마지막] Seed Ensemble 결과')
    print('=' * 55)
    for seed, cv in zip(SEEDS, all_cvs):
        print(f'  SEED={seed}  CV AUC: {cv:.5f}')
    print(f'  평균 CV AUC  : {np.mean(all_cvs):.5f}')

    # 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp  = datetime.now().strftime('%m%d_%H%M')
    mean_cv    = np.mean(all_cvs)
    auc_str    = f'{mean_cv:.5f}'.replace('.', 'p')
    filename   = f'{SAVE_PATH}submission_{timestamp}_auc{auc_str}.csv'

    sub['probability'] = final_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')