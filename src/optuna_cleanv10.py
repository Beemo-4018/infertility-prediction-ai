# -*- coding: utf-8 -*-
"""
====================================================
optuna_vclean10.py  -  LightGBM Optuna 하이퍼파라미터 튜닝
베이스: v_clean10 피처 26개
목표  : CV 0.73841 → 0.740+ 돌파
실행  : python src/optuna_vclean10.py

[전략]
  - LightGBM만 튜닝 (속도 빠름, 보통 XGB/Cat보다 Optuna 효과 큼)
  - n_trials=50 (약 30~40분 소요 예상)
  - 최적 파라미터로 최종 앙상블 제출파일까지 자동 생성
====================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ====================================================
# 설정값
# ====================================================
SEED      = 42
N_FOLDS   = 5
TARGET    = '임신 성공 여부'
DATA_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
N_TRIALS  = 50   # 시간 여유 있으면 100으로 늘려도 됨

# ====================================================
# 피처 목록 (v_clean10 동일)
# ====================================================
CLEAN_FEATURES = [
    '나이_수치',
    '과거_임신성공률',
    '시술_ICSI',
    '배란 자극 여부',
    '이식된 배아 수',
    '배아_이식비율',
    '미세주입_성공률',
    '배아_저장비율',
    '배아_활용률',
    '총_불임원인_수',
    '불명확_단독원인',
    '남성_불임원인_수',
    '여성_불임원인_수',
    '총 시술 횟수_num',
    'failure_streak',
    '출산_경험',
    'IVF_경험',
    'DI_경험',
    '시술시기코드_성공률',
    '시술시기코드_시술건수',
    '클리닉_집중도',
    'IVF시술_여부',
    '혼합_이식_간격',
    '해동_이식_간격',
    '남성요인_ICSI매칭',
    # 파트너정자_비율 제거 (중요도 0)
]


# ====================================================
# 1. 데이터 로드 & 전처리 (v_clean10 동일)
# ====================================================
def load_and_preprocess():
    print('=' * 55)
    print('[1] 데이터 로드 & 전처리')
    print('=' * 55)

    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    sub   = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    print(f'  train: {train.shape}, test: {test.shape}')

    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
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

    for df in [train, test]:
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)
        for col in count_cols:
            df[col + '_num'] = df[col].apply(parse_count)
        df['과거_임신성공률'] = (
            df['총 임신 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )
        df['출산_경험'] = (df['총 출산 횟수_num'] > 0).astype(int)

        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']    = 시술유형.str.contains('ICSI', na=False).astype(int)
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)

        df['배아_이식비율'] = (
            df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        )
        df['배아_저장비율'] = (
            df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['배아_활용률'] = (
            (df['이식된 배아 수'] + df['저장된 배아 수'])
            / (df['총 생성 배아 수'] + 1e-6)
        )

        male_cause_cols = [
            '불임 원인 - 남성 요인', '불임 원인 - 정자 농도',
            '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
            '불임 원인 - 정자 형태'
        ]
        female_cause_cols = [
            '불임 원인 - 난관 질환', '불임 원인 - 배란 장애',
            '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제',
            '불임 원인 - 자궁내막증'
        ]
        all_cause_cols = male_cause_cols + female_cause_cols + [
            '남성 주 불임 원인', '남성 부 불임 원인',
            '여성 주 불임 원인', '여성 부 불임 원인',
            '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인'
        ]
        df['총_불임원인_수'] = df[all_cause_cols].sum(axis=1)
        df['불명확_단독원인'] = (
            (df['불명확 불임 원인'] == 1) & (df['총_불임원인_수'] == 1)
        ).astype(int)

        male_flags = [c for c in (
            ['남성 주 불임 원인', '남성 부 불임 원인'] + male_cause_cols
        ) if c in df.columns]
        df['남성_불임원인_수'] = df[male_flags].sum(axis=1)

        female_flags = [c for c in (
            ['여성 주 불임 원인', '여성 부 불임 원인'] + female_cause_cols
        ) if c in df.columns]
        df['여성_불임원인_수'] = df[female_flags].sum(axis=1)

        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)
        df['IVF_경험'] = df['IVF 시술 횟수_num'].fillna(0)
        df['DI_경험']  = df['DI 시술 횟수_num'].fillna(0)

        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']

        df['남성요인_ICSI매칭'] = (
            (df['남성_불임원인_수'] > 0) & (df['시술_ICSI'] == 1)
        ).astype(int)

    # 결측치 (train median)
    num_base = [
        c for c in CLEAN_FEATURES
        if c in train.select_dtypes(include=np.number).columns
    ]
    medians = train[num_base].median()
    train[num_base] = train[num_base].fillna(medians)
    test[num_base]  = test[num_base].fillna(medians)

    # 클리닉 집계 (K-Fold OOF)
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()

    train['시술시기코드_성공률']  = np.nan
    train['시술시기코드_시술건수'] = np.nan

    skf_c = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for _, (tr_idx, val_idx) in enumerate(skf_c.split(train, train[TARGET])):
        tr_fold = train.iloc[tr_idx]
        agg = (
            tr_fold.groupby(clinic_col)[TARGET]
            .agg(['mean', 'count'])
            .reset_index()
        )
        agg.columns = [clinic_col, '_mean', '_count']
        val_map = train.iloc[val_idx][[clinic_col]].merge(
            agg, on=clinic_col, how='left'
        )
        train.loc[train.index[val_idx], '시술시기코드_성공률'] = (
            val_map['_mean'].values
        )
        train.loc[train.index[val_idx], '시술시기코드_시술건수'] = (
            val_map['_count'].values
        )

    train['시술시기코드_성공률']  = train['시술시기코드_성공률'].fillna(global_mean)
    train['시술시기코드_시술건수'] = train['시술시기코드_시술건수'].fillna(0)

    agg_all = (
        train.groupby(clinic_col)[TARGET]
        .agg(['mean', 'count'])
        .reset_index()
    )
    agg_all.columns = [clinic_col, '_mean', '_count']
    test = test.merge(agg_all, on=clinic_col, how='left')
    test['시술시기코드_성공률']  = test['_mean'].fillna(global_mean)
    test['시술시기코드_시술건수'] = test['_count'].fillna(0)
    test.drop(columns=['_mean', '_count'], inplace=True, errors='ignore')

    clinic_size = (
        train.groupby(clinic_col).size().reset_index(name='클리닉_집중도')
    )
    train = train.merge(clinic_size, on=clinic_col, how='left')
    test  = test.merge(clinic_size, on=clinic_col, how='left')
    train['클리닉_집중도'] = train['클리닉_집중도'].fillna(0)
    test['클리닉_집중도']  = test['클리닉_집중도'].fillna(0)

    feature_cols = [f for f in CLEAN_FEATURES if f in train.columns]
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    print(f'  피처 수: {len(feature_cols)}개  (파트너정자_비율 제거됨)')
    print(f'  결측치: {X_train.isnull().sum().sum()}')
    return X_train, y_train, X_test, sub


# ====================================================
# 2. Optuna Objective
# ====================================================
def objective(trial, X, y):
    params = {
        'objective'        : 'binary',
        'metric'           : 'auc',
        'verbosity'        : -1,
        'n_jobs'           : -1,
        'random_state'     : SEED,
        # ── 탐색 공간 ──────────────────────────────────
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves'       : trial.suggest_int('num_leaves', 31, 255),
        'max_depth'        : trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'n_estimators'     : 3000,
    }

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ]
        )
        oof[val_idx] = m.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof)


# ====================================================
# 3. 최적 파라미터로 최종 학습 & 제출파일 생성
# ====================================================
def train_final(X_train, y_train, X_test, best_params, sub):
    print('\n' + '=' * 55)
    print('[3] 최적 파라미터로 최종 학습')
    print('=' * 55)

    params = {
        'objective'   : 'binary',
        'metric'      : 'auc',
        'verbosity'   : -1,
        'n_jobs'      : -1,
        'random_state': SEED,
        'n_estimators': 3000,
        **best_params,
    }

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ]
        )
        oof[val_idx] = m.predict_proba(X_val)[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'\n  최종 CV AUC : {cv:.5f}')

    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    auc_str   = f'{cv:.5f}'.replace('.', 'p')
    filename  = f'{SAVE_PATH}submission_{timestamp}_vclean10_optuna_auc{auc_str}.csv'

    sub['probability'] = pred
    sub.to_csv(filename, index=False)
    print(f'  저장 완료 : {filename}')

    return cv, pred, filename


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  v_clean10 Optuna 튜닝')
    print(f'  n_trials = {N_TRIALS}')
    print('=' * 55)

    X_train, y_train, X_test, sub = load_and_preprocess()

    # ── Optuna 탐색 ──────────────────────────────────────
    print(f'\n[2] Optuna 탐색 시작 (trials={N_TRIALS})')
    print('=' * 55)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f'\n  Best CV AUC : {study.best_value:.5f}')
    print(f'  Best params :')
    for k, v in study.best_params.items():
        print(f'    {k}: {v}')

    # ── 최종 학습 ─────────────────────────────────────────
    cv, pred, saved_file = train_final(
        X_train, y_train, X_test, study.best_params, sub
    )

    print('\n' + '=' * 55)
    print('  결과 요약')
    print('=' * 55)
    print(f'  v_clean10 기본  CV : 0.73841')
    print(f'  v_clean10 Optuna CV: {cv:.5f}  (Δ {cv-0.73841:+.5f})')
    print(f'  저장 파일 : {saved_file}')
    print('\n  ✅ 내일 제출 전략:')
    print('     1번) 이 파일 단독 제출 → LB 실측')
    print('     2번) LB 확인 후 best(0.74205)와 블렌딩 비율 결정')
    print('     3번) 승부수 블렌딩 제출')