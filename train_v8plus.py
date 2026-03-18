# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v8_plus_final
평가 지표  : ROC-AUC
실행 방법  : python train_v8plus_final.py

[추가 기능]
  - GPU/CPU 전환 스위치 (USE_GPU = True/False)
  - 피처 중요도 출력 및 CSV 저장
  - 다양한 앙상블 비중 비교 (최고 조합 자동 선택)
  - OOF 예측값 저장 (블렌딩 실험용)
  - 모델 간 상관관계 출력
====================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import rankdata, pearsonr
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
# ⚙️ 설정값 — 여기만 바꾸면 됩니다
# ====================================================
USE_GPU       = True        # ← True: GPU 사용 / False: CPU 사용
SEED          = 42
N_FOLDS       = 5
TARGET        = '임신 성공 여부'
DATA_PATH     = './data/raw/'
SAVE_PATH     = './submissions/'
USE_OPTUNA    = True
OPTUNA_TRIALS = 50
VERSION       = 'v8plus'     # 파일명에 사용될 버전명


# ====================================================
# 1. 데이터 로드
# ====================================================
def load_data():
    print('=' * 55)
    print(f'[1] 데이터 로드  (GPU={USE_GPU})')
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
# 2. Target Encoding (K-Fold OOF)
# ====================================================
def target_encode(train, test, col, target, n_splits=5, smooth=20):
    global_mean = train[target].mean()
    train_enc   = np.zeros(len(train))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for tr_idx, val_idx in skf.split(train, train[target]):
        stats    = train.iloc[tr_idx].groupby(col)[target].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * smooth) \
                 / (stats['count'] + smooth)
        train_enc[val_idx] = train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values

    full_stats  = train.groupby(col)[target].agg(['mean', 'count'])
    full_smooth = (full_stats['mean'] * full_stats['count'] + global_mean * smooth) \
                / (full_stats['count'] + smooth)
    test_enc    = test[col].map(full_smooth).fillna(global_mean).values
    return train_enc, test_enc


# ====================================================
# 3. 전처리 & 피처 엔지니어링
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 피처 엔지니어링')
    print('=' * 55)

    train = train.copy()
    test  = test.copy()

    # ── 나이 수치화 ──────────────────────────────────────────
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치']     = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

    # ── 기증자 나이 수치화 ───────────────────────────────────
    donor_age_map = {
        '만20세 이하': 19, '만21-25세': 23, '만26-30세': 28,
        '만31-35세': 33, '만36-40세': 38, '만41-45세': 43,
        '만46-50세': 48, '알 수 없음': -1
    }
    for df in [train, test]:
        df['난자기증자_나이_수치'] = df['난자 기증자 나이'].map(donor_age_map).fillna(-1)
        df['정자기증자_나이_수치'] = df['정자 기증자 나이'].map(donor_age_map).fillna(-1)
        df['난자기증자_있음']      = (df['난자기증자_나이_수치'] > 0).astype(int)
        df['정자기증자_있음']      = (df['정자기증자_나이_수치'] > 0).astype(int)
        df['난자기증자_젊음']      = (
            (df['난자기증자_나이_수치'] > 0) &
            (df['난자기증자_나이_수치'] <= 30)
        ).astype(int)

    # ── 횟수 컬럼 수치화 ─────────────────────────────────────
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수',
        'IVF 시술 횟수', 'DI 시술 횟수',
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]

    def parse_count(val):
        if pd.isna(val): return np.nan
        s = str(val).replace('회', '').replace(' 이상', '').strip()
        try:    return float(s)
        except: return np.nan

    for col in count_cols:
        for df in [train, test]:
            df[col + '_num'] = df[col].apply(parse_count)

    # ── 파생 피처 ────────────────────────────────────────────
    for df in [train, test]:

        # 과거 이력 성공률
        df['과거_임신성공률'] = df['총 임신 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['과거_출산성공률'] = df['총 출산 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['IVF_임신성공률'] = df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['IVF_출산성공률'] = df['IVF 출산 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['DI_임신성공률']  = df['DI 임신 횟수_num']  / (df['DI 시술 횟수_num'] + 1e-6)

        # 시술 경험 플래그
        df['IVF_경험']      = (df['IVF 시술 횟수_num'] > 0).astype(int)
        df['DI_경험']       = (df['DI 시술 횟수_num']  > 0).astype(int)
        df['임신_경험']     = (df['총 임신 횟수_num']   > 0).astype(int)
        df['출산_경험']     = (df['총 출산 횟수_num']   > 0).astype(int)
        df['반복시술_여부'] = (df['총 시술 횟수_num']   >= 3).astype(int)
        df['클리닉_집중도'] = df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        df['IVF_비율']      = df['IVF 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)

        # 배아 관련 비율
        df['배아_이식비율']   = df['이식된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율']   = df['저장된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_활용률']     = (df['이식된 배아 수'] + df['저장된 배아 수']) / (df['총 생성 배아 수'] + 1e-6)
        df['미세주입_성공률'] = df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        df['미세주입_이식률'] = df['미세주입 배아 이식 수'] / (df['미세주입에서 생성된 배아 수'] + 1e-6)
        df['난자_수정률']     = df['혼합된 난자 수']   / (df['수집된 신선 난자 수'] + 1e-6)
        df['파트너정자_비율'] = df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        df['배아_생성률']     = df['총 생성 배아 수']  / (df['혼합된 난자 수'] + 1e-6)

        # v8 추가 피처
        df['미세주입후_저장비율'] = df['미세주입 후 저장된 배아 수'] / (df['미세주입에서 생성된 배아 수'] + 1e-6)
        df['해동난자_있음']       = (df['해동 난자 수'] > 0).astype(int)
        df['해동난자_비율']       = df['해동 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        df['신선난자_저장됨']     = (df['저장된 신선 난자 수'] > 0).astype(int)
        df['신선난자_저장수']     = df['저장된 신선 난자 수'].fillna(0)

        # v8_plus 추가 피처
        df['failure_streak'] = (df['총 시술 횟수_num'] - df['총 임신 횟수_num']).clip(lower=0)
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']    = 시술유형.str.contains('ICSI', na=False).astype(int)
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)

        # 시간 간격
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        df['배반포_이식추정'] = (df['혼합_이식_간격'] >= 5).astype(int)

        # 불임 원인
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

        # 결측 여부
        for col in ['착상 전 유전 검사 사용 여부', 'PGD 시술 여부',
                    'PGS 시술 여부', '난자 해동 경과일', '배아 해동 경과일',
                    '임신 시도 또는 마지막 임신 경과 연수']:
            df[col + '_결측'] = df[col].isnull().astype(int)

        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

        # 교호작용
        df['시술유형_나이조합']      = df['특정 시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        df['시술유형_불임주원인조합'] = df['특정 시술 유형'].astype(str) + '_male' \
                                     + df['남성 주 불임 원인'].astype(str) \
                                     + '_female' + df['여성 주 불임 원인'].astype(str)

        # 도메인 피처
        df['남성요인_ICSI매칭'] = (
            (df['불임 원인 - 남성 요인'] == 1) &
            (df['특정 시술 유형'].astype(str).str.contains('ICSI'))
        ).astype(int)
        df['배란장애_자극매칭'] = (
            (df['불임 원인 - 배란 장애'] == 1) & (df['배란 자극 여부'] == 1)
        ).astype(int)
        df['고령_동결배아조합'] = (
            (df['고령_여부'] == 1) & (df['동결배아_시술'] == 1)
        ).astype(int)
        sperm_issues = df[['불임 원인 - 정자 농도', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']].sum(axis=1)
        df['정자문제_파트너정자'] = (
            (sperm_issues > 0) & (df['파트너정자_비율'] > 0.5)
        ).astype(int)
        df['초고령_반복시술'] = (
            (df['초고령_여부'] == 1) & (df['반복시술_여부'] == 1)
        ).astype(int)

        # 기증 관련
        df['기증난자_고령조합']   = ((df['난자기증자_있음'] == 1) & (df['고령_여부'] == 1)).astype(int)
        df['기증난자_젊음_고령모'] = ((df['난자기증자_젊음'] == 1) & (df['고령_여부'] == 1)).astype(int)
        df['기증배아_사용']       = df['기증 배아 사용 여부'].fillna(0)
        df['대리모_여부_f']       = df['대리모 여부'].fillna(0)

    # ── 피처 컬럼 확정 ───────────────────────────────────────
    drop_cols    = ['ID', TARGET] + count_cols + ['클리닉_나이조합', '클리닉_시술유형조합']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train[feature_cols].select_dtypes(include='object').columns.tolist()

    medians         = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)
    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # ── 클리닉 집계 피처 ─────────────────────────────────────
    print('  클리닉 집계 피처 생성 중...')
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tr_clinic_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        stats    = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * 20) / (stats['count'] + 20)
        tr_clinic_rate[val_idx] = train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (full_stats['mean'] * full_stats['count'] + global_mean * 20) / (full_stats['count'] + 20)
    train['시술시기코드_성공률'] = tr_clinic_rate
    test['시술시기코드_성공률']  = test[clinic_col].map(full_smooth).fillna(global_mean).values

    tr_clinic_cnt = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        cnt_map = train.iloc[tr_idx].groupby(clinic_col).size()
        tr_clinic_cnt[val_idx] = train.iloc[val_idx][clinic_col].map(cnt_map).fillna(1).values
    full_cnt_map = train.groupby(clinic_col).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_clinic_cnt)
    test['시술시기코드_시술건수']  = np.log1p(test[clinic_col].map(full_cnt_map).fillna(1).values)

    train['시술시기코드_성공률편차'] = train['시술시기코드_성공률'] - global_mean
    test['시술시기코드_성공률편차']  = test['시술시기코드_성공률']  - global_mean

    train['클리닉_나이조합'] = train[clinic_col].astype(str) + '_' + train['시술 당시 나이'].astype(str)
    test['클리닉_나이조합']  = test[clinic_col].astype(str)  + '_' + test['시술 당시 나이'].astype(str)
    tr_enc2, te_enc2 = target_encode(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = tr_enc2
    test['클리닉_나이별성공률']  = te_enc2

    train['클리닉_시술유형조합'] = train[clinic_col].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    test['클리닉_시술유형조합']  = test[clinic_col].astype(str)  + '_' + test['특정 시술 유형'].astype(str)
    tr_enc3, te_enc3 = target_encode(train, test, '클리닉_시술유형조합', TARGET)
    train['클리닉_시술유형별성공률'] = tr_enc3
    test['클리닉_시술유형별성공률']  = te_enc3

    tr_emb_mean = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        emb_map = train.iloc[tr_idx].groupby(clinic_col)['이식된 배아 수'].mean()
        tr_emb_mean[val_idx] = train.iloc[val_idx][clinic_col].map(emb_map).fillna(train['이식된 배아 수'].mean()).values
    full_emb_map = train.groupby(clinic_col)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb_mean
    test['시술시기코드_배아이식수평균']  = test[clinic_col].map(full_emb_map).fillna(train['이식된 배아 수'].mean()).values

    train['클리닉대비_개인성공률차이'] = train['시술시기코드_성공률'] - train['과거_임신성공률']
    test['클리닉대비_개인성공률차이']  = test['시술시기코드_성공률']  - test['과거_임신성공률']

    # ── Target Encoding ──────────────────────────────────────
    te_cols = ['시술 시기 코드', '특정 시술 유형', '배란 유도 유형',
               '배아 생성 주요 이유', '난자 출처', '정자 출처',
               '시술 유형', '난자 기증자 나이', '정자 기증자 나이']
    te_interaction_cols = ['시술유형_나이조합', '시술유형_불임주원인조합']

    print('  Target Encoding 적용 중...')
    for col in te_cols + te_interaction_cols:
        if col in train.columns and col in test.columns:
            tr_enc, te_enc = target_encode(train, test, col, TARGET)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    # ── Label Encoding ───────────────────────────────────────
    feature_cols = [c for c in train.columns if c not in drop_cols]
    cat_cols     = train[feature_cols].select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col].astype(str).tolist() + ['Unknown'])
        known      = set(le.classes_)
        test[col]  = test[col].astype(str).apply(lambda x: x if x in known else 'Unknown')
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    feature_cols = [c for c in train.columns if c not in drop_cols]
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    print(f'  피처 수      : {len(feature_cols)}')
    print(f'  X_train      : {X_train.shape}')
    print(f'  결측치 합계  : {X_train.isnull().sum().sum()}')
    return X_train, y_train, X_test, feature_cols


# ====================================================
# 4. Optuna (LGB만)
# ====================================================
def optuna_lgb(X_train, y_train, n_trials=50):
    print('\n' + '=' * 55)
    print(f'[3] Optuna 튜닝 (LightGBM, {n_trials} trials, GPU={USE_GPU})')
    print('=' * 55)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc',
            'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
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
        if USE_GPU:
            params['device'] = 'gpu'

        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = lgb.LGBMClassifier(**params, n_estimators=2000)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'\n  최적 CV AUC  : {study.best_value:.5f}')
    print(f'  최적 파라미터 : {study.best_params}')
    return study.best_params


# ====================================================
# 5. 모델 학습
# ====================================================
def train_lgb(X_train, y_train, X_test, feature_cols, best_params=None):
    print('\n' + '=' * 55)
    print(f'[4-1] LightGBM  (GPU={USE_GPU})')
    print('=' * 55)

    if best_params:
        params = {**best_params, 'objective': 'binary', 'metric': 'auc',
                  'verbose': -1, 'random_state': SEED, 'n_jobs': -1}
    else:
        params = {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': -1,
            'min_child_samples': 20, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'verbose': -1, 'random_state': SEED, 'n_jobs': -1
        }
    if USE_GPU:
        params['device'] = 'gpu'

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof          = np.zeros(len(X_train))
    pred         = np.zeros(len(X_test))
    importance_list = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = lgb.LGBMClassifier(**params, n_estimators=5000)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        importance_list.append(m.feature_importances_)
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
              f'  (best_iter: {m.best_iteration_})')

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')

    # 피처 중요도 저장
    importance = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': np.mean(importance_list, axis=0)
    }).sort_values('importance', ascending=False)

    return oof, pred, cv, importance


def train_xgb(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print(f'[4-2] XGBoost  (GPU={USE_GPU})')
    print('=' * 55)

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'learning_rate': 0.02, 'max_depth': 7,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': SEED, 'verbosity': 0,
        'tree_method': 'hist',
    }
    if USE_GPU:
        params['device'] = 'cuda'
    else:
        params['n_jobs'] = -1

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = xgb.XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=200)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
              f'  (best_iter: {m.best_iteration})')

    cv = roc_auc_score(y_train, oof)
    print(f'  XGBoost CV AUC  : {cv:.5f}')
    return oof, pred, cv


def train_cat(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print(f'[4-3] CatBoost  (GPU={USE_GPU})')
    print('=' * 55)

    task_type = 'GPU' if USE_GPU else 'CPU'

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.02, depth=7,
            eval_metric='AUC', random_seed=SEED,
            verbose=False, early_stopping_rounds=200,
            task_type=task_type
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
              f'  (best_iter: {m.best_iteration_})')

    cv = roc_auc_score(y_train, oof)
    print(f'  CatBoost CV AUC : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 6. 앙상블 비교 & 최고 조합 선택
# ====================================================
def ensemble_and_save(y_train, sub, oof_lgb, oof_xgb, oof_cat,
                      pred_lgb, pred_xgb, pred_cat,
                      cv_lgb, cv_xgb, cv_cat):
    print('\n' + '=' * 55)
    print('[5] 앙상블 비교 & 저장')
    print('=' * 55)

    # ── 단일 모델 점수 ───────────────────────────────────────
    print(f'\n  단일 모델 CV AUC:')
    print(f'    LGB : {cv_lgb:.5f}')
    print(f'    XGB : {cv_xgb:.5f}')
    print(f'    CAT : {cv_cat:.5f}')

    # ── 모델 간 상관관계 ─────────────────────────────────────
    print(f'\n  모델 간 상관관계 (낮을수록 앙상블 효과 큼):')
    corr_lx = pearsonr(oof_lgb, oof_xgb)[0]
    corr_lc = pearsonr(oof_lgb, oof_cat)[0]
    corr_xc = pearsonr(oof_xgb, oof_cat)[0]
    print(f'    LGB-XGB: {corr_lx:.4f}{"  ⚠️ 높음" if corr_lx > 0.98 else ""}')
    print(f'    LGB-CAT: {corr_lc:.4f}{"  ⚠️ 높음" if corr_lc > 0.98 else ""}')
    print(f'    XGB-CAT: {corr_xc:.4f}{"  ⚠️ 높음" if corr_xc > 0.98 else ""}')

    # ── 다양한 앙상블 조합 비교 ──────────────────────────────
    print(f'\n  앙상블 조합 비교:')

    results = {}
    cvs = np.array([cv_lgb, cv_xgb, cv_cat])
    w_perf = cvs / cvs.sum()

    # Rank 변환
    r_lgb = rankdata(oof_lgb) / len(oof_lgb)
    r_xgb = rankdata(oof_xgb) / len(oof_xgb)
    r_cat = rankdata(oof_cat) / len(oof_cat)
    rt_lgb = rankdata(pred_lgb) / len(pred_lgb)
    rt_xgb = rankdata(pred_xgb) / len(pred_xgb)
    rt_cat = rankdata(pred_cat) / len(pred_cat)

    # 단순 평균
    results['단순평균'] = (
        roc_auc_score(y_train, (oof_lgb+oof_xgb+oof_cat)/3),
        (pred_lgb+pred_xgb+pred_cat)/3
    )
    # 성능 가중
    results[f'성능가중({w_perf[0]:.2f}/{w_perf[1]:.2f}/{w_perf[2]:.2f})'] = (
        roc_auc_score(y_train, w_perf[0]*oof_lgb+w_perf[1]*oof_xgb+w_perf[2]*oof_cat),
        w_perf[0]*pred_lgb+w_perf[1]*pred_xgb+w_perf[2]*pred_cat
    )
    # LGB 강조
    results['LGB강조(0.5/0.25/0.25)'] = (
        roc_auc_score(y_train, 0.5*oof_lgb+0.25*oof_xgb+0.25*oof_cat),
        0.5*pred_lgb+0.25*pred_xgb+0.25*pred_cat
    )
    # LGB+CAT 2개
    results['LGB+CAT(0.6/0.4)'] = (
        roc_auc_score(y_train, 0.6*oof_lgb+0.4*oof_cat),
        0.6*pred_lgb+0.4*pred_cat
    )
    results['LGB+CAT(0.5/0.5)'] = (
        roc_auc_score(y_train, 0.5*oof_lgb+0.5*oof_cat),
        0.5*pred_lgb+0.5*pred_cat
    )
    # LGB 단독
    results['LGB단독'] = (cv_lgb, pred_lgb)

    # Rank 버전
    results['Rank단순평균'] = (
        roc_auc_score(y_train, (r_lgb+r_xgb+r_cat)/3),
        (rt_lgb+rt_xgb+rt_cat)/3
    )
    results['Rank성능가중'] = (
        roc_auc_score(y_train, w_perf[0]*r_lgb+w_perf[1]*r_xgb+w_perf[2]*r_cat),
        w_perf[0]*rt_lgb+w_perf[1]*rt_xgb+w_perf[2]*rt_cat
    )
    results['Rank LGB강조(0.5/0.25/0.25)'] = (
        roc_auc_score(y_train, 0.5*r_lgb+0.25*r_xgb+0.25*r_cat),
        0.5*rt_lgb+0.25*rt_xgb+0.25*rt_cat
    )
    results['Rank LGB+CAT(0.6/0.4)'] = (
        roc_auc_score(y_train, 0.6*r_lgb+0.4*r_cat),
        0.6*rt_lgb+0.4*rt_cat
    )
    results['Rank LGB+CAT(0.5/0.5)'] = (
        roc_auc_score(y_train, 0.5*r_lgb+0.5*r_cat),
        0.5*rt_lgb+0.5*rt_cat
    )

    # 결과 출력 (내림차순)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    best_name, (best_cv, best_pred) = sorted_results[0]

    for name, (auc, _) in sorted_results:
        marker = '  ⭐ BEST' if name == best_name else ''
        print(f'    {auc:.5f}  {name}{marker}')

    # ── OOF 저장 (블렌딩 실험용) ─────────────────────────────
    os.makedirs(SAVE_PATH + 'oof/', exist_ok=True)
    gpu_tag = 'gpu' if USE_GPU else 'cpu'
    np.save(f'{SAVE_PATH}oof/oof_lgb_{VERSION}_{gpu_tag}.npy',  oof_lgb)
    np.save(f'{SAVE_PATH}oof/oof_xgb_{VERSION}_{gpu_tag}.npy',  oof_xgb)
    np.save(f'{SAVE_PATH}oof/oof_cat_{VERSION}_{gpu_tag}.npy',  oof_cat)
    np.save(f'{SAVE_PATH}oof/pred_lgb_{VERSION}_{gpu_tag}.npy', pred_lgb)
    np.save(f'{SAVE_PATH}oof/pred_xgb_{VERSION}_{gpu_tag}.npy', pred_xgb)
    np.save(f'{SAVE_PATH}oof/pred_cat_{VERSION}_{gpu_tag}.npy', pred_cat)
    print(f'\n  💾 OOF 저장 완료: {SAVE_PATH}oof/')

    # ── 최고 조합 제출 파일 저장 ─────────────────────────────
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    filename  = f'{SAVE_PATH}sub_{VERSION}_{gpu_tag}_{timestamp}_cv{best_cv:.5f}.csv'
    sub['probability'] = best_pred
    sub.to_csv(filename, index=False)
    print(f'  ✨ 최고조합 저장: {filename}')
    print(f'  🏆 최고 조합: {best_name}  CV: {best_cv:.5f}')

    return best_cv


# ====================================================
# 7. 피처 중요도 출력 & 저장
# ====================================================
def print_feature_importance(importance, top_n=30):
    print('\n' + '=' * 55)
    print(f'[6] 피처 중요도 (LGB 5-Fold 평균, Top {top_n})')
    print('=' * 55)
    print(f'\n  총 피처 수: {len(importance)}')
    print(f'\n  Top {top_n}:')
    print(importance.head(top_n).to_string(index=False))
    print(f'\n  Bottom 15 (제거 후보):')
    print(importance.tail(15).to_string(index=False))

    zero = importance[importance['importance'] == 0]
    print(f'\n  중요도 0인 피처: {len(zero)}개')
    if len(zero) > 0:
        print(f'  {zero["feature"].tolist()}')

    gpu_tag = 'gpu' if USE_GPU else 'cpu'
    fname   = f'{SAVE_PATH}feature_importance_{VERSION}_{gpu_tag}.csv'
    importance.to_csv(fname, index=False)
    print(f'\n  ✅ 피처 중요도 저장: {fname}')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    gpu_tag = 'GPU' if USE_GPU else 'CPU'
    print('=' * 55)
    print(f'  난임 환자 임신 성공 여부 예측 ({VERSION} / {gpu_tag})')
    print('=' * 55)

    train, test, sub                          = load_data()
    X_train, y_train, X_test, feature_cols    = preprocess(train, test)

    best_params = None
    if USE_OPTUNA:
        best_params = optuna_lgb(X_train, y_train, n_trials=OPTUNA_TRIALS)

    oof_lgb, pred_lgb, cv_lgb, importance = train_lgb(X_train, y_train, X_test, feature_cols, best_params)
    oof_xgb, pred_xgb, cv_xgb            = train_xgb(X_train, y_train, X_test)
    oof_cat, pred_cat, cv_cat            = train_cat(X_train, y_train, X_test)

    ensemble_and_save(
        y_train.values, sub,
        oof_lgb, oof_xgb, oof_cat,
        pred_lgb, pred_xgb, pred_cat,
        cv_lgb, cv_xgb, cv_cat
    )

    print_feature_importance(importance, top_n=30)