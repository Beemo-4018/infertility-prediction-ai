# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v12 (Group KFold + Cycle Progression)
평가 지표  : ROC-AUC
실행 방법  : python src/train_v12.py

[v12 변경사항 - v11 대비]
    1. Group KFold (클리닉 기준) - 핵심 변경
       - 기존 StratifiedKFold → StratifiedGroupKFold
       - group = 시술 시기 코드 (클리닉)
       - 같은 클리닉이 train/valid에 동시 존재하는 문제 해결
       - CV가 살짝 내려가지만 LB는 올라가는 패턴 기대
    2. Cycle Progression 피처 추가
       - failure_streak: 시술횟수 - 임신횟수 (연속 실패 횟수)
       - cycle_progress: (임신횟수+1) / (시술횟수+1)
       - recent_success_flag: 임신 경험 있고 시술 횟수 적은 경우
    3. Bayesian smoothing alpha 50으로 상향 (기존 20)....
       - 클리닉 샘플 수 차이에 의한 노이즈 감소

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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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
SEED          = 42
N_FOLDS       = 5
TARGET        = '임신 성공 여부'
DATA_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
USE_OPTUNA    = True
OPTUNA_TRIALS = 50
SMOOTH_ALPHA  = 50    # v12: 20 → 50 (Bayesian smoothing 강화)


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
def target_encode(train, test, col, target, n_splits=5, smooth=None):
    if smooth is None:
        smooth = SMOOTH_ALPHA
    global_mean = train[target].mean()
    train_enc   = np.zeros(len(train))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

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
# 3. 전처리 & 피처 엔지니어링 (v12)
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 피처 엔지니어링 (v12)')
    print('=' * 55)

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

    # ── 파생 피처 ───────────────────────────────────────────────
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
        df['클리닉_집중도'] = (
            df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )

        # ── [v12 NEW] Cycle Progression 피처 ─────────────────────
        # 연속 실패 횟수 (높을수록 어려운 케이스)
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)

        # 임신 성공 진행률 (높을수록 성공 경험 많음)
        df['cycle_progress'] = (
            (df['총 임신 횟수_num'] + 1) / (df['총 시술 횟수_num'] + 1)
        )

        # IVF 연속 실패
        df['IVF_failure_streak'] = (
            df['IVF 시술 횟수_num'] - df['IVF 임신 횟수_num']
        ).clip(lower=0)

        # 최근 성공 플래그 (임신 경험 있고 시술 횟수 2회 이하)
        df['recent_success_flag'] = (
            (df['총 임신 횟수_num'] > 0) & (df['총 시술 횟수_num'] <= 2)
        ).astype(int)

        # 첫 시술 여부
        df['첫시술_여부'] = (df['총 시술 횟수_num'] <= 1).astype(int)

        # 배아 관련 비율
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

        # ── [v12 NEW] Embryo log transform ───────────────────────
        df['배아_이식비율_log']  = np.log1p(df['배아_이식비율'].clip(0, 3))
        df['배아_저장비율_log']  = np.log1p(df['배아_저장비율'].clip(0, 3))
        df['배아_활용률_log']    = np.log1p(df['배아_활용률'].clip(0, 3))
        df['미세주입_성공률_log'] = np.log1p(df['미세주입_성공률'].clip(0, 3))

        # 시간 간격 피처
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        df['배반포_이식추정'] = (df['혼합_이식_간격'] >= 5).astype(int)

        # 불임 원인 조합
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

        # 결측 여부 피처
        for col in [
            '착상 전 유전 검사 사용 여부', 'PGD 시술 여부',
            'PGS 시술 여부', '난자 해동 경과일', '배아 해동 경과일',
            '임신 시도 또는 마지막 임신 경과 연수'
        ]:
            df[col + '_결측'] = df[col].isnull().astype(int)

        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

        # 교호작용 피처
        df['시술유형_나이조합'] = (
            df['특정 시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        )
        df['시술유형_불임주원인조합'] = (
            df['특정 시술 유형'].astype(str)
            + '_male' + df['남성 주 불임 원인'].astype(str)
            + '_female' + df['여성 주 불임 원인'].astype(str)
        )

        # 도메인 지식 피처
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

        # 시술 유형 분해
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']       = 시술유형.str.contains('ICSI',       na=False).astype(int)
        df['시술_FER']        = 시술유형.str.contains('FER',        na=False).astype(int)
        df['시술_BLASTOCYST'] = 시술유형.str.contains('BLASTOCYST', na=False).astype(int)
        df['시술_AH']         = 시술유형.str.contains('AH',         na=False).astype(int)
        df['시술_IUI']        = 시술유형.str.contains('IUI',        na=False).astype(int)
        df['시술_IVF순수']    = (
            시술유형.str.contains('IVF', na=False) &
            ~시술유형.str.contains('ICSI', na=False)
        ).astype(int)
        df['시술_복합여부']   = 시술유형.str.contains(r'[/:]', na=False).astype(int)
        df['시술_개수']       = (
            시술유형.str.count(r'[/:]') + 1
        ).where(시술유형 != 'nan', 1)
        df['FER_동결배아매칭'] = (
            (df['시술_FER'] == 1) & (df['동결배아_시술'] == 1)
        ).astype(int)
        df['BLASTOCYST_간격매칭'] = (
            (df['시술_BLASTOCYST'] == 1) & (df['혼합_이식_간격'] >= 5)
        ).astype(int)
        df['ICSI_정자문제매칭'] = (
            (df['시술_ICSI'] == 1) & (sperm_issues > 0)
        ).astype(int)

        # 단일 배아 이식
        df['단일배아_고령조합'] = (
            (df['단일 배아 이식 여부'] == 1) & (df['고령_여부'] == 1)
        ).astype(int)

        # IVF/DI 상위 분류
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)

        # 기증자 정자 비율
        df['기증정자_비율'] = (
            df['기증자 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        )
        df['기증정자_사용'] = (df['기증자 정자와 혼합된 난자 수'] > 0).astype(int)

        # 해동/신선 난자
        df['해동난자_비율']    = df['해동 난자 수'] / (df['수집된 신선 난자 수'] + 1e-6)
        df['신선난자_저장비율'] = (
            df['저장된 신선 난자 수'] / (df['수집된 신선 난자 수'] + 1e-6)
        )

    # ── 피처 컬럼 확정 ──────────────────────────────────────────
    drop_cols    = ['ID', TARGET] + count_cols + ['클리닉_나이조합', '클리닉_시술유형조합']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train[feature_cols].select_dtypes(include='object').columns.tolist()

    # ── 결측치 처리 ─────────────────────────────────────────────
    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)
    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # ── 클리닉 집계 피처 (alpha=50으로 강화) ─────────────────────
    print('  클리닉 집계 피처 생성 중... (alpha=50)')
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tr_clinic_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        stats    = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (
            (stats['mean'] * stats['count'] + global_mean * SMOOTH_ALPHA)
            / (stats['count'] + SMOOTH_ALPHA)
        )
        tr_clinic_rate[val_idx] = (
            train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
        )
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (
        (full_stats['mean'] * full_stats['count'] + global_mean * SMOOTH_ALPHA)
        / (full_stats['count'] + SMOOTH_ALPHA)
    )
    train['시술시기코드_성공률'] = tr_clinic_rate
    test['시술시기코드_성공률']  = (
        test[clinic_col].map(full_smooth).fillna(global_mean).values
    )

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
    tr_enc2, te_enc2 = target_encode(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = tr_enc2
    test['클리닉_나이별성공률']  = te_enc2

    train['클리닉_시술유형조합'] = (
        train[clinic_col].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    )
    test['클리닉_시술유형조합'] = (
        test[clinic_col].astype(str) + '_' + test['특정 시술 유형'].astype(str)
    )
    tr_enc3, te_enc3 = target_encode(train, test, '클리닉_시술유형조합', TARGET)
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

    # ── Target Encoding ──────────────────────────────────────────
    print('  Target Encoding 적용 중...')
    te_cols = [
        '시술 시기 코드', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유',
        '난자 출처', '정자 출처', '시술 유형'
    ]
    te_interaction_cols = ['시술유형_나이조합', '시술유형_불임주원인조합']

    for col in te_cols + te_interaction_cols:
        if col in train.columns and col in test.columns:
            tr_enc, te_enc = target_encode(train, test, col, TARGET)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    # ── Label Encoding ───────────────────────────────────────────
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
    groups  = train[clinic_col]   # Group KFold용 그룹

    print(f'  피처 수      : {len(feature_cols)}')
    print(f'  X_train      : {X_train.shape}')
    print(f'  X_test       : {X_test.shape}')
    print(f'  결측치 합계  : {X_train.isnull().sum().sum()}')
    print(f'  클리닉 수    : {groups.nunique()} (Group KFold 기준)')
    return X_train, y_train, X_test, feature_cols, groups


# ====================================================
# 4. Optuna (Group KFold 기준)
# ====================================================
def optuna_lgb(X_train, y_train, groups, n_trials=50):
    print('\n' + '=' * 55)
    print(f'[3] Optuna 튜닝 (LightGBM, {n_trials} trials, GroupKFold)')
    print('=' * 55)

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'objective'        : 'binary',
            'metric'           : 'auc',
            'verbose'          : -1,
            'random_state'     : SEED,
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
        for tr_idx, val_idx in sgkf.split(X_train, y_train, groups):
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

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n  최적 CV AUC  : {study.best_value:.5f}')
    print(f'  최적 파라미터 : {study.best_params}')
    return study.best_params


# ====================================================
# 5-1. LightGBM (Group KFold)
# ====================================================
def train_lgb(X_train, y_train, X_test, groups, best_params=None):
    print('\n' + '=' * 55)
    print('[4-1] LightGBM (StratifiedGroupKFold)')
    print('=' * 55)

    if best_params:
        params = {
            **best_params,
            'objective'   : 'binary',
            'metric'      : 'auc',
            'verbose'     : -1,
            'random_state': SEED,
            'n_jobs'      : -1
        }
        print('  → Optuna 최적 파라미터 사용')
    else:
        params = {
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
            'random_state'     : SEED,
            'n_jobs'           : -1
        }
        print('  → 기본 파라미터 사용')

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        m = lgb.LGBMClassifier(**params, n_estimators=5000)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 5-2. XGBoost (Group KFold)
# ====================================================
def train_xgb(X_train, y_train, X_test, groups):
    print('\n' + '=' * 55)
    print('[4-2] XGBoost (StratifiedGroupKFold)')
    print('=' * 55)

    params = {
        'objective'       : 'binary:logistic',
        'eval_metric'     : 'auc',
        'learning_rate'   : 0.02,
        'max_depth'       : 7,
        'subsample'       : 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha'       : 0.1,
        'reg_lambda'      : 1.0,
        'random_state'    : SEED,
        'n_jobs'          : -1,
        'verbosity'       : 0,
        'tree_method'     : 'hist'
    }

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        m = xgb.XGBClassifier(
            **params, n_estimators=5000, early_stopping_rounds=200
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  XGBoost CV AUC  : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 5-3. CatBoost (Group KFold)
# ====================================================
def train_cat(X_train, y_train, X_test, groups):
    print('\n' + '=' * 55)
    print('[4-3] CatBoost (StratifiedGroupKFold)')
    print('=' * 55)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        m = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.02,
            depth=7,
            eval_metric='AUC',
            random_seed=SEED,
            verbose=False,
            early_stopping_rounds=200,
            task_type='CPU'
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  CatBoost CV AUC : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 6. 앙상블 & 저장
# ====================================================
def ensemble_and_save(y_train, sub, results):
    print('\n' + '=' * 55)
    print('[5] 앙상블 & 제출 파일 저장')
    print('=' * 55)

    oofs  = [r[0] for r in results]
    preds = [r[1] for r in results]
    cvs   = np.array([r[2] for r in results])
    names = ['LightGBM', 'XGBoost', 'CatBoost']

    weights       = cvs / cvs.sum()
    ensemble_oof  = sum(w * o for w, o in zip(weights, oofs))
    ensemble_pred = sum(w * p for w, p in zip(weights, preds))
    ensemble_cv   = roc_auc_score(y_train, ensemble_oof)

    print(f'\n  {"모델":<12} {"CV AUC":>10}  {"weight":>8}')
    print('  ' + '-' * 38)
    for name, cv, w in zip(names, cvs, weights):
        print(f'  {name:<12} {cv:>10.5f}  {w:>8.3f}')
    print('  ' + '-' * 38)
    print(f'  {"앙상블":<12} {ensemble_cv:>10.5f}')

    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    auc_str   = f'{ensemble_cv:.5f}'.replace('.', 'p')
    filename  = f'{SAVE_PATH}submission_{timestamp}_auc{auc_str}.csv'

    sub['probability'] = ensemble_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v12 - GroupKFold)')
    print('=' * 55)

    train, test, sub                              = load_data()
    X_train, y_train, X_test, feature_cols, groups = preprocess(train, test)

    best_params = None
    if USE_OPTUNA:
        best_params = optuna_lgb(X_train, y_train, groups, n_trials=OPTUNA_TRIALS)

    print('\n' + '=' * 55)
    print('[4] 모델 학습')
    print('=' * 55)
    lgb_result = train_lgb(X_train, y_train, X_test, groups, best_params)
    xgb_result = train_xgb(X_train, y_train, X_test, groups)
    cat_result = train_cat(X_train, y_train, X_test, groups)

    ensemble_and_save(y_train, sub, [lgb_result, xgb_result, cat_result])