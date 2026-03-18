# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v_clean11 (v_clean10 + v5 통합 최강)
평가 지표  : ROC-AUC
실행 방법  : python src/train_vclean11.py

[v_clean10 대비 추가]
    v5에서 발굴한 핵심 신호:
    1. 실제 파트너정자_비율  (파트너 정자와 혼합된 난자 수 / 혼합된 난자 수)
    2. 난자_수정률           (혼합된 난자 수 / 수집된 신선 난자 수)
    3. 채취_이식_간격        (배아 이식 - 난자 채취)
    4. 채취_혼합_간격        (난자 혼합 - 난자 채취)
    5. Smoothed TE (smooth=20) → 클리닉 성공률 안정화
    6. 클리닉_시술유형별성공률 (클리닉 × 특정시술유형 조합 TE)
    7. 시술시기코드_성공률편차 (클리닉 - 전체 평균)
    8. 시술시기코드_배아이식수평균
    9. 시술시기코드_시술건수  (log1p 스케일)
    10. TE: 특정 시술 유형, 정자 출처, 난자 출처, 배란 유도 유형, 배아 생성 주요 이유
    11. 동결배아_시술, 배란장애_자극매칭, 고령_동결배아조합
    12. 임신_경험, 반복시술_여부, IVF_임신성공률, 과거_출산성공률
    feature_search_vclean10 양수 피처:
    13. 클리닉내_시술비율 +0.00022
    14. 긴_배양_여부     +0.00014
    15. 출산_성공률      +0.00010
    16. 나이x배아활용률  +0.00010
    17. 해동_당일이식    +0.00009
    18. 잉여배아_유무    +0.00009
    19. 신선배아_이식여부 +0.00008

[Data Leakage 방지 원칙]
    1. 결측치 median은 train 기준
    2. 파생 변수는 행(row) 내 연산만
    3. Target/집계 인코딩은 K-Fold OOF
    4. test는 train 전체 통계만 적용
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

warnings.filterwarnings('ignore')

# ====================================================
# 설정값
# ====================================================
SEED      = 42
N_FOLDS   = 5
TARGET    = '임신 성공 여부'
DATA_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'


# ====================================================
# Smoothed Target Encoding (v5 방식, smooth=20)
# ====================================================
def target_encode(train, test, col, target, n_splits=5, smooth=20):
    global_mean = train[target].mean()
    train_enc   = np.zeros(len(train))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for tr_idx, val_idx in skf.split(train, train[target]):
        stats    = train.iloc[tr_idx].groupby(col)[target].agg(['mean', 'count'])
        smoothed = (
            stats['mean'] * stats['count'] + global_mean * smooth
        ) / (stats['count'] + smooth)
        train_enc[val_idx] = (
            train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
        )

    full_stats  = train.groupby(col)[target].agg(['mean', 'count'])
    full_smooth = (
        full_stats['mean'] * full_stats['count'] + global_mean * smooth
    ) / (full_stats['count'] + smooth)
    test_enc = test[col].map(full_smooth).fillna(global_mean).values

    return train_enc, test_enc


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
# 2. 전처리 & 피처 엔지니어링
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 피처 엔지니어링 (v_clean11)')
    print('=' * 55)

    train = train.copy()
    test  = test.copy()

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

        # ── A. 나이 ───────────────────────────────────────────
        df['나이_수치']     = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

        # ── B. 횟수 수치화 ────────────────────────────────────
        for col in count_cols:
            df[col + '_num'] = df[col].apply(parse_count)

        # ── C. 과거 이력 ──────────────────────────────────────
        df['과거_임신성공률'] = (
            df['총 임신 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )
        df['과거_출산성공률'] = (
            df['총 출산 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )
        df['IVF_임신성공률'] = (
            df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        )
        df['출산_경험']     = (df['총 출산 횟수_num'] > 0).astype(int)
        df['임신_경험']     = (df['총 임신 횟수_num'] > 0).astype(int)
        df['IVF_경험']      = df['IVF 시술 횟수_num'].fillna(0)
        df['DI_경험']       = df['DI 시술 횟수_num'].fillna(0)
        df['반복시술_여부'] = (df['총 시술 횟수_num'] >= 3).astype(int)
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)

        # ── D. 시술 정보 ──────────────────────────────────────
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']    = 시술유형.str.contains('ICSI', na=False).astype(int)
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)
        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

        # ── E. 배아 품질 ──────────────────────────────────────
        df['배아_이식비율'] = (
            df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['배아_저장비율'] = (
            df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['배아_활용률'] = (
            (df['이식된 배아 수'] + df['저장된 배아 수'])
            / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수']
            / (df['미세주입된 난자 수'] + 1e-6)
        )
        # v5 신규: 실제 파트너정자_비율
        df['파트너정자_비율'] = (
            df['파트너 정자와 혼합된 난자 수']
            / (df['혼합된 난자 수'] + 1e-6)
        )
        # v5 신규: 난자_수정률
        df['난자_수정률'] = (
            df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1e-6)
        )
        # feature_search 신규
        df['신선배아_이식여부'] = (
            (df['저장된 배아 수'] == 0) & (df['이식된 배아 수'] > 0)
        ).astype(int)
        df['잉여배아_유무'] = (
            (df['총 생성 배아 수']
            - df['이식된 배아 수']
            - df['저장된 배아 수']) > 0
        ).astype(int)

        # ── F. 시간 간격 ──────────────────────────────────────
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        # v5 신규: 채취 기반 간격
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        # feature_search 신규
        df['긴_배양_여부']  = (df['혼합_이식_간격'] >= 5).astype(int)
        df['해동_당일이식'] = (df['해동_이식_간격'] == 0).astype(int)

        # ── G. 불임 원인 ──────────────────────────────────────
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

        # ── H. 교호작용 피처 ──────────────────────────────────
        df['남성요인_ICSI매칭'] = (
            (df['남성_불임원인_수'] > 0) & (df['시술_ICSI'] == 1)
        ).astype(int)
        df['배란장애_자극매칭'] = (
            (df['불임 원인 - 배란 장애'] == 1)
            & (df['배란 자극 여부'] == 1)
        ).astype(int)
        df['고령_동결배아조합'] = (
            (df['고령_여부'] == 1) & (df['동결배아_시술'] == 1)
        ).astype(int)
        df['나이x배아활용률'] = df['나이_수치'] * df['배아_활용률']

        # ── I. 클리닉 내 시술 비율 (feature_search 1위) ───────
        df['클리닉내_시술비율'] = (
            df['클리닉 내 총 시술 횟수_num'].fillna(0)
            / (df['총 시술 횟수_num'].fillna(0) + 1e-6)
        )

        # ── J. 교호작용 컬럼 (TE용) ───────────────────────────
        df['클리닉_나이조합'] = (
            df['시술 시기 코드'].astype(str)
            + '_' + df['시술 당시 나이'].astype(str)
        )
        df['클리닉_시술유형조합'] = (
            df['시술 시기 코드'].astype(str)
            + '_' + df['특정 시술 유형'].astype(str)
        )

    # ── 결측치 처리 (train median) ────────────────────────────
    num_cols = train.select_dtypes(include=np.number).columns.tolist()
    excl = [TARGET] + [c + '_num' for c in count_cols]
    num_base = [c for c in num_cols if c not in excl]
    medians = train[num_base].median()
    train[num_base] = train[num_base].fillna(medians)
    test[num_base]  = test[num_base].fillna(medians)

    # ── 클리닉 집계 피처 (K-Fold OOF + Smoothed TE) ───────────
    print('  클리닉 집계 피처 생성 중...')
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_c = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # [1] 클리닉 성공률 (smoothed TE, smooth=20)
    tr_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        stats = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (
            stats['mean'] * stats['count'] + global_mean * 20
        ) / (stats['count'] + 20)
        tr_rate[val_idx] = (
            train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
        )
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (
        full_stats['mean'] * full_stats['count'] + global_mean * 20
    ) / (full_stats['count'] + 20)
    train['시술시기코드_성공률'] = tr_rate
    test['시술시기코드_성공률']  = test[clinic_col].map(full_smooth).fillna(global_mean).values

    # [2] 클리닉 규모 (log1p - v5 방식)
    tr_cnt = np.zeros(len(train))
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        cnt_map = train.iloc[tr_idx].groupby(clinic_col).size()
        tr_cnt[val_idx] = (
            train.iloc[val_idx][clinic_col].map(cnt_map).fillna(1).values
        )
    full_cnt = train.groupby(clinic_col).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_cnt)
    test['시술시기코드_시술건수']  = np.log1p(
        test[clinic_col].map(full_cnt).fillna(1).values
    )

    # [3] 클리닉 성공률 편차 (v5 신규)
    train['시술시기코드_성공률편차'] = train['시술시기코드_성공률'] - global_mean
    test['시술시기코드_성공률편차']  = test['시술시기코드_성공률']  - global_mean

    # [4] 클리닉 × 나이그룹 성공률 (smoothed TE)
    tr_enc, te_enc = target_encode(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = tr_enc
    test['클리닉_나이별성공률']  = te_enc

    # [5] 클리닉 × 시술유형 성공률 (smoothed TE, v5 신규)
    tr_enc2, te_enc2 = target_encode(train, test, '클리닉_시술유형조합', TARGET)
    train['클리닉_시술유형별성공률'] = tr_enc2
    test['클리닉_시술유형별성공률']  = te_enc2

    # [6] 클리닉별 평균 배아 이식 수 (v5 신규)
    tr_emb = np.zeros(len(train))
    emb_global = train['이식된 배아 수'].mean()
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        emb_map = train.iloc[tr_idx].groupby(clinic_col)['이식된 배아 수'].mean()
        tr_emb[val_idx] = (
            train.iloc[val_idx][clinic_col].map(emb_map).fillna(emb_global).values
        )
    full_emb = train.groupby(clinic_col)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb
    test['시술시기코드_배아이식수평균']  = (
        test[clinic_col].map(full_emb).fillna(emb_global).values
    )

    # [7] 클리닉 집중도 (클리닉별 건수, v_clean10 방식도 유지)
    clinic_size = train.groupby(clinic_col).size().reset_index(name='클리닉_집중도')
    train = train.merge(clinic_size, on=clinic_col, how='left')
    test  = test.merge(clinic_size, on=clinic_col, how='left')
    train['클리닉_집중도'] = train['클리닉_집중도'].fillna(0)
    test['클리닉_집중도']  = test['클리닉_집중도'].fillna(0)

    # ── Target Encoding (v5 신규 카테고리 컬럼들) ─────────────
    print('  Target Encoding 적용 중...')
    te_cols = [
        '특정 시술 유형',
        '배란 유도 유형',
        '배아 생성 주요 이유',
        '난자 출처',
        '정자 출처',
    ]
    for col in te_cols:
        if col in train.columns and col in test.columns:
            train[col] = train[col].fillna('Unknown')
            test[col]  = test[col].fillna('Unknown')
            tr_enc, te_enc = target_encode(train, test, col, TARGET)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    # ── 최종 피처 리스트 ──────────────────────────────────────
    FEATURES = [
        # A. 나이
        '나이_수치', '고령_여부', '초고령_여부', '최적연령_여부',

        # B. 과거 이력
        '과거_임신성공률', '과거_출산성공률', 'IVF_임신성공률',
        '출산_경험', '임신_경험', 'IVF_경험', 'DI_경험',
        '반복시술_여부', 'failure_streak',
        '총 시술 횟수_num',

        # C. 시술 정보
        '시술_ICSI', '배란 자극 여부', 'IVF시술_여부', '동결배아_시술',

        # D. 배아 품질
        '이식된 배아 수', '배아_이식비율', '배아_저장비율', '배아_활용률',
        '미세주입_성공률', '파트너정자_비율', '난자_수정률',
        '신선배아_이식여부', '잉여배아_유무',

        # E. 불임 원인
        '총_불임원인_수', '불명확_단독원인',
        '남성_불임원인_수', '여성_불임원인_수',

        # F. 시간 간격
        '혼합_이식_간격', '해동_이식_간격',
        '채취_이식_간격', '채취_혼합_간격',
        '긴_배양_여부', '해동_당일이식',

        # G. 교호작용
        '남성요인_ICSI매칭', '배란장애_자극매칭', '고령_동결배아조합',
        '나이x배아활용률',

        # H. 클리닉 집계
        '시술시기코드_성공률', '시술시기코드_시술건수',
        '시술시기코드_성공률편차', '시술시기코드_배아이식수평균',
        '클리닉_나이별성공률', '클리닉_시술유형별성공률',
        '클리닉_집중도', '클리닉내_시술비율',

        # I. Target Encoding (카테고리 → 성공률)
        '특정 시술 유형_te', '배란 유도 유형_te',
        '배아 생성 주요 이유_te', '난자 출처_te', '정자 출처_te',
    ]

    feature_cols = [f for f in FEATURES if f in train.columns]
    missing = [f for f in FEATURES if f not in train.columns]
    if missing:
        print(f'\n  [경고] 없는 피처: {missing}')

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    # 최종 결측치 처리
    medians2 = X_train.median()
    X_train  = X_train.fillna(medians2)
    X_test   = X_test.fillna(medians2)

    print(f'\n  총 피처 수   : {len(feature_cols)}개')
    print(f'  X_train      : {X_train.shape}')
    print(f'  X_test       : {X_test.shape}')
    print(f'  결측치 합계  : {X_train.isnull().sum().sum()}')

    return X_train, y_train, X_test, feature_cols


# ====================================================
# 3. LightGBM
# ====================================================
def train_lgb(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print('[3-1] LightGBM (Stratified 5-Fold)')
    print('=' * 55)

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

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
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
        oof[val_idx] = m.predict_proba(X_val)[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')

    importance = pd.DataFrame({
        'feature'   : X_train.columns,
        'importance': m.feature_importances_
    }).sort_values('importance', ascending=False)

    print('\n  [피처 중요도 Top 20]')
    print(f'  {"피처":<35} {"중요도":>8}')
    print('  ' + '-' * 47)
    for _, row in importance.head(20).iterrows():
        bar = '█' * int(row['importance'] / importance['importance'].max() * 20)
        print(f'  {row["feature"]:<35} {row["importance"]:>8.0f}  {bar}')

    return oof, pred, cv, importance


# ====================================================
# 4. XGBoost
# ====================================================
def train_xgb(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print('[3-2] XGBoost (Stratified 5-Fold)')
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

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        m = xgb.XGBClassifier(
            **params, n_estimators=5000, early_stopping_rounds=200
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = m.predict_proba(X_val)[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  XGBoost CV AUC  : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 5. CatBoost
# ====================================================
def train_cat(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print('[3-3] CatBoost (Stratified 5-Fold)')
    print('=' * 55)

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
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
        oof[val_idx] = m.predict_proba(X_val)[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
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
def ensemble_and_save(y_train, sub, lgb_result, xgb_result, cat_result):
    print('\n' + '=' * 55)
    print('[4] 앙상블 & 제출 파일 저장')
    print('=' * 55)

    lgb_oof, lgb_pred, lgb_cv, _ = lgb_result
    xgb_oof, xgb_pred, xgb_cv   = xgb_result
    cat_oof, cat_pred, cat_cv   = cat_result

    cvs     = np.array([lgb_cv, xgb_cv, cat_cv])
    weights = cvs / cvs.sum()
    names   = ['LightGBM', 'XGBoost', 'CatBoost']

    ensemble_oof  = sum(w * o for w, o in zip(weights, [lgb_oof, xgb_oof, cat_oof]))
    ensemble_pred = sum(w * p for w, p in zip(weights, [lgb_pred, xgb_pred, cat_pred]))
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
    filename  = f'{SAVE_PATH}submission_{timestamp}_vclean11_auc{auc_str}.csv'

    sub['probability'] = ensemble_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')
    print(f'\n  ✅ v_clean10 앙상블 CV: 0.73841')
    print(f'     v_clean11 앙상블 CV: {ensemble_cv:.5f}  (Δ {ensemble_cv-0.73841:+.5f})')
    print(f'\n  내일 제출 전략:')
    print(f'     1번) 이 파일 단독 → LB 실측')
    print(f'     2번) LB 확인 후 best(0.74205)와 블렌딩')
    print(f'     3번) 승부수')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v_clean11)')
    print('  v_clean10 + v5 통합 최강 버전')
    print('=' * 55)

    train, test, sub                       = load_data()
    X_train, y_train, X_test, feature_cols = preprocess(train, test)

    lgb_result = train_lgb(X_train, y_train, X_test)
    xgb_result = train_xgb(X_train, y_train, X_test)
    cat_result = train_cat(X_train, y_train, X_test)

    ensemble_and_save(y_train, sub, lgb_result, xgb_result, cat_result)