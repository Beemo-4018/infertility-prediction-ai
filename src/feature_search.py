# -*- coding: utf-8 -*-
"""
====================================================
피처 기여도 자동 측정 스크립트
실행 방법: python src/feature_search.py

[동작 방식]
    1. v_clean7 베이스 (17개) 위에 후보 피처를 하나씩 추가
    2. 각 피처 추가 시 LightGBM 3-Fold CV AUC 측정
    3. 기여도 순으로 정렬해서 출력
    4. 결과를 feature_search_result.csv로 저장

[베이스 피처 - v_clean7 기준 CV 0.73184]
    핵심 15개 + 혼합_이식_간격 + 해동_이식_간격

[후보 피처 그룹]
    A. 클리닉 집계 (Target Encoding)
    B. 불임 원인 세분화
    C. 시술 유형 세분화
    D. 배아 추가 비율
    E. 시간 간격 추가
    F. 과거 이력 추가
====================================================
"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ====================================================
# 설정값
# ====================================================
SEED      = 42
N_FOLDS   = 3    # 속도를 위해 3-Fold
TARGET    = '임신 성공 여부'
DATA_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'

# ====================================================
# 베이스 피처 (v_clean7 기준 - CV 0.73184)
# ====================================================
BASE_FEATURES = [
    '나이_수치', '과거_임신성공률',
    '시술_ICSI', '배란 자극 여부',
    '이식된 배아 수', '배아_이식비율', '미세주입_성공률',
    '총_불임원인_수', '불명확_단독원인',
    '총 시술 횟수_num', 'failure_streak', '출산_경험',
    '시술시기코드_성공률', '시술시기코드_시술건수',
    'IVF시술_여부',
    '혼합_이식_간격', '해동_이식_간격',
]

# ====================================================
# 후보 피처 (하나씩 추가해서 기여도 측정)
# ====================================================
CANDIDATE_FEATURES = {
    # A. 클리닉 집계 (Target Encoding)
    '클리닉_나이별성공률'   : 'A. 클리닉 x 나이 TE',
    '클리닉_시술유형별성공률': 'A. 클리닉 x 시술유형 TE',

    # B. 불임 원인 세분화
    '남성_불임원인_수'      : 'B. 남성 불임 원인 수',
    '여성_불임원인_수'      : 'B. 여성 불임 원인 수',
    '복합_불임원인'         : 'B. 복합 불임 여부',
    '남성요인_ICSI매칭'     : 'B. 남성요인 x ICSI 매칭',

    # C. 시술 유형 세분화
    '시술_FER'             : 'C. FER 시술 여부',
    '시술_BLASTOCYST'      : 'C. BLASTOCYST 시술 여부',
    '시술_AH'              : 'C. AH 시술 여부',
    '시술_복합여부'         : 'C. 복합 시술 여부',
    '단일 배아 이식 여부'   : 'C. 단일 배아 이식 여부',

    # D. 배아 추가 비율
    '배아_저장비율'         : 'D. 저장/생성 배아 비율',
    '배아_활용률'           : 'D. (이식+저장)/생성 배아',
    '미세주입_이식률'       : 'D. 미세주입 배아 이식률',
    '파트너정자_비율'       : 'D. 파트너 정자 비율',

    # E. 시간 간격 추가
    '채취_이식_간격'        : 'E. 채취→이식 간격',
    '채취_혼합_간격'        : 'E. 채취→혼합 간격',

    # F. 과거 이력 추가
    'cycle_progress'       : 'F. 임신 진행률',
    'IVF_경험'             : 'F. IVF 경험 여부',
    'DI_경험'              : 'F. DI 경험 여부',
    '반복시술_여부'         : 'F. 3회 이상 반복 시술',
    '클리닉_집중도'         : 'F. 클리닉 집중도',
}


# ====================================================
# 데이터 로드
# ====================================================
def load_data():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    return train, test


# ====================================================
# Target Encoding (K-Fold OOF)
# ====================================================
def target_encode_col(train, test, col, target, n_splits=5, smooth=50):
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
# 전처리 (베이스 + 후보 피처 전부 생성)
# ====================================================
def build_features(train, test):
    train = train.copy()
    test  = test.copy()

    # 나이 수치화
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)

    # 횟수 수치화
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

    for df in [train, test]:
        # 베이스 파생 피처
        df['과거_임신성공률'] = df['총 임신 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['출산_경험']      = (df['총 출산 횟수_num']   > 0).astype(int)
        df['IVF_경험']       = (df['IVF 시술 횟수_num'] > 0).astype(int)
        df['DI_경험']        = (df['DI 시술 횟수_num']  > 0).astype(int)
        df['반복시술_여부']  = (df['총 시술 횟수_num']   >= 3).astype(int)
        df['클리닉_집중도']  = (
            df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)
        df['cycle_progress'] = (
            (df['총 임신 횟수_num'] + 1) / (df['총 시술 횟수_num'] + 1)
        )

        # 배아 비율
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
        df['파트너정자_비율'] = (
            df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        )

        # 시간 간격
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']

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

        # 시술 유형 분해
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']       = 시술유형.str.contains('ICSI',       na=False).astype(int)
        df['시술_FER']        = 시술유형.str.contains('FER',        na=False).astype(int)
        df['시술_BLASTOCYST'] = 시술유형.str.contains('BLASTOCYST', na=False).astype(int)
        df['시술_AH']         = 시술유형.str.contains('AH',         na=False).astype(int)
        df['시술_복합여부']   = 시술유형.str.contains(r'[/:]',      na=False).astype(int)
        df['IVF시술_여부']    = (df['시술 유형'] == 'IVF').astype(int)
        df['남성요인_ICSI매칭'] = (
            (df['불임 원인 - 남성 요인'] == 1) &
            (df['특정 시술 유형'].astype(str).str.contains('ICSI'))
        ).astype(int)

    # 결측치 처리 (train median) - test에 없는 컬럼 제외
    all_num_cols      = train.select_dtypes(include=np.number).columns.tolist()
    test_num_cols     = test.select_dtypes(include=np.number).columns.tolist()
    shared_num_cols   = [c for c in all_num_cols if c in test_num_cols]
    medians           = train[all_num_cols].median()
    train[all_num_cols]   = train[all_num_cols].fillna(medians)
    test[shared_num_cols] = test[shared_num_cols].fillna(medians[shared_num_cols])

    # 클리닉 집계 피처 (베이스)
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    smooth      = 50

    tr_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        stats    = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (
            (stats['mean'] * stats['count'] + global_mean * smooth)
            / (stats['count'] + smooth)
        )
        tr_rate[val_idx] = (
            train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
        )
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (
        (full_stats['mean'] * full_stats['count'] + global_mean * smooth)
        / (full_stats['count'] + smooth)
    )
    train['시술시기코드_성공률'] = tr_rate
    test['시술시기코드_성공률']  = (
        test[clinic_col].map(full_smooth).fillna(global_mean).values
    )

    tr_cnt = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        cnt_map = train.iloc[tr_idx].groupby(clinic_col).size()
        tr_cnt[val_idx] = (
            train.iloc[val_idx][clinic_col].map(cnt_map).fillna(1).values
        )
    full_cnt = train.groupby(clinic_col).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_cnt)
    test['시술시기코드_시술건수']  = np.log1p(
        test[clinic_col].map(full_cnt).fillna(1).values
    )

    # 클리닉 × 나이 TE (후보 피처)
    train['클리닉_나이조합'] = (
        train[clinic_col].astype(str) + '_' + train['시술 당시 나이'].astype(str)
    )
    test['클리닉_나이조합'] = (
        test[clinic_col].astype(str) + '_' + test['시술 당시 나이'].astype(str)
    )
    tr_enc, te_enc = target_encode_col(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = tr_enc
    test['클리닉_나이별성공률']  = te_enc

    # 클리닉 × 시술유형 TE (후보 피처)
    train['클리닉_시술유형조합'] = (
        train[clinic_col].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    )
    test['클리닉_시술유형조합'] = (
        test[clinic_col].astype(str) + '_' + test['특정 시술 유형'].astype(str)
    )
    tr_enc2, te_enc2 = target_encode_col(train, test, '클리닉_시술유형조합', TARGET)
    train['클리닉_시술유형별성공률'] = tr_enc2
    test['클리닉_시술유형별성공률']  = te_enc2

    return train, test


# ====================================================
# CV 측정 함수
# ====================================================
def measure_cv(X, y, params=None):
    if params is None:
        params = {
            'objective'        : 'binary',
            'metric'           : 'auc',
            'learning_rate'    : 0.02,
            'num_leaves'       : 63,
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
    oof  = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        X_tr  = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]
        m = lgb.LGBMClassifier(**params, n_estimators=1000)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        oof[val_idx] = m.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof)


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  피처 기여도 자동 측정')
    print(f'  베이스 피처: {len(BASE_FEATURES)}개')
    print(f'  후보 피처:   {len(CANDIDATE_FEATURES)}개')
    print('=' * 60)

    # 데이터 로드 및 전처리
    print('\n[1] 데이터 로드 & 전처리 중...')
    train_raw, test_raw = load_data()
    train, test = build_features(train_raw, test_raw)
    y = train[TARGET]

    # 베이스 CV 측정
    print('\n[2] 베이스 CV 측정 중...')
    avail_base = [f for f in BASE_FEATURES if f in train.columns]
    base_cv    = measure_cv(train[avail_base], y)
    print(f'  베이스 CV AUC: {base_cv:.5f} ({len(avail_base)}개 피처)')

    # 후보 피처 하나씩 추가해서 CV 측정
    print(f'\n[3] 후보 피처 {len(CANDIDATE_FEATURES)}개 순차 측정 중...')
    print('  ' + '-' * 56)

    results = []
    for feat, desc in CANDIDATE_FEATURES.items():
        if feat not in train.columns:
            print(f'  ⚠️ {feat}: 피처 없음 - 스킵')
            continue

        test_features = avail_base + [feat]
        cv = measure_cv(train[test_features], y)
        delta = cv - base_cv
        symbol = '↑' if delta > 0.0001 else ('↓' if delta < -0.0001 else '→')
        print(f'  {symbol} {feat:<28} CV: {cv:.5f}  ({delta:+.5f})  {desc}')
        results.append({
            'feature'    : feat,
            'description': desc,
            'cv_auc'     : cv,
            'delta'      : delta
        })

    # 결과 정렬 및 출력
    results_df = pd.DataFrame(results).sort_values('delta', ascending=False)

    print('\n' + '=' * 60)
    print('  결과 요약 (기여도 순)')
    print('=' * 60)
    print(f'  베이스 CV AUC: {base_cv:.5f}')
    print('  ' + '-' * 56)
    print(f'  {"피처":<28} {"CV AUC":>8}  {"기여도":>8}  설명')
    print('  ' + '-' * 56)
    for _, row in results_df.iterrows():
        symbol = '✅' if row['delta'] > 0.0001 else ('❌' if row['delta'] < -0.0001 else '➖')
        print(
            f'  {symbol} {row["feature"]:<26} '
            f'{row["cv_auc"]:>8.5f}  '
            f'{row["delta"]:>+8.5f}  '
            f'{row["description"]}'
        )

    # CSV 저장
    timestamp = datetime.now().strftime('%m%d_%H%M')
    save_path = f'{SAVE_PATH}feature_search_{timestamp}.csv'
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'\n  저장 완료: {save_path}')
    print('\n  ✅ 기여도 양수인 피처만 골라서 다음 버전에 추가하세요!')