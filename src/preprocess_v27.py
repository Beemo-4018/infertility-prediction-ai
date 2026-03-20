import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

SEED   = 42
TARGET = '임신 성공 여부'

def target_encode(train, test, col, target, n_splits=5, smooth=20):
    global_mean = train[target].mean()
    train_enc   = np.zeros(len(train))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in skf.split(train, train[target]):
        tr_fold  = train.iloc[tr_idx]
        val_fold = train.iloc[val_idx]
        stats    = tr_fold.groupby(col)[target].agg(['mean', 'count'])
        smoothed = (stats['mean']*stats['count'] + global_mean*smooth) / (stats['count']+smooth)
        train_enc[val_idx] = val_fold[col].map(smoothed).fillna(global_mean).values
    full_stats  = train.groupby(col)[target].agg(['mean', 'count'])
    full_smooth = (full_stats['mean']*full_stats['count'] + global_mean*smooth) / (full_stats['count']+smooth)
    test_enc    = test[col].map(full_smooth).fillna(global_mean).values
    return train_enc, test_enc


def preprocess_v27(train, test):
    train = train.copy()
    test  = test.copy()

    # [1] 나이 수치화
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    donor_age_map = {
        '만21세 이하': 20, '만21-25세': 23, '만26-30세': 28,
        '만31-35세': 33, '만36-40세': 38, '만40세 이상': 42, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치']     = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

        # ── [V27 핵심] 난자 실제 나이 분리 ──────────────────
        # 본인 난자 → 시술 당시 나이 사용
        # 기증 난자 → 기증자 나이 사용 (더 젊음 → 성공률 높음)
        df['기증자나이_수치'] = df['난자 기증자 나이'].map(donor_age_map).fillna(-1)
        df['난자_실제나이'] = df['나이_수치']  # 기본값: 본인 나이
        mask_donor = df['난자 출처'] == '기증 제공'
        df.loc[mask_donor, '난자_실제나이'] = df.loc[mask_donor, '기증자나이_수치']
        # 기증자 나이 모를 경우 젊은 나이(23)로 대체 (기증자는 대체로 젊음)
        df.loc[mask_donor & (df['난자_실제나이'] == -1), '난자_실제나이'] = 23

    # [2] 횟수 컬럼 수치화
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

    # [3] 파생 피처
    for df in [train, test]:

        # ── V5 기본 피처 ──────────────────────────────────
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
        df['클리닉_집중도'] = df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)

        df['배아_이식비율']   = df['이식된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율']   = df['저장된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_활용률']     = (df['이식된 배아 수'] + df['저장된 배아 수']) / (df['총 생성 배아 수'] + 1e-6)
        df['미세주입_성공률'] = df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        df['미세주입_이식률'] = df['미세주입 배아 이식 수'] / (df['미세주입에서 생성된 배아 수'] + 1e-6)
        df['난자_수정률']     = df['혼합된 난자 수']   / (df['수집된 신선 난자 수'] + 1e-6)
        df['파트너정자_비율'] = df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)

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
        exist_male   = [c for c in male_cause_cols   if c in df.columns]
        exist_female = [c for c in female_cause_cols if c in df.columns]
        exist_all    = [c for c in all_cause_cols    if c in df.columns]

        df['남성_불임원인_수'] = df[exist_male].sum(axis=1)
        df['여성_불임원인_수'] = df[exist_female].sum(axis=1)
        df['총_불임원인_수']   = df[exist_all].sum(axis=1)
        df['복합_불임원인']    = (df['총_불임원인_수'] >= 2).astype(int)
        if '불명확 불임 원인' in df.columns:
            df['불명확_단독원인'] = ((df['불명확 불임 원인']==1) & (df['총_불임원인_수']==1)).astype(int)

        # 불임 심각도 비율 (수장삭 아이디어 - 내 방식으로 구현)
        df['여성_불임_비율'] = df['여성_불임원인_수'] / (df['남성_불임원인_수'] + df['여성_불임원인_수'] + 1e-6)

        for col in ['배아 해동 경과일', '임신 시도 또는 마지막 임신 경과 연수']:
            if col in df.columns:
                df[col + '_결측'] = df[col].isnull().astype(int)

        df['시술유형_나이조합']       = df['특정 시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        df['시술유형_불임주원인조합'] = df['특정 시술 유형'].astype(str) + '_male' \
                                     + df['남성 주 불임 원인'].astype(str) \
                                     + '_female' + df['여성 주 불임 원인'].astype(str)

        df['남성요인_ICSI매칭'] = (
            (df['불임 원인 - 남성 요인'] == 1) &
            (df['특정 시술 유형'].astype(str).str.contains('ICSI'))
        ).astype(int)
        df['배란장애_자극매칭'] = ((df['불임 원인 - 배란 장애']==1) & (df['배란 자극 여부']==1)).astype(int)
        df['고령_동결배아조합'] = ((df['고령_여부']==1) & (df['해동된 배아 수'].fillna(0) > 0)).astype(int)
        df['초고령_반복시술']   = ((df['초고령_여부']==1) & (df['반복시술_여부']==1)).astype(int)

        # ── V18 기존 피처 ──────────────────────────────────
        df['Implant_Efficiency']       = (df['이식된 배아 수'].fillna(0) / (df['총 생성 배아 수'].fillna(0) + 1e-5)).clip(0, 1)
        df['Frozen_Ratio']             = (df['해동된 배아 수'].fillna(0) / (df['이식된 배아 수'].fillna(0) + 1e-5)).clip(0, 1)
        df['Clinic_Concentration']     = (df['클리닉 내 총 시술 횟수_num'].fillna(0) / (df['총 시술 횟수_num'].fillna(0) + 1e-5))

        # 기존 나이 교호작용 (시술 나이 기준)
        df['Age_x_Implant_Efficiency'] = df['나이_수치'] * df['Implant_Efficiency']
        df['Age_x_Embryo_Count']       = df['나이_수치'] * df['이식된 배아 수'].fillna(0)
        df['Age_x_Transfer_Day']       = df['나이_수치'] * df['배아 이식 경과일'].fillna(0)

        # ── [V27 신규] 난자 실제나이 교호작용 ──────────────
        # 기증 난자 케이스에서 기존 Age_x_Implant_Efficiency가 잘못된 나이를 쓰고 있었음
        df['RealAge_x_Implant_Efficiency'] = df['난자_실제나이'] * df['Implant_Efficiency']
        df['RealAge_x_Embryo_Count']       = df['난자_실제나이'] * df['이식된 배아 수'].fillna(0)
        # 기증 여부 플래그
        df['기증난자_여부'] = (df['난자 출처'] == '기증 제공').astype(int)
        # 나이 차이 (시술나이 - 난자실제나이): 기증일수록 양수로 커짐
        df['나이_난자나이_차이'] = df['나이_수치'] - df['난자_실제나이']

        df['Past_Success_Index']       = (df['총 출산 횟수_num'].fillna(0) + 1) / (df['총 시술 횟수_num'].fillna(0) + 2)

        df['Donor_Egg_Age_Reversal'] = (
            (df['나이_수치'] >= 43) &
            (df['난자 출처'] == '기증 제공')
        ).astype(int)

        df['Poor_Prognosis_Multi_Early'] = (
            (df['단일 배아 이식 여부'] == 0) &
            (df['배아 이식 경과일'].fillna(0) <= 3) &
            (df['이식된 배아 수'].fillna(0) >= 2)
        ).astype(int)

        df['Surplus_Blastocyst_Reserve'] = (
            df['저장된 배아 수'].fillna(0) *
            (df['배아 이식 경과일'] == 5).astype(int)
        )

        df['Oocyte_to_Embryo_Attrition'] = (
            (df['수집된 신선 난자 수'].fillna(0) - df['총 생성 배아 수'].fillna(0)) /
            (df['수집된 신선 난자 수'].fillna(0) + 1e-5)
        ).clip(0, 1)

        df['Consecutive_IVF_Failure_Burden'] = (
            df['IVF 시술 횟수_num'].fillna(0) -
            df['IVF 임신 횟수_num'].fillna(0)
        ).clip(lower=0)

        attrition = (df['수집된 신선 난자 수'].fillna(0) - df['혼합된 난자 수'].fillna(0)).clip(lower=0)
        df['Male_Factor_Attrition'] = (df['남성_불임원인_수'] > 0).astype(int) * attrition

    # [4] drop_cols 정의
    v23_drops = [
        '불임 원인 - 정자 형태', '신선 배아 사용 여부',
        '불임 원인 - 정자 운동성', '불임 원인 - 정자 농도',
    ]
    v24_drops = [
        '난자 채취 경과일', '착상 전 유전 검사 사용 여부',
        '난자 해동 경과일', 'PGD 시술 여부', 'PGS 시술 여부',
        '저장된 신선 난자 수', '불임 원인 - 정자 면역학적 요인',
        '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제',
    ]
    v25_drops = [
        '초고령_반복시술', '채취_혼합_간격', '동결 배아 사용 여부',
        '여성 주 불임 원인', '고령_동결배아조합', '여성 부 불임 원인',
        '대리모 여부', '난자 혼합 경과일', '남성 부 불임 원인',
        '배아 해동 경과일_결측', '배아 해동 경과일',
    ]
    v26a_drops = [
        '반복시술_여부', '부부 주 불임 원인',
        '부부 부 불임 원인', '남성 주 불임 원인',
    ]

    drop_cols = ['ID', TARGET] + count_cols + v23_drops + v24_drops + v25_drops + v26a_drops
    drop_cols = [c for c in drop_cols if c in train.columns]

    feature_cols = [c for c in train.columns if c not in drop_cols]
    num_cols     = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = train[feature_cols].select_dtypes(include='object').columns.tolist()

    medians         = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)
    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # [5] 클리닉 집계 피처
    print('  클리닉 집계 피처 생성 중...')
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tr_clinic_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        stats    = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean','count'])
        smoothed = (stats['mean']*stats['count'] + global_mean*20) / (stats['count']+20)
        tr_clinic_rate[val_idx] = train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean','count'])
    full_smooth = (full_stats['mean']*full_stats['count'] + global_mean*20) / (full_stats['count']+20)
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
    tr_enc, te_enc = target_encode(train, test, '클리닉_나이조합', TARGET, smooth=20)
    train['클리닉_나이별성공률'] = tr_enc
    test['클리닉_나이별성공률']  = te_enc

    train['클리닉_시술유형조합'] = train[clinic_col].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    test['클리닉_시술유형조합']  = test[clinic_col].astype(str)  + '_' + test['특정 시술 유형'].astype(str)
    tr_enc, te_enc = target_encode(train, test, '클리닉_시술유형조합', TARGET, smooth=20)
    train['클리닉_시술유형별성공률'] = tr_enc
    test['클리닉_시술유형별성공률']  = te_enc

    tr_emb_mean = np.zeros(len(train))
    for tr_idx, val_idx in skf_clinic.split(train, train[TARGET]):
        emb_map = train.iloc[tr_idx].groupby(clinic_col)['이식된 배아 수'].mean()
        tr_emb_mean[val_idx] = train.iloc[val_idx][clinic_col].map(emb_map).fillna(train['이식된 배아 수'].mean()).values
    full_emb_map = train.groupby(clinic_col)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb_mean
    test['시술시기코드_배아이식수평균']  = test[clinic_col].map(full_emb_map).fillna(train['이식된 배아 수'].mean()).values

    # [6] Target Encoding
    print('  Target Encoding 적용 중...')
    te_cols = ['시술 시기 코드', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유',
               '난자 출처', '정자 출처', '시술유형_나이조합', '시술유형_불임주원인조합']
    for col in te_cols:
        if col in train.columns:
            tr_enc, te_enc = target_encode(train, test, col, TARGET)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    # [7] Label Encoding
    final_drop = drop_cols + [
        '클리닉_나이조합', '클리닉_시술유형조합',
        '시술유형_나이조합', '시술유형_불임주원인조합'
    ]

    feature_cols = [c for c in train.columns if c not in final_drop]
    cat_cols     = train[feature_cols].select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col].astype(str).tolist() + ['Unknown'])
        known      = set(le.classes_)
        test[col]  = test[col].astype(str).apply(lambda x: x if x in known else 'Unknown')
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    feature_cols = [c for c in train.columns if c not in final_drop]
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    print(f'  ✅ 피처 수: {len(feature_cols)}개')
    print(f'  ✅ X_train: {X_train.shape}')
    return X_train, y_train, X_test