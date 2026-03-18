# -*- coding: utf-8 -*-
"""
====================================================
feature_search_vclean10.py
v_clean10 베이스(25개) 위에 새 후보 피처 기여도 측정
실행: python src/feature_search_vclean10.py

[측정 방식]
  베이스 CV - 후보 피처 1개 추가 CV = 기여도
  3-Fold (속도 우선) LGB만 사용

[후보 피처 그룹]
  Group 1. 배아 관련 추가 신호
  Group 2. 나이 관련 교호작용
  Group 3. 시술 이력 비율
  Group 4. 클리닉 관련 추가 집계
  Group 5. 시간 간격 파생
====================================================
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ====================================================
# 설정 (3-Fold, 속도 우선)
# ====================================================
SEED      = 42
N_FOLDS   = 3
TARGET    = '임신 성공 여부'
DATA_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'

# v_clean10 베이스 파라미터 (기본 파라미터 사용)
LGB_PARAMS = {
    'objective'        : 'binary',
    'metric'           : 'auc',
    'learning_rate'    : 0.05,
    'num_leaves'       : 63,
    'max_depth'        : -1,
    'min_child_samples': 20,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'n_estimators'     : 1000,
    'verbose'          : -1,
    'random_state'     : SEED,
    'n_jobs'           : -1,
}

# ====================================================
# v_clean10 베이스 피처 (25개, 파트너정자_비율 제거)
# ====================================================
BASE_FEATURES = [
    '나이_수치', '과거_임신성공률',
    '시술_ICSI', '배란 자극 여부',
    '이식된 배아 수', '배아_이식비율', '미세주입_성공률',
    '배아_저장비율', '배아_활용률',
    '총_불임원인_수', '불명확_단독원인',
    '남성_불임원인_수', '여성_불임원인_수',
    '총 시술 횟수_num', 'failure_streak', '출산_경험',
    'IVF_경험', 'DI_경험',
    '시술시기코드_성공률', '시술시기코드_시술건수', '클리닉_집중도',
    'IVF시술_여부',
    '혼합_이식_간격', '해동_이식_간격',
    '남성요인_ICSI매칭',
]


# ====================================================
# 1. 데이터 로드 & 전처리
# ====================================================
def load_and_preprocess():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')

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

        # ── Group 1. 배아 추가 신호 ──────────────────────────
        # '총 난자 수' 컬럼명 탐색 (데이터마다 다를 수 있음)
        egg_col  = next(
            (c for c in df.columns if '난자' in c and '총' in c and '수' in c), None
        )
        mat_col  = next(
            (c for c in df.columns if '성숙' in c and '난자' in c), None
        )

        if egg_col:
            df['미수정란_비율'] = (
                (df[egg_col] - df['미세주입된 난자 수'].fillna(0))
                / (df[egg_col] + 1e-6)
            )
        else:
            # 총 난자 수 없으면 미세주입된 난자 수로 대리
            df['미수정란_비율'] = 0.0

        if mat_col and egg_col:
            df['난자_성숙률'] = df[mat_col] / (df[egg_col] + 1e-6)
            df['배아_생성률'] = df['총 생성 배아 수'] / (df[mat_col] + 1e-6)
        elif egg_col:
            df['난자_성숙률'] = df['미세주입된 난자 수'] / (df[egg_col] + 1e-6)
            df['배아_생성률'] = (
                df['총 생성 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
            )
        else:
            df['난자_성숙률'] = 0.0
            df['배아_생성률'] = (
                df['총 생성 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
            )

        # 신선배아_이식여부: 저장 없이 바로 이식
        df['신선배아_이식여부'] = (
            (df['저장된 배아 수'] == 0) & (df['이식된 배아 수'] > 0)
        ).astype(int)
        # 잉여배아_유무: 이식+저장 이후 잉여
        df['잉여배아_유무'] = (
            (df['총 생성 배아 수']
            - df['이식된 배아 수']
            - df['저장된 배아 수']) > 0
        ).astype(int)

        # ── Group 2. 나이 교호작용 ───────────────────────────
        # 고령_여부: 38세 이상
        df['고령_여부'] = (df['나이_수치'] >= 38).astype(int)
        # 나이x시술횟수: 나이가 많고 시술 많을수록 어려운 케이스
        df['나이x시술횟수'] = df['나이_수치'] * df['총 시술 횟수_num'].fillna(0)
        # 나이x배아품질: 나이가 많을수록 배아 품질 중요
        df['나이x배아활용률'] = df['나이_수치'] * df['배아_활용률']
        # 고령x배란자극: 고령이면서 배란 자극
        df['고령x배란자극'] = (
            df['고령_여부'] * df['배란 자극 여부']
        )

        # ── Group 3. 시술 이력 비율 ──────────────────────────
        # IVF_성공률: IVF 임신 / IVF 시술
        df['IVF_성공률'] = (
            df['IVF 임신 횟수_num'].fillna(0)
            / (df['IVF 시술 횟수_num'].fillna(0) + 1e-6)
        )
        # DI_성공률: DI 임신 / DI 시술
        df['DI_성공률'] = (
            df['DI 임신 횟수_num'].fillna(0)
            / (df['DI 시술 횟수_num'].fillna(0) + 1e-6)
        )
        # 클리닉내_시술비율: 클리닉 내 시술 / 총 시술
        df['클리닉내_시술비율'] = (
            df['클리닉 내 총 시술 횟수_num'].fillna(0)
            / (df['총 시술 횟수_num'].fillna(0) + 1e-6)
        )
        # 출산_성공률: 출산 / 임신
        df['출산_성공률'] = (
            df['총 출산 횟수_num'].fillna(0)
            / (df['총 임신 횟수_num'].fillna(0) + 1e-6)
        )

        # ── Group 4. 시간 간격 파생 ──────────────────────────
        # 이식_간격_비율: 해동간격 / 혼합간격
        df['이식_간격_비율'] = (
            df['해동_이식_간격'].abs()
            / (df['혼합_이식_간격'].abs() + 1e-6)
        )
        # 긴_배양_여부: 혼합→이식 5일 이상 (배반포)
        df['긴_배양_여부'] = (df['혼합_이식_간격'] >= 5).astype(int)
        # 해동_당일이식: 해동 당일 이식 여부
        df['해동_당일이식'] = (df['해동_이식_간격'] == 0).astype(int)

        # ── Group 5. 클리닉 집계 (추가) ──────────────────────
        # 클리닉_IVF비율: 클리닉 내 IVF 비율
        df['클리닉_IVF비율_raw'] = df['IVF시술_여부']   # 집계는 아래 OOF에서

    # ── 결측치 처리 (train median) ───────────────────────
    all_cands = BASE_FEATURES + [
        '미수정란_비율', '난자_성숙률', '배아_생성률',
        '신선배아_이식여부', '잉여배아_유무',
        '고령_여부', '나이x시술횟수', '나이x배아활용률', '고령x배란자극',
        'IVF_성공률', 'DI_성공률', '클리닉내_시술비율', '출산_성공률',
        '이식_간격_비율', '긴_배양_여부', '해동_당일이식',
    ]
    num_cols = [
        c for c in all_cands
        if c in train.select_dtypes(include=np.number).columns
    ]
    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)

    # ── 클리닉 집계 OOF ──────────────────────────────────
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()

    train['시술시기코드_성공률']  = np.nan
    train['시술시기코드_시술건수'] = np.nan
    train['클리닉_IVF비율']      = np.nan

    skf_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for _, (tr_idx, val_idx) in enumerate(skf_c.split(train, train[TARGET])):
        tr_fold = train.iloc[tr_idx]

        # 성공률/건수
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

        # 클리닉 IVF 비율
        agg_ivf = (
            tr_fold.groupby(clinic_col)['클리닉_IVF비율_raw']
            .mean()
            .reset_index()
        )
        agg_ivf.columns = [clinic_col, '_ivf']
        val_ivf = train.iloc[val_idx][[clinic_col]].merge(
            agg_ivf, on=clinic_col, how='left'
        )
        train.loc[train.index[val_idx], '클리닉_IVF비율'] = (
            val_ivf['_ivf'].values
        )

    train['시술시기코드_성공률']  = train['시술시기코드_성공률'].fillna(global_mean)
    train['시술시기코드_시술건수'] = train['시술시기코드_시술건수'].fillna(0)
    train['클리닉_IVF비율']      = train['클리닉_IVF비율'].fillna(
        train['클리닉_IVF비율_raw'].mean()
    )

    # test
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

    agg_ivf_all = (
        train.groupby(clinic_col)['클리닉_IVF비율_raw']
        .mean()
        .reset_index()
    )
    agg_ivf_all.columns = [clinic_col, '_ivf']
    test = test.merge(agg_ivf_all, on=clinic_col, how='left')
    test['클리닉_IVF비율'] = test['_ivf'].fillna(
        train['클리닉_IVF비율_raw'].mean()
    )
    test.drop(columns=['_ivf'], inplace=True, errors='ignore')

    clinic_size = (
        train.groupby(clinic_col).size().reset_index(name='클리닉_집중도')
    )
    train = train.merge(clinic_size, on=clinic_col, how='left')
    test  = test.merge(clinic_size, on=clinic_col, how='left')
    train['클리닉_집중도'] = train['클리닉_집중도'].fillna(0)
    test['클리닉_집중도']  = test['클리닉_집중도'].fillna(0)

    return train, test


# ====================================================
# 2. CV 측정 함수
# ====================================================
def get_cv(train, features):
    X = train[features]
    y = train[TARGET]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ]
        )
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

    return roc_auc_score(y, oof)


# ====================================================
# 3. 메인: 후보 피처 기여도 측정
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  feature_search - v_clean10 베이스')
    print('=' * 55)

    print('\n데이터 로드 & 전처리 중...')
    train, test = load_and_preprocess()

    # 베이스 CV 측정
    print('\n베이스 CV 측정 중 (3-Fold)...')
    base_cv = get_cv(train, BASE_FEATURES)
    print(f'  베이스 CV : {base_cv:.5f}')

    # 후보 피처 목록
    CANDIDATES = [
        # Group 1. 배아
        '미수정란_비율',
        '난자_성숙률',
        '배아_생성률',
        '신선배아_이식여부',
        '잉여배아_유무',
        # Group 2. 나이 교호작용
        '고령_여부',
        '나이x시술횟수',
        '나이x배아활용률',
        '고령x배란자극',
        # Group 3. 시술 이력 비율
        'IVF_성공률',
        'DI_성공률',
        '클리닉내_시술비율',
        '출산_성공률',
        # Group 4. 시간 간격 파생
        '이식_간격_비율',
        '긴_배양_여부',
        '해동_당일이식',
        # Group 5. 클리닉 추가
        '클리닉_IVF비율',
    ]

    # 존재하는 후보만 필터링
    valid_candidates = [c for c in CANDIDATES if c in train.columns]
    missing = [c for c in CANDIDATES if c not in train.columns]
    if missing:
        print(f'\n  [경고] 생성 안 된 후보: {missing}')

    print(f'\n후보 피처 {len(valid_candidates)}개 측정 시작...')
    print('=' * 55)

    results = []
    for i, feat in enumerate(valid_candidates, 1):
        cv = get_cv(train, BASE_FEATURES + [feat])
        delta = cv - base_cv
        mark = '✅' if delta > 0 else '❌'
        print(f'  [{i:02d}/{len(valid_candidates)}] {mark} {feat:<25} {delta:+.5f}  (CV: {cv:.5f})')
        results.append({'feature': feat, 'delta': delta, 'cv': cv})

    # ── 결과 정리 ──────────────────────────────────────────
    df_res = pd.DataFrame(results).sort_values('delta', ascending=False)

    print('\n' + '=' * 55)
    print('  최종 결과 (기여도 순)')
    print('=' * 55)
    print(f'  베이스 CV : {base_cv:.5f}')
    print()
    print(f'  {"피처":<28} {"기여도":>8}  {"CV":>8}')
    print('  ' + '-' * 50)
    for _, row in df_res.iterrows():
        mark = '✅' if row['delta'] > 0 else '❌'
        print(
            f'  {mark} {row["feature"]:<26} '
            f'{row["delta"]:>+8.5f}  {row["cv"]:>8.5f}'
        )

    print('\n  → 기여도 양수 피처를 v_clean11에 추가하세요')
    print(f'  → 베이스: {base_cv:.5f}')

    # CSV 저장
    save_path = '/Users/admin/Downloads/infertility-prediction-ai/data/'
    df_res.to_csv(save_path + 'feature_search_vclean10.csv', index=False)
    print(f'\n  결과 저장: {save_path}feature_search_vclean10.csv')