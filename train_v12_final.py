# -*- coding: utf-8 -*-
"""
====================================================
 난임 환자 임신 성공 여부 예측 - v12_final
 평가 지표 : ROC-AUC
 실행 방법 : python train_v12_final.py

[v11 → v12 핵심 변경]
 ① EDA 기반 신규 강력 피처 4종
    A. 저장배아_최적 (1~4개): 36~37% vs 기준 21%  (+16%p!)
    B. 수집난자_최적 (11~15개): 32.9% (최고 구간)
    C. 이식일5_단일이식: Day5 × 이식 1개 = 42% (최고 조합)
    D. 배아이식_복합점수: (이식일 - 2) × 이식배아 수 → 비선형 포착
 ② XGB, CatBoost Optuna 튜닝 추가
    (v11은 v8 파라미터 그대로 사용 → 성능 잠재력 미발휘)
 ③ RF 추가 (ET보다 CV 높고 다양성 유지)
    ET(0.73585) → RF(~0.737) 교체, ET도 유지
 ④ 최적 가중치 탐색 유지 (Nelder-Mead)
====================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
SEEDS         = [42, 2024, 777]
N_FOLDS       = 5
TARGET        = '임신 성공 여부'
DATA_PATH     = './data/raw/'
SAVE_PATH     = './data/submissions/'
USE_OPTUNA    = True
OPTUNA_TRIALS = 50
USE_FEATURE_DIET = False
DIET_FOLDS       = 3
DIET_MAX_STEPS   = 20
DIET_MIN_IMPROVE = 0.00001
TOP_IMPORTANCE_N = 30

# v11 최적값 (warm-start)
V11_LGB = {'learning_rate':0.021905,'num_leaves':172,'max_depth':5,
            'min_child_samples':80,'feature_fraction':0.451,
            'bagging_fraction':0.680,'bagging_freq':3,
            'reg_alpha':2.555,'reg_lambda':3.089}
V8_XGB  = {'learning_rate':0.011084,'max_depth':4,'subsample':0.887,
            'colsample_bytree':0.982,'min_child_weight':12,
            'reg_alpha':7.674,'reg_lambda':0.010,'gamma':0.190}
V8_CAT  = {'learning_rate':0.036286,'depth':6,'l2_leaf_reg':0.857,
            'bagging_temperature':0.019,'random_strength':3.027,'border_count':126}

MANUAL_DROP_COLS = [
    'PGD 시술 여부',
    'PGS 시술 여부',
    'has_AH',
    'has_BLASTOCYST',
    '기증_목적',
    '난자 채취 경과일',
    '난자 해동 경과일',
    '동결 배아 사용 여부',
    '동결배아_시술',
    '배아 해동 경과일_결측',
    '배아저장_목적',
    '불임 원인 - 여성 요인',
    '불임 원인 - 자궁경부 문제',
    '불임 원인 - 정자 농도',
    '불임 원인 - 정자 면역학적 요인',
    '불임 원인 - 정자 운동성',
    '불임 원인 - 정자 형태',
    '시술_ICSI',
    '신선 배아 사용 여부',
    '신선난자_저장됨',
    '저장된 신선 난자 수',
    '정자_문제_수',
    '착상 전 유전 검사 사용 여부',
    '현재시술_목적',
]


# ====================================================
# 1. 데이터 로드
# ====================================================
def load_data():
    print('=' * 60)
    print('  난임 환자 임신 성공 여부 예측 (v12_final)')
    print('=' * 60)
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    sub   = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    print(f'  train {train.shape}  |  test {test.shape}')
    r = train[TARGET].mean()
    print(f'  임신 성공률: {r:.4f} ({r*100:.2f}%)\n')
    return train, test, sub


# ====================================================
# 2. Target Encoding
# ====================================================
def target_encode(train, test, col, target, n_splits=5, smooth=20):
    gm  = train[target].mean()
    te  = np.zeros(len(train))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for ti, vi in skf.split(train, train[target]):
        s   = train.iloc[ti].groupby(col)[target].agg(['mean','count'])
        sm  = (s['mean']*s['count'] + gm*smooth) / (s['count'] + smooth)
        te[vi] = train.iloc[vi][col].map(sm).fillna(gm).values
    fs  = train.groupby(col)[target].agg(['mean','count'])
    fsm = (fs['mean']*fs['count'] + gm*smooth) / (fs['count'] + smooth)
    return te, test[col].map(fsm).fillna(gm).values


# ====================================================
# 3. 전처리 & 피처 엔지니어링 (v12 신규 추가)
# ====================================================
def preprocess(train_raw, test_raw):
    print('[2] 전처리 & 피처 엔지니어링 (v12)')
    train = train_raw.copy()
    test  = test_raw.copy()

    count_cols = ['총 시술 횟수','클리닉 내 총 시술 횟수','IVF 시술 횟수','DI 시술 횟수',
                  '총 임신 횟수','IVF 임신 횟수','DI 임신 횟수','총 출산 횟수','IVF 출산 횟수','DI 출산 횟수']
    def parse_count(v):
        if pd.isna(v): return np.nan
        try: return float(str(v).replace('회','').replace(' 이상','').strip())
        except: return np.nan

    age_map   = {'만18-34세':26,'만35-37세':36,'만38-39세':38,'만40-42세':41,
                 '만43-44세':43,'만45-50세':47,'알 수 없음':-1}
    donor_map = {'만20세 이하':19,'만21-25세':23,'만26-30세':28,'만31-35세':33,
                 '만36-40세':38,'만41-45세':43,'만46-50세':48,'알 수 없음':-1}

    for df in [train, test]:
        # ── 나이 ──────────────────────────────────────────
        df['나이_수치']      = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']      = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']    = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부']  = (df['나이_수치'] <= 36).astype(int)
        df['나이_40이상']    = (df['나이_수치'] >= 40).astype(int)

        # ── 기증자 나이 ────────────────────────────────────
        df['난자기증자_나이_수치'] = df['난자 기증자 나이'].map(donor_map).fillna(-1)
        df['정자기증자_나이_수치'] = df['정자 기증자 나이'].map(donor_map).fillna(-1)
        df['난자기증자_있음']      = (df['난자기증자_나이_수치'] > 0).astype(int)
        df['난자기증자_젊음']      = ((df['난자기증자_나이_수치'] > 0) & (df['난자기증자_나이_수치'] <= 30)).astype(int)

        # ── 횟수 수치화 ────────────────────────────────────
        for col in count_cols:
            df[col+'_num'] = df[col].apply(parse_count)

        # ── 시술 유형 파싱 ─────────────────────────────────
        stype = df['특정 시술 유형'].fillna('Unknown').astype(str)
        df['has_BLASTOCYST'] = stype.str.contains('BLASTOCYST').astype(int)
        df['has_AH']         = stype.str.contains('AH').astype(int)
        df['시술_ICSI']      = stype.str.contains('ICSI').astype(int)
        df['is_repeat_type'] = stype.str.contains(':').astype(int)
        df['is_DI_type']     = stype.str.contains('IUI|ICI|IVI|Generic DI|GIFT').astype(int)
        df['IVF시술_여부']   = (df['시술 유형'] == 'IVF').astype(int)

        # ── 이식일 Day 세분화 ──────────────────────────────
        iday = df['배아 이식 경과일'].fillna(-1)
        df['이식일_0일']       = (iday == 0).astype(int)
        df['이식일_2_3일']     = ((iday >= 2) & (iday <= 3)).astype(int)
        df['이식일_5일']       = (iday == 5).astype(int)
        df['이식일_결측']      = (iday == -1).astype(int)
        df['이식일5_최적연령'] = (df['이식일_5일'] * df['최적연령_여부']).astype(int)
        df['이식일5_고령']     = (df['이식일_5일'] * df['고령_여부']).astype(int)
        df['배반포_이식']      = ((df['이식일_5일'] == 1) | (df['has_BLASTOCYST'] == 1)).astype(int)

        # ── 시술 목적 ──────────────────────────────────────
        reason = df['배아 생성 주요 이유'].fillna('Unknown').astype(str)
        df['현재시술_목적']   = reason.str.contains('현재 시술용').astype(int)
        df['배아저장_목적']   = (reason.str.contains('저장') & ~reason.str.contains('현재')).astype(int)
        df['기증_목적']       = (reason.str.contains('기증') & ~reason.str.contains('현재')).astype(int)
        df['비현재시술_목적'] = ((df['배아저장_목적'] == 1) | (df['기증_목적'] == 1)).astype(int)

        # ── 배아 수 피처 ───────────────────────────────────
        emb_total = df['총 생성 배아 수'].fillna(0)
        emb_trans = df['이식된 배아 수'].fillna(0)
        emb_store = df['저장된 배아 수'].fillna(0)
        egg_fresh = df['수집된 신선 난자 수'].fillna(0)

        df['배아충분_여부']    = (emb_total >= 5).astype(int)
        df['배아풍부_여부']    = (emb_total >= 9).astype(int)
        df['배아이식_최적']    = ((emb_trans >= 1) & (emb_trans <= 2)).astype(int)
        df['배아이식_제로']    = (emb_trans == 0).astype(int)
        df['배아풍부_단일이식']= ((df['배아풍부_여부'] == 1) & (emb_trans == 1)).astype(int)
        df['배아풍부_이식최적']= ((df['배아충분_여부'] == 1) & (df['배아이식_최적'] == 1)).astype(int)
        df['배아수_구간']      = pd.cut(emb_total, bins=[-1,0,2,4,6,8,999],
                                       labels=[0,1,2,3,4,5]).astype(float)

        # ── [v12 NEW A] 저장 배아 최적 구간 ────────────────
        # EDA: 1~4개 = 36~37%, 없음 = 21%, 5+개 = 27%
        df['저장배아_최적']  = ((emb_store >= 1) & (emb_store <= 4)).astype(int)
        df['저장배아_풍부']  = (emb_store >= 3).astype(int)
        df['저장배아_없음']  = (emb_store == 0).astype(int)
        df['저장배아수_log'] = np.log1p(emb_store)

        # ── [v12 NEW B] 수집 난자 최적 구간 ────────────────
        # EDA: 11~15개 = 32.9%, 16+개 = 30.8%, 1-5개 = 14.8%
        df['수집난자_최적']  = ((egg_fresh >= 11) & (egg_fresh <= 15)).astype(int)
        df['수집난자_풍부']  = (egg_fresh >= 11).astype(int)
        df['수집난자_부족']  = ((egg_fresh > 0) & (egg_fresh <= 5)).astype(int)
        df['수집난자수_log'] = np.log1p(egg_fresh)

        # ── [v12 NEW C] Day5 × 이식수 최강 조합 ───────────
        # EDA: Day5 × 이식1개 = 42.1%, Day5 × 이식2개 = 38.9%
        df['이식일5_단일이식']  = ((df['이식일_5일'] == 1) & (emb_trans == 1)).astype(int)
        df['이식일5_이식최적']  = ((df['이식일_5일'] == 1) & (df['배아이식_최적'] == 1)).astype(int)
        df['이식일23_이식최적'] = ((df['이식일_2_3일'] == 1) & (df['배아이식_최적'] == 1)).astype(int)

        # ── [v12 NEW D] 이식일 × 이식배아 복합 점수 ────────
        # Day가 높을수록(배반포), 이식 수가 1~2개일수록 유리
        iday_pos = iday.clip(lower=0)
        df['이식_복합점수'] = (
            iday_pos * 0.5 +                    # Day가 클수록 +
            df['배아이식_최적'] * 3.0 +          # 1~2개 이식
            df['이식일_5일'] * 2.0 +             # Day5 가중
            df['배아풍부_단일이식'] * 2.0 -       # 배아 풍부 + 단일
            df['배아이식_제로'] * 5.0             # 이식 없음 강한 패널티
        )

        # ── SET (단일 배아 이식) ───────────────────────────
        set_f = df['단일 배아 이식 여부'].fillna(0)
        df['SET_최적연령']  = (set_f * df['최적연령_여부']).astype(int)
        df['SET_배반포']    = (set_f * df['배반포_이식']).astype(int)

        # ── 이력 성공률 ────────────────────────────────────
        t  = df['총 시술 횟수_num'].fillna(0)
        pr = df['총 임신 횟수_num'].fillna(0)
        br = df['총 출산 횟수_num'].fillna(0)
        df['과거_임신성공률']  = pr / (t + 1e-6)
        df['과거_출산성공률']  = br / (t + 1e-6)
        df['IVF_임신성공률']  = df['IVF 임신 횟수_num'].fillna(0) / (df['IVF 시술 횟수_num'].fillna(0) + 1e-6)
        df['IVF_출산성공률']  = df['IVF 출산 횟수_num'].fillna(0) / (df['IVF 시술 횟수_num'].fillna(0) + 1e-6)
        df['failure_streak']  = (t - pr).clip(lower=0)
        df['IVF_경험']        = (df['IVF 시술 횟수_num'].fillna(0) > 0).astype(int)
        df['임신_경험']       = (pr > 0).astype(int)
        df['출산_경험']       = (br > 0).astype(int)
        df['반복시술_여부']   = (t >= 3).astype(int)
        df['클리닉_집중도']   = df['클리닉 내 총 시술 횟수_num'].fillna(0) / (t + 1e-6)
        df['IVF_비율']        = df['IVF 시술 횟수_num'].fillna(0) / (t + 1e-6)
        df['클리닉_전환']     = ((t - df['클리닉 내 총 시술 횟수_num'].fillna(0)) > 0).astype(int)

        # ── 배아 효율 비율 ─────────────────────────────────
        혼합     = df['혼합된 난자 수'].fillna(0)
        생성     = df['총 생성 배아 수'].fillna(0)
        미세생성 = df['미세주입에서 생성된 배아 수'].fillna(0)
        미세주입 = df['미세주입된 난자 수'].fillna(0)
        df['수정효율']        = 생성 / (혼합 + 1e-6)
        df['이식효율']        = emb_trans / (생성 + 1e-6)
        df['배아_활용률']     = (emb_trans + emb_store) / (생성 + 1e-6)
        df['미세주입_성공률'] = 미세생성 / (미세주입 + 1e-6)
        df['미세주입_이식률'] = df['미세주입 배아 이식 수'].fillna(0) / (미세생성 + 1e-6)
        df['난자_수정률']     = 혼합 / (egg_fresh + 1e-6)
        df['파트너정자_비율'] = df['파트너 정자와 혼합된 난자 수'].fillna(0) / (혼합 + 1e-6)
        df['해동난자_있음']   = (df['해동 난자 수'].fillna(0) > 0).astype(int)
        df['동결배아_시술']   = (df['해동된 배아 수'].fillna(0) > 0).astype(int)
        df['신선난자_저장됨'] = (df['저장된 신선 난자 수'].fillna(0) > 0).astype(int)

        # ── Log1p 변환 ─────────────────────────────────────
        for sc in ['총 생성 배아 수','미세주입된 난자 수','수집된 신선 난자 수',
                   '혼합된 난자 수','저장된 배아 수','해동된 배아 수','이식된 배아 수']:
            df[f'{sc}_log'] = np.log1p(df[sc].fillna(0).clip(lower=0))

        # ── 시간 간격 ──────────────────────────────────────
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']

        # ── 불임 원인 ──────────────────────────────────────
        male_c = ['불임 원인 - 남성 요인','불임 원인 - 정자 농도',
                  '불임 원인 - 정자 면역학적 요인','불임 원인 - 정자 운동성','불임 원인 - 정자 형태']
        fem_c  = ['불임 원인 - 난관 질환','불임 원인 - 배란 장애',
                  '불임 원인 - 여성 요인','불임 원인 - 자궁경부 문제','불임 원인 - 자궁내막증']
        all_c  = male_c + fem_c + ['남성 주 불임 원인','남성 부 불임 원인',
                  '여성 주 불임 원인','여성 부 불임 원인','부부 주 불임 원인',
                  '부부 부 불임 원인','불명확 불임 원인']
        df['남성_불임원인_수'] = df[male_c].sum(axis=1)
        df['여성_불임원인_수'] = df[fem_c].sum(axis=1)
        df['총_불임원인_수']   = df[all_c].sum(axis=1)
        df['복합_불임원인']    = (df['총_불임원인_수'] >= 2).astype(int)
        df['정자_문제_수']     = df[['불임 원인 - 정자 농도','불임 원인 - 정자 운동성',
                                     '불임 원인 - 정자 형태']].sum(axis=1)

        for col in ['착상 전 유전 검사 사용 여부','PGD 시술 여부','PGS 시술 여부',
                    '난자 해동 경과일','배아 해동 경과일']:
            df[col+'_결측'] = df[col].isnull().astype(int)

        # ── 교호작용 ───────────────────────────────────────
        df['남성요인_ICSI매칭']    = ((df['불임 원인 - 남성 요인'] == 1) & df['시술_ICSI'].astype(bool)).astype(int)
        df['배란장애_자극매칭']    = ((df['불임 원인 - 배란 장애'] == 1) & (df['배란 자극 여부'] == 1)).astype(int)
        df['고령_동결배아조합']    = ((df['고령_여부'] == 1) & (df['동결배아_시술'] == 1)).astype(int)
        df['초고령_반복시술']      = ((df['초고령_여부'] == 1) & (df['반복시술_여부'] == 1)).astype(int)
        df['기증난자_고령조합']    = ((df['난자기증자_있음'] == 1) & (df['고령_여부'] == 1)).astype(int)
        df['기증난자_젊음_고령모'] = ((df['난자기증자_젊음'] == 1) & (df['고령_여부'] == 1)).astype(int)
        df['기증배아_사용']        = df['기증 배아 사용 여부'].fillna(0)
        df['대리모_여부_f']        = df['대리모 여부'].fillna(0)

        # ── 복합 성공 지수 (v12 업데이트) ─────────────────
        df['성공_복합지수'] = (
            df['최적연령_여부']   * 2.0 +
            df['배반포_이식']     * 3.0 +
            df['SET_배반포']      * 1.5 +
            df['이식일5_단일이식']* 2.0 +   # v12 신규: 최강 조합
            df['저장배아_최적']   * 2.0 +   # v12 신규: 36% 신호
            df['수집난자_풍부']   * 1.5 +   # v12 신규: 33% 신호
            df['배아이식_최적']   * 1.0 +
            df['임신_경험']       * 1.0 +
            df['출산_경험']       * 1.5 +
            df['난자기증자_젊음'] * 1.0 +
            df['현재시술_목적']   * 2.0 -
            df['비현재시술_목적'] * 5.0 -
            df['초고령_여부']     * 2.0 -
            df['is_repeat_type']  * 3.0 -
            df['배아이식_제로']   * 3.0 -
            df['failure_streak']  * 0.3
        )

        df['시술유형_나이조합']      = df['특정 시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        df['시술유형_불임주원인조합'] = (df['특정 시술 유형'].astype(str)
                                      + '_m' + df['남성 주 불임 원인'].astype(str)
                                      + '_f' + df['여성 주 불임 원인'].astype(str))

    # ── 결측치 처리 ────────────────────────────────────────
    drop_base = ['ID', TARGET] + count_cols + ['클리닉_나이조합','클리닉_시술유형조합','클리닉_배반포조합']
    feat_tmp  = [c for c in train.columns if c not in drop_base]
    num_c = train[feat_tmp].select_dtypes(include=[np.number]).columns.tolist()
    cat_c = train[feat_tmp].select_dtypes(include='object').columns.tolist()
    medians = train[num_c].median()
    train[num_c] = train[num_c].fillna(medians)
    test[num_c]  = test[num_c].fillna(medians)
    train[cat_c] = train[cat_c].fillna('Unknown')
    test[cat_c]  = test[cat_c].fillna('Unknown')

    # ── 클리닉 집계 (OOF) ──────────────────────────────────
    print('  클리닉 집계 피처 생성 중...')
    clinic = '시술 시기 코드'
    gm     = train[TARGET].mean()
    skf_c  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tr_r = np.zeros(len(train))
    for ti, vi in skf_c.split(train, train[TARGET]):
        s = train.iloc[ti].groupby(clinic)[TARGET].agg(['mean','count'])
        sm = (s['mean']*s['count'] + gm*20)/(s['count']+20)
        tr_r[vi] = train.iloc[vi][clinic].map(sm).fillna(gm).values
    fs  = train.groupby(clinic)[TARGET].agg(['mean','count'])
    fsm = (fs['mean']*fs['count'] + gm*20)/(fs['count']+20)
    train['시술시기코드_성공률'] = tr_r
    test['시술시기코드_성공률']  = test[clinic].map(fsm).fillna(gm).values
    train['시술시기코드_성공률편차'] = train['시술시기코드_성공률'] - gm
    test['시술시기코드_성공률편차']  = test['시술시기코드_성공률']  - gm

    tr_cnt = np.zeros(len(train))
    for ti, vi in skf_c.split(train, train[TARGET]):
        cm = train.iloc[ti].groupby(clinic).size()
        tr_cnt[vi] = train.iloc[vi][clinic].map(cm).fillna(1).values
    fcm = train.groupby(clinic).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_cnt)
    test['시술시기코드_시술건수']  = np.log1p(test[clinic].map(fcm).fillna(1).values)

    train['클리닉_나이조합'] = train[clinic].astype(str) + '_' + train['시술 당시 나이'].astype(str)
    test['클리닉_나이조합']  = test[clinic].astype(str)  + '_' + test['시술 당시 나이'].astype(str)
    te2, te2t = target_encode(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = te2;  test['클리닉_나이별성공률'] = te2t

    train['클리닉_시술유형조합'] = train[clinic].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    test['클리닉_시술유형조합']  = test[clinic].astype(str)  + '_' + test['특정 시술 유형'].astype(str)
    te3, te3t = target_encode(train, test, '클리닉_시술유형조합', TARGET)
    train['클리닉_시술유형별성공률'] = te3;  test['클리닉_시술유형별성공률'] = te3t

    train['클리닉_배반포조합'] = train[clinic].astype(str) + '_' + train['배반포_이식'].astype(str)
    test['클리닉_배반포조합']  = test[clinic].astype(str)  + '_' + test['배반포_이식'].astype(str)
    te4, te4t = target_encode(train, test, '클리닉_배반포조합', TARGET)
    train['클리닉_배반포별성공률'] = te4;  test['클리닉_배반포별성공률'] = te4t

    # [v12 NEW] 클리닉 × 저장배아_최적
    train['클리닉_저장배아조합'] = train[clinic].astype(str) + '_' + train['저장배아_최적'].astype(str)
    test['클리닉_저장배아조합']  = test[clinic].astype(str)  + '_' + test['저장배아_최적'].astype(str)
    te5, te5t = target_encode(train, test, '클리닉_저장배아조합', TARGET)
    train['클리닉_저장배아별성공률'] = te5;  test['클리닉_저장배아별성공률'] = te5t

    tr_emb = np.zeros(len(train))
    for ti, vi in skf_c.split(train, train[TARGET]):
        em = train.iloc[ti].groupby(clinic)['이식된 배아 수'].mean()
        tr_emb[vi] = train.iloc[vi][clinic].map(em).fillna(train['이식된 배아 수'].mean()).values
    fem = train.groupby(clinic)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb
    test['시술시기코드_배아이식수평균']  = test[clinic].map(fem).fillna(train['이식된 배아 수'].mean()).values

    train['클리닉대비_개인성공률차이'] = train['시술시기코드_성공률'] - train['과거_임신성공률']
    test['클리닉대비_개인성공률차이']  = test['시술시기코드_성공률']  - test['과거_임신성공률']

    # ── Target Encoding ────────────────────────────────────
    print('  Target Encoding 적용 중...')
    te_cols  = ['시술 시기 코드','특정 시술 유형','배란 유도 유형','배아 생성 주요 이유',
                '난자 출처','정자 출처','시술 유형','난자 기증자 나이','정자 기증자 나이']
    te_inter = ['시술유형_나이조합','시술유형_불임주원인조합']
    for col in te_cols + te_inter:
        if col in train.columns and col in test.columns:
            tr_e, te_e = target_encode(train, test, col, TARGET)
            train[col+'_te'] = tr_e;  test[col+'_te'] = te_e

    # ── Label Encoding ─────────────────────────────────────
    drop_cols = ['ID', TARGET] + count_cols + [
        '클리닉_나이조합','클리닉_시술유형조합','클리닉_배반포조합','클리닉_저장배아조합'
    ]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    cat_c2    = train[feat_cols].select_dtypes(include='object').columns.tolist()
    for col in cat_c2:
        le = LabelEncoder()
        le.fit(train[col].astype(str).tolist() + ['Unknown'])
        known     = set(le.classes_)
        test[col] = test[col].astype(str).apply(lambda x: x if x in known else 'Unknown')
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    feat_cols = [c for c in train.columns if c not in drop_cols]
    manual_drop = [c for c in MANUAL_DROP_COLS if c in feat_cols]
    if manual_drop:
        feat_cols = [c for c in feat_cols if c not in set(manual_drop)]
        print(f'  manual feature diet 적용: {len(manual_drop)}개 제거')
    X_train   = train[feat_cols]
    y_train   = train[TARGET]
    X_test    = test[feat_cols]
    print(f'  피처 수: {len(feat_cols)}  | 결측치: {X_train.isnull().sum().sum()}')
    return X_train, y_train, X_test, feat_cols


# ====================================================
# 4. Feature Selection
# ====================================================
def select_features(X_train, y_train, X_test, feat_cols):
    print('\n[3] Feature Selection')
    skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    imps = np.zeros(len(feat_cols))
    for ti, vi in skf.split(X_train, y_train):
        m = lgb.LGBMClassifier(
            objective='binary', n_estimators=500, learning_rate=0.05,
            num_leaves=127, verbose=-1, random_state=SEED, n_jobs=-1
        )
        m.fit(X_train.iloc[ti], y_train.iloc[ti],
              eval_set=[(X_train.iloc[vi], y_train.iloc[vi])],
              callbacks=[lgb.early_stopping(30,verbose=False), lgb.log_evaluation(0)])
        imps += m.feature_importances_

    keep = [c for c, imp in zip(feat_cols, imps) if imp > 0]
    print(f'  {len(keep)}개 유지 (제거: {len(feat_cols)-len(keep)}개)')

    # TOP 10 출력
    imp_df = pd.DataFrame({'feature':feat_cols,'importance':imps})
    top = imp_df[imp_df['importance']>0].sort_values('importance',ascending=False).head(10)
    print('  TOP 10:')
    for _, r in top.iterrows():
        print(f'    {r["feature"]:<40} {r["importance"]:>8.0f}')
    return X_train[keep], X_test[keep], keep


def build_lgb_screen_params():
    return {
        'objective': 'binary',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1
    }


def evaluate_lgb_feature_set(X_train, y_train, feat_cols, n_splits=3):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    imps = np.zeros(len(feat_cols))
    oof  = np.zeros(len(X_train))

    for ti, vi in skf.split(X_train, y_train):
        m = lgb.LGBMClassifier(**build_lgb_screen_params())
        m.fit(X_train.iloc[ti][feat_cols], y_train.iloc[ti],
              eval_set=[(X_train.iloc[vi][feat_cols], y_train.iloc[vi])],
              callbacks=[lgb.early_stopping(30,verbose=False), lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(X_train.iloc[vi][feat_cols])[:,1]
        imps += m.feature_importances_

    cv = roc_auc_score(y_train, oof)
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': imps})
    imp_df = imp_df.sort_values(['importance', 'feature'], ascending=[False, True]).reset_index(drop=True)
    return cv, imp_df


def print_top_importance(imp_df, title='TOP 중요도', top_n=10):
    top = imp_df[imp_df['importance'] > 0].head(top_n)
    print(f'  {title}:')
    if len(top) == 0:
        print('    importance > 0 인 피처가 없습니다.')
        return
    for _, r in top.iterrows():
        print(f'    {r["feature"]:<40} {r["importance"]:>8.0f}')


def save_importance_csv(imp_df, filename):
    os.makedirs(SAVE_PATH, exist_ok=True)
    path = os.path.join(SAVE_PATH, filename)
    imp_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  피처 중요도 저장: {path}')
    return path


def select_features_with_importance(X_train, y_train, X_test, feat_cols):
    print('\n[3] Feature Selection')
    base_cv, imp_df = evaluate_lgb_feature_set(X_train, y_train, feat_cols, n_splits=DIET_FOLDS)
    keep = imp_df.loc[imp_df['importance'] > 0, 'feature'].tolist()

    print(f'  baseline LGB CV : {base_cv:.5f}')
    print(f'  {len(keep)}개 유지 (제거: {len(feat_cols)-len(keep)}개)')
    print_top_importance(imp_df, top_n=10)
    save_importance_csv(imp_df, 'feature_importance_v12_full.csv')

    return X_train[keep], X_test[keep], keep, imp_df


def run_feature_diet(X_train, y_train, X_test, feat_cols, base_imp_df):
    print('\n[3-1] Feature Diet Flip')
    active_feats = list(feat_cols)
    current_cv, current_imp_df = evaluate_lgb_feature_set(X_train, y_train, active_feats, n_splits=DIET_FOLDS)
    best_cv = current_cv
    accepted = []
    logs = [{
        'step': 0,
        'action': 'baseline',
        'feature': '',
        'feature_count': len(active_feats),
        'cv_auc': round(current_cv, 6),
        'delta': 0.0,
        'accepted': 1
    }]

    candidate_order = [
        f for f in base_imp_df.sort_values(['importance', 'feature'], ascending=[True, True])['feature'].tolist()
        if f in active_feats
    ]

    for feature in candidate_order:
        if feature not in active_feats:
            continue
        if len(accepted) >= DIET_MAX_STEPS:
            print(f'  최대 accepted prune 수({DIET_MAX_STEPS})에 도달해 종료합니다.')
            break

        trial_feats = [f for f in active_feats if f != feature]
        trial_cv, trial_imp_df = evaluate_lgb_feature_set(X_train, y_train, trial_feats, n_splits=DIET_FOLDS)
        delta = trial_cv - best_cv
        accepted_flag = delta >= DIET_MIN_IMPROVE

        logs.append({
            'step': len(logs),
            'action': 'drop',
            'feature': feature,
            'feature_count': len(trial_feats),
            'cv_auc': round(trial_cv, 6),
            'delta': round(delta, 6),
            'accepted': int(accepted_flag)
        })

        mark = 'ACCEPT' if accepted_flag else 'reject'
        print(f'  [{mark}] drop {feature} -> CV {trial_cv:.5f} (delta {delta:+.5f})')

        if not accepted_flag:
            continue

        active_feats = trial_feats
        current_imp_df = trial_imp_df
        best_cv = trial_cv
        accepted.append(feature)

    log_df = pd.DataFrame(logs)
    os.makedirs(SAVE_PATH, exist_ok=True)
    log_path = os.path.join(SAVE_PATH, 'feature_diet_log_v12.csv')
    feat_path = os.path.join(SAVE_PATH, 'feature_diet_selected_v12.csv')
    imp_path = os.path.join(SAVE_PATH, 'feature_importance_v12_diet.csv')

    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    pd.DataFrame({'feature': active_feats}).to_csv(feat_path, index=False, encoding='utf-8-sig')
    current_imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')

    print(f'  diet baseline CV : {current_cv:.5f}')
    print(f'  diet best CV     : {best_cv:.5f}')
    print(f'  accepted drops   : {accepted}')
    print_top_importance(current_imp_df, title='Diet 후 TOP 중요도', top_n=10)
    print(f'  diet log 저장    : {log_path}')
    print(f'  diet 피처 저장   : {feat_path}')
    print(f'  diet 중요도 저장 : {imp_path}')

    return X_train[active_feats], X_test[active_feats], active_feats, current_imp_df, log_df


# ====================================================
# 5. Optuna (LGB + XGB + CAT 전부)
# ====================================================
def run_optuna_all(X_train, y_train):
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ── LGB ──
    print(f'\n[4-1] Optuna LightGBM ({OPTUNA_TRIALS} trials)')
    def lgb_obj(trial):
        p = {'objective':'binary','metric':'auc','verbose':-1,'random_state':SEED,'n_jobs':-1,
             'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.05),
             'num_leaves'       : trial.suggest_int('num_leaves', 100, 300),
             'max_depth'        : trial.suggest_int('max_depth', 3, 7),
             'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
             'feature_fraction' : trial.suggest_float('feature_fraction', 0.4, 0.8),
             'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.4, 0.8),
             'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 7),
             'reg_alpha'        : trial.suggest_float('reg_alpha', 1.0, 15.0, log=True),
             'reg_lambda'       : trial.suggest_float('reg_lambda', 0.5, 10.0, log=True)}
        aucs = []
        for ti, vi in skf3.split(X_train, y_train):
            m = lgb.LGBMClassifier(**p, n_estimators=2000)
            m.fit(X_train.iloc[ti], y_train.iloc[ti],
                  eval_set=[(X_train.iloc[vi], y_train.iloc[vi])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(y_train.iloc[vi], m.predict_proba(X_train.iloc[vi])[:,1]))
        return np.mean(aucs)
    st = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    st.enqueue_trial(V11_LGB)
    st.optimize(lgb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_lgb = st.best_params
    print(f'  LGB 최적 AUC: {st.best_value:.5f}')

    # ── XGB ──
    print(f'\n[4-2] Optuna XGBoost ({OPTUNA_TRIALS} trials)')
    def xgb_obj(trial):
        p = {'objective':'binary:logistic','eval_metric':'auc','random_state':SEED,
             'n_jobs':-1,'verbosity':0,'tree_method':'hist',
             'learning_rate'   : trial.suggest_float('learning_rate', 0.005, 0.05),
             'max_depth'       : trial.suggest_int('max_depth', 3, 7),
             'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
             'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
             'reg_alpha'       : trial.suggest_float('reg_alpha', 0.1, 15.0, log=True),
             'reg_lambda'      : trial.suggest_float('reg_lambda', 0.001, 5.0, log=True),
             'gamma'           : trial.suggest_float('gamma', 0.0, 1.0)}
        aucs = []
        for ti, vi in skf3.split(X_train, y_train):
            m = xgb.XGBClassifier(**p, n_estimators=2000, early_stopping_rounds=50)
            m.fit(X_train.iloc[ti], y_train.iloc[ti],
                  eval_set=[(X_train.iloc[vi], y_train.iloc[vi])], verbose=False)
            aucs.append(roc_auc_score(y_train.iloc[vi], m.predict_proba(X_train.iloc[vi])[:,1]))
        return np.mean(aucs)
    st2 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    st2.enqueue_trial(V8_XGB)
    st2.optimize(xgb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_xgb = st2.best_params
    print(f'  XGB 최적 AUC: {st2.best_value:.5f}')

    # ── CAT ──
    print(f'\n[4-3] Optuna CatBoost ({OPTUNA_TRIALS} trials)')
    def cat_obj(trial):
        p = {'eval_metric':'AUC','random_seed':SEED,'verbose':False,'task_type':'CPU',
             'learning_rate'      : trial.suggest_float('learning_rate', 0.01, 0.05),
             'depth'              : trial.suggest_int('depth', 4, 8),
             'l2_leaf_reg'        : trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
             'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
             'random_strength'    : trial.suggest_float('random_strength', 0.5, 5.0),
             'border_count'       : trial.suggest_int('border_count', 64, 254)}
        aucs = []
        for ti, vi in skf3.split(X_train, y_train):
            m = CatBoostClassifier(**p, iterations=1000, early_stopping_rounds=50)
            m.fit(X_train.iloc[ti], y_train.iloc[ti],
                  eval_set=(X_train.iloc[vi], y_train.iloc[vi]))
            aucs.append(roc_auc_score(y_train.iloc[vi], m.predict_proba(X_train.iloc[vi])[:,1]))
        return np.mean(aucs)
    st3 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    st3.enqueue_trial(V8_CAT)
    st3.optimize(cat_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_cat = st3.best_params
    print(f'  CAT 최적 AUC: {st3.best_value:.5f}')

    return best_lgb, best_xgb, best_cat


# ====================================================
# 6. 모델 학습 (Multi-Seed)
# ====================================================
def _multi(X_tr, y_tr, X_te, make_fn, label, seeds=SEEDS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_oof, all_pred = [], []
    for seed in seeds:
        oof = np.zeros(len(X_tr));  pred = np.zeros(len(X_te))
        m_obj = make_fn(seed)
        for ti, vi in skf.split(X_tr, y_tr):
            m = make_fn(seed)
            m.fit(X_tr.iloc[ti], y_tr.iloc[ti],
                  eval_set=[(X_tr.iloc[vi], y_tr.iloc[vi])])
            oof[vi] = m.predict_proba(X_tr.iloc[vi])[:,1]
            pred   += m.predict_proba(X_te)[:,1] / N_FOLDS
        cv = roc_auc_score(y_tr, oof)
        print(f'  [{label}] Seed {seed}  CV: {cv:.5f}')
        all_oof.append(oof);  all_pred.append(pred)
    oof_m = np.mean(all_oof,0);  pred_m = np.mean(all_pred,0)
    cv = roc_auc_score(y_tr, oof_m)
    print(f'  [{label}] Multi-Seed CV: {cv:.5f}\n')
    return oof_m, pred_m, cv


class LGBWrapper:
    def __init__(self, params):
        self.params = params
    def fit(self, X, y, eval_set=None):
        self.m = lgb.LGBMClassifier(**self.params, n_estimators=5000)
        Xv, yv = eval_set[0]
        self.m.fit(X, y, eval_set=[(Xv, yv)],
                   callbacks=[lgb.early_stopping(200,verbose=False), lgb.log_evaluation(0)])
    def predict_proba(self, X): return self.m.predict_proba(X)

class XGBWrapper:
    def __init__(self, params):
        self.params = params
    def fit(self, X, y, eval_set=None):
        self.m = xgb.XGBClassifier(**self.params, n_estimators=5000, early_stopping_rounds=200)
        self.m.fit(X, y, eval_set=eval_set, verbose=False)
    def predict_proba(self, X): return self.m.predict_proba(X)

class CATWrapper:
    def __init__(self, params):
        self.params = params
    def fit(self, X, y, eval_set=None):
        self.m = CatBoostClassifier(**self.params, iterations=5000, eval_metric='AUC',
                                    verbose=False, early_stopping_rounds=200)
        self.m.fit(X, y, eval_set=eval_set[0])
    def predict_proba(self, X): return self.m.predict_proba(X)

class ETWrapper:
    def __init__(self, seed):
        self.seed = seed
    def fit(self, X, y, eval_set=None):
        self.m = ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                      min_samples_leaf=20, n_jobs=-1, random_state=self.seed)
        self.m.fit(X, y)
    def predict_proba(self, X): return self.m.predict_proba(X)

class RFWrapper:
    def __init__(self, seed):
        self.seed = seed
    def fit(self, X, y, eval_set=None):
        self.m = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                        min_samples_leaf=15, n_jobs=-1, random_state=self.seed)
        self.m.fit(X, y)
    def predict_proba(self, X): return self.m.predict_proba(X)


def train_all_models(X_tr, y_tr, X_te, best_lgb, best_xgb, best_cat):
    lgb_p = {**best_lgb,'objective':'binary','metric':'auc','verbose':-1,'n_jobs':-1}
    xgb_p = {**best_xgb,'objective':'binary:logistic','eval_metric':'auc',
              'n_jobs':-1,'verbosity':0,'tree_method':'hist'}
    cat_p = {**best_cat,'task_type':'CPU'}

    print('\n[5-1] LightGBM Multi-Seed')
    lgb_r = _multi(X_tr, y_tr, X_te, lambda s: LGBWrapper({**lgb_p,'random_state':s}), 'LGB')
    print('\n[5-2] XGBoost Multi-Seed')
    xgb_r = _multi(X_tr, y_tr, X_te, lambda s: XGBWrapper({**xgb_p,'random_state':s}), 'XGB')
    print('\n[5-3] CatBoost Multi-Seed')
    cat_r = _multi(X_tr, y_tr, X_te, lambda s: CATWrapper({**cat_p,'random_seed':s}), 'CAT')
    print('\n[5-4] ExtraTrees Multi-Seed')
    et_r  = _multi(X_tr, y_tr, X_te, lambda s: ETWrapper(s), 'ET')
    print('\n[5-5] RandomForest Multi-Seed')
    rf_r  = _multi(X_tr, y_tr, X_te, lambda s: RFWrapper(s), 'RF')

    return [lgb_r, xgb_r, cat_r, et_r, rf_r], ['LightGBM','XGBoost','CatBoost','ExtraTrees','RandomForest']


# ====================================================
# 7. 최적 블렌딩 가중치 탐색
# ====================================================
def optimize_weights(y_train, oofs, names):
    print('\n[6] OOF 최적 블렌딩 가중치 탐색')
    n = len(oofs)

    def neg_auc(w):
        w = np.clip(w, 0.02, 0.65)
        w = w / w.sum()
        return -roc_auc_score(y_train, sum(wi*o for wi,o in zip(w, oofs)))

    w0  = np.ones(n) / n
    res = minimize(neg_auc, w0, method='Nelder-Mead',
                   options={'maxiter':3000,'xatol':1e-7,'fatol':1e-7})
    w_opt = np.clip(res.x, 0.02, 0.65);  w_opt /= w_opt.sum()

    print(f'\n  {"모델":<14} {"균등":>6}  {"최적":>6}  {"CV AUC":>10}')
    print('  ' + '-' * 44)
    for nm, oof, wi_eq, wi_opt in zip(names, oofs, w0, w_opt):
        cv = roc_auc_score(y_train, oof)
        print(f'  {nm:<14} {wi_eq:>6.3f}  {wi_opt:>6.3f}  {cv:>10.5f}')

    eq_cv  = roc_auc_score(y_train, sum(w*o for w,o in zip(w0, oofs)))
    opt_cv = roc_auc_score(y_train, sum(w*o for w,o in zip(w_opt, oofs)))
    print('  ' + '-' * 44)
    print(f'  {"균등 앙상블":<14} {"":>6}  {"":>6}  {eq_cv:>10.5f}')
    print(f'  {"최적 앙상블":<14} {"":>6}  {"":>6}  {opt_cv:>10.5f}')
    return w_opt, opt_cv


# ====================================================
# 8. 저장
# ====================================================
def rank_norm(arr):
    return rankdata(arr) / len(arr)


def save_submissions(y_train, sub, results, names):
    print('\n[7] 제출 파일 저장')
    oofs  = [r[0] for r in results]
    preds = [r[1] for r in results]
    cvs   = np.array([r[2] for r in results])

    # AUC 가중 균등
    w_eq   = cvs / cvs.sum()
    l1_oof  = sum(w*o for w,o in zip(w_eq, oofs))
    l1_pred = sum(w*p for w,p in zip(w_eq, preds))
    l1_cv   = roc_auc_score(y_train, l1_oof)

    # 최적 가중치
    w_opt, opt_cv = optimize_weights(y_train, oofs, names)
    opt_pred = sum(w*p for w,p in zip(w_opt, preds))

    print(f'\n  L1 (AUC 가중) CV : {l1_cv:.5f}')
    print(f'  최적 가중치   CV : {opt_cv:.5f}')

    os.makedirs(SAVE_PATH, exist_ok=True)
    ts = datetime.now().strftime('%m%d_%H%M')

    f1 = f'{SAVE_PATH}submission_{ts}_L1_auc{str(l1_cv)[:7].replace(".", "p")}.csv'
    sub_l1 = sub.copy();  sub_l1['probability'] = rank_norm(l1_pred);  sub_l1.to_csv(f1, index=False)

    f2 = f'{SAVE_PATH}submission_{ts}_OPT_auc{str(opt_cv)[:7].replace(".", "p")}.csv'
    sub_op = sub.copy();  sub_op['probability'] = rank_norm(opt_pred);  sub_op.to_csv(f2, index=False)

    print(f'\n  [1] L1  → {f1}')
    print(f'  [2] OPT → {f2}')
    print(f'\n  ★ 더 높은 파일을 제출하세요')
    print(f'    LB 예상: v11 LB=0.74209 + 신규피처 + XGB/CAT Optuna = 0.742~0.743 목표')


# ====================================================
# 메인
# ====================================================
if __name__ == '__main__':
    train, test, sub = load_data()
    X_train, y_train, X_test, feat_cols = preprocess(train, test)
    X_train, X_test, feat_cols, base_imp_df = select_features_with_importance(X_train, y_train, X_test, feat_cols)

    if USE_FEATURE_DIET:
        X_train, X_test, feat_cols, _, _ = run_feature_diet(X_train, y_train, X_test, feat_cols, base_imp_df)

    if USE_OPTUNA:
        best_lgb, best_xgb, best_cat = run_optuna_all(X_train, y_train)
    else:
        best_lgb, best_xgb, best_cat = V11_LGB, V8_XGB, V8_CAT

    results, names = train_all_models(X_train, y_train, X_test, best_lgb, best_xgb, best_cat)
    save_submissions(y_train, sub, results, names)
