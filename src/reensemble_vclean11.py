# -*- coding: utf-8 -*-
"""
reensemble_vclean11.py
v_clean11 OOF를 CatBoost 60% 비중으로 재앙상블
실행: python src/reensemble_vclean11.py

[비교]
  기존 33:33:34 → CV 0.73962
  신규 20:20:60 → CV ???
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

SEED    = 42
TARGET  = '임신 성공 여부'
DATA_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
N_FOLDS = 5


# ====================================================
# 전처리 (v_clean11 동일)
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


def load_and_preprocess():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    sub   = pd.read_csv(DATA_PATH + 'sample_submission.csv')

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
        df['나이_수치']     = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

        for col in count_cols:
            df[col + '_num'] = df[col].apply(parse_count)

        df['과거_임신성공률'] = df['총 임신 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        df['과거_출산성공률'] = df['총 출산 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        df['IVF_임신성공률'] = df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['출산_경험']     = (df['총 출산 횟수_num'] > 0).astype(int)
        df['임신_경험']     = (df['총 임신 횟수_num'] > 0).astype(int)
        df['IVF_경험']      = df['IVF 시술 횟수_num'].fillna(0)
        df['DI_경험']       = df['DI 시술 횟수_num'].fillna(0)
        df['반복시술_여부'] = (df['총 시술 횟수_num'] >= 3).astype(int)
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)

        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']    = 시술유형.str.contains('ICSI', na=False).astype(int)
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)
        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

        df['배아_이식비율'] = df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율'] = df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        df['배아_활용률']   = (
            (df['이식된 배아 수'] + df['저장된 배아 수'])
            / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        )
        df['파트너정자_비율'] = (
            df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)
        )
        df['난자_수정률'] = (
            df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1e-6)
        )
        df['신선배아_이식여부'] = (
            (df['저장된 배아 수'] == 0) & (df['이식된 배아 수'] > 0)
        ).astype(int)
        df['잉여배아_유무'] = (
            (df['총 생성 배아 수'] - df['이식된 배아 수'] - df['저장된 배아 수']) > 0
        ).astype(int)

        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        df['긴_배양_여부']  = (df['혼합_이식_간격'] >= 5).astype(int)
        df['해동_당일이식'] = (df['해동_이식_간격'] == 0).astype(int)

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

        df['남성요인_ICSI매칭'] = (
            (df['남성_불임원인_수'] > 0) & (df['시술_ICSI'] == 1)
        ).astype(int)
        df['배란장애_자극매칭'] = (
            (df['불임 원인 - 배란 장애'] == 1) & (df['배란 자극 여부'] == 1)
        ).astype(int)
        df['고령_동결배아조합'] = (
            (df['고령_여부'] == 1) & (df['동결배아_시술'] == 1)
        ).astype(int)
        df['나이x배아활용률'] = df['나이_수치'] * df['배아_활용률']
        df['클리닉내_시술비율'] = (
            df['클리닉 내 총 시술 횟수_num'].fillna(0)
            / (df['총 시술 횟수_num'].fillna(0) + 1e-6)
        )
        df['클리닉_나이조합'] = (
            df['시술 시기 코드'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
        )
        df['클리닉_시술유형조합'] = (
            df['시술 시기 코드'].astype(str) + '_' + df['특정 시술 유형'].astype(str)
        )

    num_cols = train.select_dtypes(include=np.number).columns.tolist()
    excl = [TARGET] + [c + '_num' for c in count_cols]
    num_base = [c for c in num_cols if c not in excl]
    medians = train[num_base].median()
    train[num_base] = train[num_base].fillna(medians)
    test[num_base]  = test[num_base].fillna(medians)

    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_c = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    tr_rate = np.zeros(len(train))
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        stats = train.iloc[tr_idx].groupby(clinic_col)[TARGET].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * 20) / (stats['count'] + 20)
        tr_rate[val_idx] = train.iloc[val_idx][clinic_col].map(smoothed).fillna(global_mean).values
    full_stats  = train.groupby(clinic_col)[TARGET].agg(['mean', 'count'])
    full_smooth = (full_stats['mean'] * full_stats['count'] + global_mean * 20) / (full_stats['count'] + 20)
    train['시술시기코드_성공률'] = tr_rate
    test['시술시기코드_성공률']  = test[clinic_col].map(full_smooth).fillna(global_mean).values

    tr_cnt = np.zeros(len(train))
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        cnt_map = train.iloc[tr_idx].groupby(clinic_col).size()
        tr_cnt[val_idx] = train.iloc[val_idx][clinic_col].map(cnt_map).fillna(1).values
    full_cnt = train.groupby(clinic_col).size()
    train['시술시기코드_시술건수'] = np.log1p(tr_cnt)
    test['시술시기코드_시술건수']  = np.log1p(test[clinic_col].map(full_cnt).fillna(1).values)

    train['시술시기코드_성공률편차'] = train['시술시기코드_성공률'] - global_mean
    test['시술시기코드_성공률편차']  = test['시술시기코드_성공률']  - global_mean

    tr_enc, te_enc = target_encode(train, test, '클리닉_나이조합', TARGET)
    train['클리닉_나이별성공률'] = tr_enc
    test['클리닉_나이별성공률']  = te_enc

    tr_enc2, te_enc2 = target_encode(train, test, '클리닉_시술유형조합', TARGET)
    train['클리닉_시술유형별성공률'] = tr_enc2
    test['클리닉_시술유형별성공률']  = te_enc2

    tr_emb = np.zeros(len(train))
    emb_global = train['이식된 배아 수'].mean()
    for tr_idx, val_idx in skf_c.split(train, train[TARGET]):
        emb_map = train.iloc[tr_idx].groupby(clinic_col)['이식된 배아 수'].mean()
        tr_emb[val_idx] = train.iloc[val_idx][clinic_col].map(emb_map).fillna(emb_global).values
    full_emb = train.groupby(clinic_col)['이식된 배아 수'].mean()
    train['시술시기코드_배아이식수평균'] = tr_emb
    test['시술시기코드_배아이식수평균']  = test[clinic_col].map(full_emb).fillna(emb_global).values

    clinic_size = train.groupby(clinic_col).size().reset_index(name='클리닉_집중도')
    train = train.merge(clinic_size, on=clinic_col, how='left')
    test  = test.merge(clinic_size, on=clinic_col, how='left')
    train['클리닉_집중도'] = train['클리닉_집중도'].fillna(0)
    test['클리닉_집중도']  = test['클리닉_집중도'].fillna(0)

    te_cols = ['특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '난자 출처', '정자 출처']
    for col in te_cols:
        if col in train.columns:
            train[col] = train[col].fillna('Unknown')
            test[col]  = test[col].fillna('Unknown')
            tr_enc, te_enc = target_encode(train, test, col, TARGET)
            train[col + '_te'] = tr_enc
            test[col + '_te']  = te_enc

    FEATURES = [
        '나이_수치', '고령_여부', '초고령_여부', '최적연령_여부',
        '과거_임신성공률', '과거_출산성공률', 'IVF_임신성공률',
        '출산_경험', '임신_경험', 'IVF_경험', 'DI_경험',
        '반복시술_여부', 'failure_streak', '총 시술 횟수_num',
        '시술_ICSI', '배란 자극 여부', 'IVF시술_여부', '동결배아_시술',
        '이식된 배아 수', '배아_이식비율', '배아_저장비율', '배아_활용률',
        '미세주입_성공률', '파트너정자_비율', '난자_수정률',
        '신선배아_이식여부', '잉여배아_유무',
        '총_불임원인_수', '불명확_단독원인', '남성_불임원인_수', '여성_불임원인_수',
        '혼합_이식_간격', '해동_이식_간격', '채취_이식_간격', '채취_혼합_간격',
        '긴_배양_여부', '해동_당일이식',
        '남성요인_ICSI매칭', '배란장애_자극매칭', '고령_동결배아조합', '나이x배아활용률',
        '시술시기코드_성공률', '시술시기코드_시술건수', '시술시기코드_성공률편차',
        '시술시기코드_배아이식수평균', '클리닉_나이별성공률', '클리닉_시술유형별성공률',
        '클리닉_집중도', '클리닉내_시술비율',
        '특정 시술 유형_te', '배란 유도 유형_te',
        '배아 생성 주요 이유_te', '난자 출처_te', '정자 출처_te',
    ]
    feature_cols = [f for f in FEATURES if f in train.columns]
    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train[TARGET]
    X_test  = test[feature_cols].fillna(train[feature_cols].median())

    return X_train, y_train, X_test, sub


# ====================================================
# 학습 (OOF + test pred 반환)
# ====================================================
def run_lgb(X_train, y_train, X_test):
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbose': -1,
        'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': -1,
        'min_child_samples': 20, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'random_state': SEED, 'n_jobs': -1
    }
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        m = lgb.LGBMClassifier(**params, n_estimators=5000)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
    cv = roc_auc_score(y_train, oof)
    print(f'  LGB  CV: {cv:.5f}')
    return oof, pred, cv


def run_xgb(X_train, y_train, X_test):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'learning_rate': 0.02, 'max_depth': 7,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': SEED, 'n_jobs': -1, 'verbosity': 0, 'tree_method': 'hist'
    }
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        m = xgb.XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=200)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])], verbose=False)
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
    cv = roc_auc_score(y_train, oof)
    print(f'  XGB  CV: {cv:.5f}')
    return oof, pred, cv


def run_cat(X_train, y_train, X_test):
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.02, depth=7,
            eval_metric='AUC', random_seed=SEED,
            verbose=False, early_stopping_rounds=200, task_type='CPU'
        )
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]))
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        pred        += m.predict_proba(X_test)[:, 1] / N_FOLDS
    cv = roc_auc_score(y_train, oof)
    print(f'  CAT  CV: {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 메인
# ====================================================
if __name__ == '__main__':
    from datetime import datetime

    print('=' * 55)
    print('  v_clean11 가중 앙상블 비교')
    print('=' * 55)

    print('\n데이터 로드 & 전처리 중...')
    X_train, y_train, X_test, sub = load_and_preprocess()

    print('\n모델 학습 중...')
    lgb_oof, lgb_pred, lgb_cv = run_lgb(X_train, y_train, X_test)
    xgb_oof, xgb_pred, xgb_cv = run_xgb(X_train, y_train, X_test)
    cat_oof, cat_pred, cat_cv = run_cat(X_train, y_train, X_test)

    print('\n' + '=' * 55)
    print('  앙상블 비율 비교')
    print('=' * 55)

    configs = [
        ('균등  33:33:34', [1/3, 1/3, 1/3]),
        ('Cat60 20:20:60', [0.20, 0.20, 0.60]),
        ('Cat70 15:15:70', [0.15, 0.15, 0.70]),
        ('Cat단독 0:0:100', [0.00, 0.00, 1.00]),
    ]

    best_cv   = 0
    best_pred = None
    best_name = ''

    for name, (wl, wx, wc) in configs:
        oof_blend = wl * lgb_oof + wx * xgb_oof + wc * cat_oof
        cv = roc_auc_score(y_train, oof_blend)
        print(f'  {name}  →  CV: {cv:.5f}')
        if cv > best_cv:
            best_cv   = cv
            best_pred = wl * lgb_pred + wx * xgb_pred + wc * cat_pred
            best_name = name

    print(f'\n  ✅ 최고: {best_name}  CV: {best_cv:.5f}')
    print(f'     기존 33:33:34 대비: {best_cv - 0.73962:+.5f}')

    # 최고 비율로 저장
    timestamp = datetime.now().strftime('%m%d_%H%M')
    auc_str   = f'{best_cv:.5f}'.replace('.', 'p')
    filename  = f'{SAVE_PATH}submission_{timestamp}_vclean11_weighted_auc{auc_str}.csv'
    sub['probability'] = best_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료: {filename}')