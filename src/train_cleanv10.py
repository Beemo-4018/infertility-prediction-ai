# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v_clean10 (배아 저장/활용 피처 추가)
평가 지표  : ROC-AUC
실행 방법  : python src/train_vclean10.py

[v_clean7 대비 변경사항]
    - 제거: 배반포_이식추정 (혼합_이식_간격과 중복, v_clean8에서 제거됨)
    - 제거: 클리닉_나이별성공률 (-0.00018, feature_search 음수)
    - 유지: 시술시기코드_성공률 (클리닉_나이별성공률 제거하므로 유지)
    - 추가: 배아_저장비율      +0.00558 (압도적 1위)
    - 추가: 배아_활용률        +0.00551 (2위)
    - 추가: 파트너정자_비율    +0.00052
    - 추가: DI_경험            +0.00027
    - 추가: 남성_불임원인_수   +0.00020
    - 추가: 여성_불임원인_수   +0.00018
    - 추가: 클리닉_집중도      +0.00015
    - 추가: 남성요인_ICSI매칭  +0.00013
    - 추가: IVF_경험           +0.00010

[Data Leakage 방지 원칙]
    1. LabelEncoder는 train 데이터로만 fit, test는 transform만 수행
    2. 결측치 보간 통계값(median)은 train 기준으로만 계산 후 test에 적용
    3. 파생 변수는 각 행(row) 내 연산만 수행
    4. 클리닉 집계는 K-Fold OOF 방식으로 leakage 차단
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
# 피처 목록
# ====================================================
CLEAN_FEATURES = [
    # A. 환자 상태
    '나이_수치',
    '과거_임신성공률',

    # B. 시술 정보
    '시술_ICSI',
    '배란 자극 여부',

    # C. 배아 품질 (기존)
    '이식된 배아 수',
    '배아_이식비율',
    '미세주입_성공률',

    # C-2. 배아 저장/활용 (신규) ★
    '배아_저장비율',        # +0.00558 저장된배아수 / (총생성배아수 + 1e-6)
    '배아_활용률',          # +0.00551 (이식+저장) / (총생성배아수 + 1e-6)

    # D. 불임 원인 (기존)
    '총_불임원인_수',
    '불명확_단독원인',

    # D-2. 불임 원인 세분화 (신규)
    '남성_불임원인_수',     # +0.00020
    '여성_불임원인_수',     # +0.00018

    # E. 과거 이력 (기존)
    '총 시술 횟수_num',
    'failure_streak',
    '출산_경험',

    # E-2. 시술 경험 세분화 (신규)
    'IVF_경험',             # +0.00010
    'DI_경험',              # +0.00027

    # F. 클리닉
    '시술시기코드_성공률',
    '시술시기코드_시술건수',
    '클리닉_집중도',        # +0.00015

    # G. 시술 유형
    'IVF시술_여부',

    # H. 시간 간격
    '혼합_이식_간격',
    '해동_이식_간격',

    # I. 교호작용 (신규)
    '파트너정자_비율',      # +0.00052
    '남성요인_ICSI매칭',    # +0.00013
]


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
# 2. 전처리
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 핵심 피처 생성')
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

        # ── A. 환자 상태 ─────────────────────────────────────
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)

        for col in count_cols:
            df[col + '_num'] = df[col].apply(parse_count)

        df['과거_임신성공률'] = (
            df['총 임신 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )
        df['출산_경험'] = (df['총 출산 횟수_num'] > 0).astype(int)

        # ── B. 시술 정보 ──────────────────────────────────────
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_ICSI']    = 시술유형.str.contains('ICSI', na=False).astype(int)
        df['IVF시술_여부'] = (df['시술 유형'] == 'IVF').astype(int)

        # ── C. 배아 품질 (기존) ───────────────────────────────
        df['배아_이식비율'] = (
            df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        )

        # ── C-2. 배아 저장/활용 피처 (신규) ★ ────────────────
        df['배아_저장비율'] = (
            df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['배아_활용률'] = (
            (df['이식된 배아 수'] + df['저장된 배아 수'])
            / (df['총 생성 배아 수'] + 1e-6)
        )

        # ── D. 불임 원인 (기존) ───────────────────────────────
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

        # ── D-2. 불임 원인 세분화 (신규) ─────────────────────
        male_flags = [c for c in (
            ['남성 주 불임 원인', '남성 부 불임 원인'] + male_cause_cols
        ) if c in df.columns]
        df['남성_불임원인_수'] = df[male_flags].sum(axis=1)

        female_flags = [c for c in (
            ['여성 주 불임 원인', '여성 부 불임 원인'] + female_cause_cols
        ) if c in df.columns]
        df['여성_불임원인_수'] = df[female_flags].sum(axis=1)

        # ── E. 과거 이력 (기존) ───────────────────────────────
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)

        # ── E-2. 시술 경험 세분화 (신규) ─────────────────────
        df['IVF_경험'] = df['IVF 시술 횟수_num'].fillna(0)
        df['DI_경험']  = df['DI 시술 횟수_num'].fillna(0)

        # ── H. 시간 간격 (행 내 연산 → leakage 없음) ─────────
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']

        # ── I. 교호작용 피처 (신규) ───────────────────────────
        # 파트너정자_비율: 공여 정자 사용 아님 = 파트너 정자  (+0.00052)
        if '정자 출처' in df.columns:
            df['파트너정자_비율'] = (df['정자 출처'] == '파트너').astype(int)
        elif '파트너 정자 사용 여부' in df.columns:
            df['파트너정자_비율'] = df['파트너 정자 사용 여부'].fillna(0).astype(int)
        else:
            # IVF이면서 DI가 아닌 경우 파트너 정자 추정
            df['파트너정자_비율'] = df['IVF시술_여부'].copy()

        # 남성요인_ICSI매칭: 남성불임 AND ICSI 교호작용  (+0.00013)
        df['남성요인_ICSI매칭'] = (
            (df['남성_불임원인_수'] > 0) & (df['시술_ICSI'] == 1)
        ).astype(int)

    # ── 결측치 처리 (train median만 사용) ────────────────────
    num_base = [
        c for c in CLEAN_FEATURES
        if c in train.select_dtypes(include=np.number).columns
    ]
    medians = train[num_base].median()
    train[num_base] = train[num_base].fillna(medians)
    test[num_base]  = test[num_base].fillna(medians)

    # ── F. 클리닉 집계 피처 (K-Fold OOF) ────────────────────
    print('  클리닉 집계 피처 생성 중...')
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

    # 클리닉_집중도: 클리닉별 전체 시술 건수 (train 기준)  (+0.00015)
    clinic_size = (
        train.groupby(clinic_col).size().reset_index(name='클리닉_집중도')
    )
    train = train.merge(clinic_size, on=clinic_col, how='left')
    test  = test.merge(clinic_size, on=clinic_col, how='left')
    train['클리닉_집중도'] = train['클리닉_집중도'].fillna(0)
    test['클리닉_집중도']  = test['클리닉_집중도'].fillna(0)

    # ── 최종 피처 확인 ────────────────────────────────────────
    feature_cols = [f for f in CLEAN_FEATURES if f in train.columns]
    missing = [f for f in CLEAN_FEATURES if f not in train.columns]
    if missing:
        print(f'\n  [경고] 생성 안 된 피처: {missing}')
        print('  → 원본 컬럼명 확인 후 preprocess() 수정 필요')

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    print(f'\n  [피처 목록]')
    for col in feature_cols:
        print(f'    {col}')
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
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        m = lgb.LGBMClassifier(**params, n_estimators=3000)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
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

    print('\n  [피처 중요도]')
    print(f'  {"피처":<30} {"중요도":>8}')
    print('  ' + '-' * 42)
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] / importance['importance'].max() * 20)
        print(f'  {row["feature"]:<30} {row["importance"]:>8.0f}  {bar}')

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
        'max_depth'       : 6,
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
            **params, n_estimators=3000, early_stopping_rounds=100
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
            iterations=3000,
            learning_rate=0.02,
            depth=6,
            eval_metric='AUC',
            random_seed=SEED,
            verbose=False,
            early_stopping_rounds=100,
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
    filename  = f'{SAVE_PATH}submission_{timestamp}_vclean10_auc{auc_str}.csv'

    sub['probability'] = ensemble_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')
    print(f'\n  ✅ 다음: v_clean10 제출파일 + v5(LB 0.74198) rank blend → 제출')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v_clean10)')
    print('  베이스: v_clean7 | 신규 피처 9개 추가')
    print('  핵심: 배아_저장비율 +0.00558, 배아_활용률 +0.00551')
    print('=' * 55)

    train, test, sub                       = load_data()
    X_train, y_train, X_test, feature_cols = preprocess(train, test)

    lgb_result = train_lgb(X_train, y_train, X_test)
    xgb_result = train_xgb(X_train, y_train, X_test)
    cat_result = train_cat(X_train, y_train, X_test)

    ensemble_and_save(y_train, sub, lgb_result, xgb_result, cat_result)