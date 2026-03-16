# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - 학습 및 추론 스크립트
평가 지표  : ROC-AUC
실행 방법  : python train.py

개발 환경:
    OS           : macOS / Windows / Linux
    Python       : 3.9+
    pandas       : 2.0+
    numpy        : 1.24+
    scikit-learn : 1.3+
    lightgbm     : 4.0+
    xgboost      : 2.0+
    catboost     : 1.2+

[Data Leakage 방지 원칙]
    1. LabelEncoder는 train 데이터로만 fit, test는 transform만 수행
    2. 결측치 보간 통계값(median)은 train 기준으로만 계산 후 test에 적용
    3. 파생 변수는 각 행(row) 내 연산만 수행
    4. test 데이터의 통계/분포 정보를 학습에 활용하지 않음
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
# 1. 데이터 로드
# ====================================================
def load_data():
    print('=' * 50)
    print('[1] 데이터 로드')
    print('=' * 50)
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    sub   = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    print(f'  train shape : {train.shape}')
    print(f'  test  shape : {test.shape}')
    success = train[TARGET].mean()
    print(f'  임신 성공 비율 : {success:.4f} ({success*100:.2f}%)')
    return train, test, sub


# ====================================================
# 2. 전처리 & 피처 엔지니어링
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 50)
    print('[2] 전처리 & 피처 엔지니어링')
    print('=' * 50)

    train = train.copy()
    test  = test.copy()

    # ── 2-1. 나이 수치화 (단순 매핑, leakage 없음) ──────────────
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)

    # ── 2-2. 횟수 컬럼 수치화 ('0회'~'6회 이상' → 숫자) ─────────
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
        except:
            return np.nan

    for col in count_cols:
        for df in [train, test]:
            df[col + '_num'] = df[col].apply(parse_count)

    # ── 2-3. 파생 피처 (행(row) 내 연산만 수행 → leakage 없음) ──
    for df in [train, test]:
        # 과거 임신/출산 성공률
        df['과거_임신성공률']  = df['총 임신 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['과거_출산성공률']  = df['총 출산 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)

        # IVF 특화 비율
        df['IVF_임신성공률'] = df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['IVF_출산성공률'] = df['IVF 출산 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)

        # 배아 관련 비율
        df['배아_이식비율']   = df['이식된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율']   = df['저장된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['미세주입_성공률'] = df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        df['난자_수정률']     = df['혼합된 난자 수']   / (df['수집된 신선 난자 수'] + 1e-6)
        df['배아_활용률']     = (df['이식된 배아 수'] + df['저장된 배아 수']) / (df['총 생성 배아 수'] + 1e-6)

        # 시술 경험 유무 (이진 플래그)
        df['IVF_경험']  = (df['IVF 시술 횟수_num'] > 0).astype(int)
        df['DI_경험']   = (df['DI 시술 횟수_num']  > 0).astype(int)
        df['임신_경험'] = (df['총 임신 횟수_num']   > 0).astype(int)
        df['출산_경험'] = (df['총 출산 횟수_num']   > 0).astype(int)

        # 클리닉 집중도
        df['클리닉_집중도'] = df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)

        # 결측 여부 자체를 피처로 (행 단위 → leakage 없음)
        for col in ['착상 전 유전 검사 사용 여부', 'PGD 시술 여부',
                    'PGS 시술 여부', '난자 해동 경과일', '배아 해동 경과일']:
            df[col + '_결측'] = df[col].isnull().astype(int)

    # ── 2-4. 피처 컬럼 확정 ──────────────────────────────────────
    drop_cols    = ['ID', TARGET] + count_cols
    feature_cols = [c for c in train.columns if c not in drop_cols]

    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train[feature_cols].select_dtypes(include='object').columns.tolist()

    # ── 2-5. 결측치 처리 ─────────────────────────────────────────
    # train median으로만 계산 후 test에 적용 (test 통계 사용 금지)
    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)

    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # ── 2-6. Label Encoding ──────────────────────────────────────
    # train으로만 fit → test는 transform만 수행 (합산 fit 금지)
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col].astype(str))   # train만 fit

        # test에 train에서 없던 카테고리 → 'Unknown'으로 대체
        known     = set(le.classes_)
        test[col] = test[col].astype(str).apply(
            lambda x: x if x in known else 'Unknown'
        )
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    print(f'  피처 수      : {len(feature_cols)}')
    print(f'  X_train      : {X_train.shape}')
    print(f'  X_test       : {X_test.shape}')
    print(f'  결측치 합계  : {X_train.isnull().sum().sum()}')
    return X_train, y_train, X_test, feature_cols


# ====================================================
# 3-1. LightGBM 학습
# ====================================================
def train_lgb(X_train, y_train, X_test):
    print('\n' + '=' * 50)
    print('[3-1] LightGBM (Stratified 5-Fold)')
    print('=' * 50)

    params = {
        'objective'        : 'binary',
        'metric'           : 'auc',
        'learning_rate'    : 0.05,
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
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = lgb.LGBMClassifier(**params, n_estimators=3000)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})')

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 3-2. XGBoost 학습
# ====================================================
def train_xgb(X_train, y_train, X_test):
    print('\n' + '=' * 50)
    print('[3-2] XGBoost (Stratified 5-Fold)')
    print('=' * 50)

    params = {
        'objective'        : 'binary:logistic',
        'eval_metric'      : 'auc',
        'learning_rate'    : 0.05,
        'max_depth'        : 7,
        'subsample'        : 0.8,
        'colsample_bytree' : 0.8,
        'min_child_weight' : 5,
        'reg_alpha'        : 0.1,
        'reg_lambda'       : 1.0,
        'random_state'     : SEED,
        'n_jobs'           : -1,
        'verbosity'        : 0,
        'tree_method'      : 'hist'   # CPU 최적화
    }

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = xgb.XGBClassifier(**params, n_estimators=3000, early_stopping_rounds=100)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration})')

    cv = roc_auc_score(y_train, oof)
    print(f'  XGBoost CV AUC  : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 3-3. CatBoost 학습
# ====================================================
def train_cat(X_train, y_train, X_test):
    print('\n' + '=' * 50)
    print('[3-3] CatBoost (Stratified 5-Fold)')
    print('=' * 50)

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = CatBoostClassifier(
            iterations            = 3000,
            learning_rate         = 0.05,
            depth                 = 7,
            eval_metric           = 'AUC',
            random_seed           = SEED,
            verbose               = False,
            early_stopping_rounds = 100,
            task_type             = 'CPU'
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
# 4. 앙상블 & 제출 파일 저장
# ====================================================
def ensemble_and_save(y_train, sub, results):
    print('\n' + '=' * 50)
    print('[4] 앙상블 & 제출 파일 저장')
    print('=' * 50)

    oofs  = [r[0] for r in results]
    preds = [r[1] for r in results]
    cvs   = np.array([r[2] for r in results])
    names = ['LightGBM', 'XGBoost', 'CatBoost']

    # CV AUC 기반 가중 평균 앙상블
    weights       = cvs / cvs.sum()
    ensemble_oof  = sum(w * o for w, o in zip(weights, oofs))
    ensemble_pred = sum(w * p for w, p in zip(weights, preds))
    ensemble_cv   = roc_auc_score(y_train, ensemble_oof)

    print(f'\n  {"모델":<12} {"CV AUC":>10}  {"weight":>8}')
    print('  ' + '-' * 36)
    for name, cv, w in zip(names, cvs, weights):
        print(f'  {name:<12} {cv:>10.5f}  {w:>8.3f}')
    print('  ' + '-' * 36)
    print(f'  {"앙상블":<12} {ensemble_cv:>10.5f}')

    # 제출 파일 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    auc_str   = f'{ensemble_cv:.5f}'.replace('.', 'p')
    filename  = f'{SAVE_PATH}submission_{timestamp}_auc{auc_str}.csv'

    sub[TARGET] = ensemble_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 50)
    print('  난임 환자 임신 성공 여부 예측')
    print('=' * 50)

    train, test, sub               = load_data()
    X_train, y_train, X_test, _    = preprocess(train, test)
    lgb_result                     = train_lgb(X_train, y_train, X_test)
    xgb_result                     = train_xgb(X_train, y_train, X_test)
    cat_result                     = train_cat(X_train, y_train, X_test)
    ensemble_and_save(y_train, sub, [lgb_result, xgb_result, cat_result])