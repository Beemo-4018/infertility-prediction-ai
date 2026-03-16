# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v2 (피처 강화 + 파라미터 튜닝)
평가 지표  : ROC-AUC
실행 방법  : python train_v2.py

개발 환경:
    OS           : macOS / Windows / Linux
    Python       : 3.9+
    pandas       : 2.0+
    numpy        : 1.24+
    scikit-learn : 1.3+
    lightgbm     : 4.0+
    xgboost      : 2.0+
    catboost     : 1.2+
    optuna       : 3.0+

[v2 변경사항]
    1. 피처 엔지니어링 강화
       - 시간 간격 파생 피처 추가 (이식-채취 경과일 차이 등)
       - 불임 원인 조합 피처 추가 (총 불임 원인 수, 복합 원인 여부)
       - 배아 품질 관련 파생 피처 추가
       - 나이 그룹 기반 추가 피처
    2. 하이퍼파라미터 조정
       - learning_rate: 0.05 → 0.02 (더 세밀한 학습)
       - n_estimators: 3000 → 5000
       - early_stopping: 100 → 200
    3. Optuna 하이퍼파라미터 튜닝 추가 (LightGBM 기준)

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
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ====================================================
# 설정값
# ====================================================
SEED          = 42
N_FOLDS       = 5
TARGET      = '임신 성공 여부'
DATA_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
USE_OPTUNA    = True   # False로 바꾸면 기본 파라미터로 실행 (빠름)
OPTUNA_TRIALS  = 50    # Optuna 탐색 횟수 (시간 여유 있으면 100으로)


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
# 2. 전처리 & 피처 엔지니어링 (v2 강화)
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 피처 엔지니어링 (v2)')
    print('=' * 55)

    train = train.copy()
    test  = test.copy()

    # ── 2-1. 나이 수치화 ────────────────────────────────────────
    age_map = {
        '만18-34세': 26, '만35-37세': 36, '만38-39세': 38,
        '만40-42세': 41, '만43-44세': 43, '만45-50세': 47, '알 수 없음': -1
    }
    for df in [train, test]:
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)
        # 나이 그룹 (고령 여부)
        df['고령_여부']     = (df['나이_수치'] >= 38).astype(int)
        df['초고령_여부']   = (df['나이_수치'] >= 43).astype(int)
        df['최적연령_여부'] = (df['나이_수치'] <= 36).astype(int)

    # ── 2-2. 횟수 컬럼 수치화 ───────────────────────────────────
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

    # ── 2-3. 파생 피처 (행(row) 내 연산만 → leakage 없음) ───────
    for df in [train, test]:

        # [A] 과거 이력 기반 성공률
        df['과거_임신성공률']  = df['총 임신 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['과거_출산성공률']  = df['총 출산 횟수_num']  / (df['총 시술 횟수_num'] + 1e-6)
        df['IVF_임신성공률']  = df['IVF 임신 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['IVF_출산성공률']  = df['IVF 출산 횟수_num'] / (df['IVF 시술 횟수_num'] + 1e-6)
        df['DI_임신성공률']   = df['DI 임신 횟수_num']  / (df['DI 시술 횟수_num'] + 1e-6)

        # [B] 시술 경험 플래그
        df['IVF_경험']      = (df['IVF 시술 횟수_num'] > 0).astype(int)
        df['DI_경험']       = (df['DI 시술 횟수_num']  > 0).astype(int)
        df['임신_경험']     = (df['총 임신 횟수_num']   > 0).astype(int)
        df['출산_경험']     = (df['총 출산 횟수_num']   > 0).astype(int)
        df['반복시술_여부'] = (df['총 시술 횟수_num']   >= 3).astype(int)  # 3회 이상 반복
        df['클리닉_집중도'] = df['클리닉 내 총 시술 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)

        # [C] 배아 관련 비율
        df['배아_이식비율']   = df['이식된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_저장비율']   = df['저장된 배아 수']   / (df['총 생성 배아 수'] + 1e-6)
        df['배아_활용률']     = (df['이식된 배아 수'] + df['저장된 배아 수']) / (df['총 생성 배아 수'] + 1e-6)
        df['미세주입_성공률'] = df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        df['미세주입_이식률'] = df['미세주입 배아 이식 수'] / (df['미세주입에서 생성된 배아 수'] + 1e-6)
        df['난자_수정률']     = df['혼합된 난자 수']   / (df['수집된 신선 난자 수'] + 1e-6)
        df['파트너정자_비율'] = df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1e-6)

        # [D] 시간 간격 파생 피처 (행 내 연산 → leakage 없음)
        # 이식까지 걸린 총 일수 (채취 → 이식)
        df['채취_이식_간격'] = df['배아 이식 경과일'] - df['난자 채취 경과일']
        # 채취 → 혼합 간격 (수정 시점)
        df['채취_혼합_간격'] = df['난자 혼합 경과일'] - df['난자 채취 경과일']
        # 혼합 → 이식 간격 (배양 기간, 클수록 배반포 이식)
        df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
        # 해동 → 이식 간격
        df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
        # 배반포 이식 여부 추정 (혼합→이식 5일 이상)
        df['배반포_이식추정'] = (df['혼합_이식_간격'] >= 5).astype(int)

        # [E] 불임 원인 조합 피처 (행 내 합산 → leakage 없음)
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

        df['남성_불임원인_수']  = df[male_cause_cols].sum(axis=1)
        df['여성_불임원인_수']  = df[female_cause_cols].sum(axis=1)
        df['총_불임원인_수']    = df[all_cause_cols].sum(axis=1)
        df['복합_불임원인']     = (df['총_불임원인_수'] >= 2).astype(int)
        df['불명확_단독원인']   = (
            (df['불명확 불임 원인'] == 1) & (df['총_불임원인_수'] == 1)
        ).astype(int)

        # [F] 결측 여부 피처 (행 단위 → leakage 없음)
        for col in ['착상 전 유전 검사 사용 여부', 'PGD 시술 여부',
                    'PGS 시술 여부', '난자 해동 경과일', '배아 해동 경과일',
                    '임신 시도 또는 마지막 임신 경과 연수']:
            df[col + '_결측'] = df[col].isnull().astype(int)

        # [G] 동결 배아 시술 여부 (해동 배아 수 > 0)
        df['동결배아_시술'] = (df['해동된 배아 수'] > 0).astype(int)

    # ── 2-4. 피처 컬럼 확정 ────────────────────────────────────
    drop_cols    = ['ID', TARGET] + count_cols
    feature_cols = [c for c in train.columns if c not in drop_cols]

    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train[feature_cols].select_dtypes(include='object').columns.tolist()

    # ── 2-5. 결측치 처리 ────────────────────────────────────────
    # train median으로만 계산 후 test에 적용 (test 통계 사용 금지)
    medians = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)

    train[cat_cols] = train[cat_cols].fillna('Unknown')
    test[cat_cols]  = test[cat_cols].fillna('Unknown')

    # ── 2-6. Label Encoding ─────────────────────────────────────
    # train으로만 fit → test는 transform만 (합산 fit 금지)
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col].astype(str))

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
# 3. Optuna 하이퍼파라미터 튜닝 (LightGBM 기준)
# ====================================================
def optuna_lgb(X_train, y_train, n_trials=50):
    print('\n' + '=' * 55)
    print(f'[3-0] Optuna 튜닝 (LightGBM, {n_trials} trials)')
    print('=' * 55)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

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
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = lgb.LGBMClassifier(**params, n_estimators=2000)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
            aucs.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n  최적 CV AUC  : {study.best_value:.5f}')
    print(f'  최적 파라미터 : {study.best_params}')
    return study.best_params


# ====================================================
# 4-1. LightGBM 학습
# ====================================================
def train_lgb(X_train, y_train, X_test, best_params=None):
    print('\n' + '=' * 55)
    print('[4-1] LightGBM (Stratified 5-Fold)')
    print('=' * 55)

    # Optuna 결과 있으면 사용, 없으면 기본 파라미터
    if best_params:
        params = {**best_params, 'objective': 'binary', 'metric': 'auc',
                'verbose': -1, 'random_state': SEED, 'n_jobs': -1}
        print('  → Optuna 최적 파라미터 사용')
    else:
        params = {
            'objective'        : 'binary',
            'metric'           : 'auc',
            'learning_rate'    : 0.02,   # v2: 0.05 → 0.02
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

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = lgb.LGBMClassifier(**params, n_estimators=5000)   # v2: 3000 → 5000
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
        )
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})')

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')
    return oof, pred, cv


# ====================================================
# 4-2. XGBoost 학습
# ====================================================
def train_xgb(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print('[4-2] XGBoost (Stratified 5-Fold)')
    print('=' * 55)

    params = {
        'objective'        : 'binary:logistic',
        'eval_metric'      : 'auc',
        'learning_rate'    : 0.02,   # v2: 0.05 → 0.02
        'max_depth'        : 7,
        'subsample'        : 0.8,
        'colsample_bytree' : 0.8,
        'min_child_weight' : 5,
        'reg_alpha'        : 0.1,
        'reg_lambda'       : 1.0,
        'random_state'     : SEED,
        'n_jobs'           : -1,
        'verbosity'        : 0,
        'tree_method'      : 'hist'
    }

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = xgb.XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=200)
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
# 4-3. CatBoost 학습
# ====================================================
def train_cat(X_train, y_train, X_test):
    print('\n' + '=' * 55)
    print('[4-3] CatBoost (Stratified 5-Fold)')
    print('=' * 55)

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = CatBoostClassifier(
            iterations            = 5000,   # v2: 3000 → 5000
            learning_rate         = 0.02,   # v2: 0.05 → 0.02
            depth                 = 7,
            eval_metric           = 'AUC',
            random_seed           = SEED,
            verbose               = False,
            early_stopping_rounds = 200,    # v2: 100 → 200
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
# 5. 앙상블 & 제출 파일 저장
# ====================================================
def ensemble_and_save(y_train, sub, results):
    print('\n' + '=' * 55)
    print('[5] 앙상블 & 제출 파일 저장')
    print('=' * 55)

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
    print('  ' + '-' * 38)
    for name, cv, w in zip(names, cvs, weights):
        print(f'  {name:<12} {cv:>10.5f}  {w:>8.3f}')
    print('  ' + '-' * 38)
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
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v2)')
    print('=' * 55)

    # 1. 데이터 로드
    train, test, sub = load_data()

    # 2. 전처리
    X_train, y_train, X_test, feature_cols = preprocess(train, test)

    # 3. Optuna 튜닝 (USE_OPTUNA=True 일 때만 실행)
    best_params = None
    if USE_OPTUNA:
        best_params = optuna_lgb(X_train, y_train, n_trials=OPTUNA_TRIALS)

    # 4. 모델 학습
    print('\n' + '=' * 55)
    print('[4] 모델 학습')
    print('=' * 55)
    lgb_result = train_lgb(X_train, y_train, X_test, best_params)
    xgb_result = train_xgb(X_train, y_train, X_test)
    cat_result = train_cat(X_train, y_train, X_test)

    # 5. 앙상블 & 저장
    ensemble_and_save(y_train, sub, [lgb_result, xgb_result, cat_result])