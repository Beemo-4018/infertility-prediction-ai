# -*- coding: utf-8 -*-
"""
====================================================
난임 환자 임신 성공 여부 예측 - v_clean (Clean Baseline)
평가 지표  : ROC-AUC
실행 방법  : python src/train_vclean.py

[컨셉]
    - 피처를 무작정 늘리지 않고 의학적으로 의미 있는 것만 선택
    - 각 피처가 왜 들어갔는지 설명 가능
    - 오버피팅 없는 일반화 성능 확보
    - 여기서부터 하나씩 추가하며 LB 기여도 추적

[피처 선택 기준]
    1. 의학적으로 임신 성공에 직접 영향을 주는 것
    2. 데이터에서 실제로 신호가 있는 것
    3. 다른 피처와 독립적인 정보를 담은 것

[피처 그룹별 설명]
    A. 환자 상태 (3개)
       - 나이_수치: 나이가 많을수록 성공률 하락 (의학적 사실)
       - 고령_여부: 38세 이상 플래그 (성공률 급락 기점)
       - 과거_임신성공률: 과거 성공 경험이 미래 예측에 가장 강한 신호

    B. 이번 시술 정보 (4개)
       - 시술_FER: 동결배아이식 여부 (신선 배아보다 성공률 높은 경우 있음)
       - 시술_ICSI: 미세주입 여부 (남성 불임 케이스에 효과적)
       - 시술_BLASTOCYST: 배반포 이식 여부 (성공률 높음)
       - 배란 자극 여부: 배란 유도 치료 여부

    C. 배아 품질 (3개)
       - 이식된 배아 수: 많을수록 성공 확률 높아짐
       - 배아_이식비율: 생성된 배아 중 이식 비율 (배아 품질 간접 지표)
       - 미세주입_성공률: ICSI 성공률 (정자/난자 품질 반영)

    D. 불임 원인 (2개)
       - 총_불임원인_수: 원인이 많을수록 복잡한 케이스
       - 불명확_단독원인: 원인 불명인 경우 (예후 다름)

    E. 과거 이력 (3개)
       - 총 시술 횟수_num: 반복 시술일수록 어려운 케이스
       - failure_streak: 시술횟수 - 임신횟수 (연속 실패)
       - 출산_경험: 출산 성공 경험 있는지 여부

    F. 클리닉 (2개)
       - 시술시기코드_성공률: 클리닉별 성공률 (의료진 수준 반영)
       - 시술시기코드_시술건수: 클리닉 규모 (경험 많을수록 성공률 높음)

    G. 시술 유형 (1개)
       - IVF시술_여부: IVF vs DI 상위 분류

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
# 피처 목록 (의학적 근거 있는 것만)
# ====================================================
CLEAN_FEATURES = [
    # A. 환자 상태
    '나이_수치',           # 나이가 많을수록 성공률 하락
    '고령_여부',           # 38세 이상 플래그
    '과거_임신성공률',     # 과거 성공 경험

    # B. 이번 시술 정보
    '시술_FER',            # 동결배아이식 여부
    '시술_ICSI',           # 미세주입 여부
    '시술_BLASTOCYST',     # 배반포 이식 여부
    '배란 자극 여부',      # 배란 유도 치료 여부

    # C. 배아 품질
    '이식된 배아 수',      # 이식 배아 수
    '배아_이식비율',       # 생성 배아 중 이식 비율
    '미세주입_성공률',     # ICSI 성공률

    # D. 불임 원인
    '총_불임원인_수',      # 불임 원인 복잡도
    '불명확_단독원인',     # 원인 불명 여부

    # E. 과거 이력
    '총 시술 횟수_num',    # 반복 시술 횟수
    'failure_streak',      # 연속 실패 횟수
    '출산_경험',           # 출산 성공 경험

    # F. 클리닉
    '시술시기코드_성공률', # 클리닉별 성공률
    '시술시기코드_시술건수', # 클리닉 규모

    # G. 시술 유형
    'IVF시술_여부',        # IVF vs DI
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
# 2. 전처리 (핵심 피처만)
# ====================================================
def preprocess(train, test):
    print('\n' + '=' * 55)
    print('[2] 전처리 & 핵심 피처 생성')
    print('=' * 55)

    train = train.copy()
    test  = test.copy()

    # ── A. 환자 상태 ────────────────────────────────────────────
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
        # 나이
        df['나이_수치'] = df['시술 당시 나이'].map(age_map).fillna(-1)
        df['고령_여부'] = (df['나이_수치'] >= 38).astype(int)

        # 횟수 수치화
        for col in count_cols:
            df[col + '_num'] = df[col].apply(parse_count)

        # 과거 임신 성공률 (행 내 연산)
        df['과거_임신성공률'] = (
            df['총 임신 횟수_num'] / (df['총 시술 횟수_num'] + 1e-6)
        )

        # 출산 경험
        df['출산_경험'] = (df['총 출산 횟수_num'] > 0).astype(int)

        # ── B. 시술 정보 ──────────────────────────────────────────
        시술유형 = df['특정 시술 유형'].astype(str)
        df['시술_FER']        = 시술유형.str.contains('FER',        na=False).astype(int)
        df['시술_ICSI']       = 시술유형.str.contains('ICSI',       na=False).astype(int)
        df['시술_BLASTOCYST'] = 시술유형.str.contains('BLASTOCYST', na=False).astype(int)
        df['IVF시술_여부']    = (df['시술 유형'] == 'IVF').astype(int)

        # ── C. 배아 품질 ──────────────────────────────────────────
        df['배아_이식비율'] = (
            df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1e-6)
        )
        df['미세주입_성공률'] = (
            df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1e-6)
        )

        # ── D. 불임 원인 ──────────────────────────────────────────
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
        df['총_불임원인_수']  = df[all_cause_cols].sum(axis=1)
        df['불명확_단독원인'] = (
            (df['불명확 불임 원인'] == 1) & (df['총_불임원인_수'] == 1)
        ).astype(int)

        # ── E. 과거 이력 ──────────────────────────────────────────
        df['failure_streak'] = (
            df['총 시술 횟수_num'] - df['총 임신 횟수_num']
        ).clip(lower=0)

    # ── 결측치 처리 (train median만 사용) ───────────────────────
    num_base = [c for c in CLEAN_FEATURES if c in train.select_dtypes(include=np.number).columns]
    medians  = train[num_base].median()
    train[num_base] = train[num_base].fillna(medians)
    test[num_base]  = test[num_base].fillna(medians)

    # ── F. 클리닉 집계 피처 (K-Fold OOF) ────────────────────────
    print('  클리닉 집계 피처 생성 중...')
    clinic_col  = '시술 시기 코드'
    global_mean = train[TARGET].mean()
    skf_clinic  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    smooth      = 50

    # 클리닉별 성공률
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

    # 클리닉 규모
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

    # ── 최종 피처 확정 ───────────────────────────────────────────
    # CLEAN_FEATURES 중 실제로 존재하는 것만
    feature_cols = [c for c in CLEAN_FEATURES if c in train.columns]
    missing      = [c for c in CLEAN_FEATURES if c not in train.columns]
    if missing:
        print(f'  ⚠️ 없는 피처: {missing}')

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]

    # 최종 결측치 처리
    medians2    = X_train.median()
    X_train     = X_train.fillna(medians2)
    X_test      = X_test.fillna(medians2)

    print(f'\n  {"피처":<30} {"설명"}')
    print('  ' + '-' * 55)
    feature_desc = {
        '나이_수치'           : 'A. 환자 나이 (수치)',
        '고령_여부'           : 'A. 38세 이상 플래그',
        '과거_임신성공률'     : 'A. 과거 임신 성공률',
        '시술_FER'            : 'B. 동결배아이식 여부',
        '시술_ICSI'           : 'B. 미세주입 여부',
        '시술_BLASTOCYST'     : 'B. 배반포 이식 여부',
        '배란 자극 여부'      : 'B. 배란 유도 치료',
        '이식된 배아 수'      : 'C. 이식 배아 수',
        '배아_이식비율'       : 'C. 이식/생성 배아 비율',
        '미세주입_성공률'     : 'C. ICSI 성공률',
        '총_불임원인_수'      : 'D. 불임 원인 복잡도',
        '불명확_단독원인'     : 'D. 원인 불명 여부',
        '총 시술 횟수_num'    : 'E. 총 시술 횟수',
        'failure_streak'      : 'E. 연속 실패 횟수',
        '출산_경험'           : 'E. 출산 성공 경험',
        '시술시기코드_성공률' : 'F. 클리닉 성공률',
        '시술시기코드_시술건수': 'F. 클리닉 규모',
        'IVF시술_여부'        : 'G. IVF vs DI',
    }
    for col in feature_cols:
        desc = feature_desc.get(col, '')
        print(f'  {col:<30} {desc}')

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
        oof[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred         += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(
            f'  Fold {fold+1}  AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}'
            f'  (best_iter: {m.best_iteration_})'
        )

    cv = roc_auc_score(y_train, oof)
    print(f'  LightGBM CV AUC : {cv:.5f}')

    # 피처 중요도 출력 (설명 가능성)
    importance = pd.DataFrame({
        'feature'   : X_train.columns,
        'importance': m.feature_importances_
    }).sort_values('importance', ascending=False)

    print('\n  [피처 중요도 Top 18]')
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
def ensemble_and_save(y_train, sub, lgb_result, xgb_result, cat_result):
    print('\n' + '=' * 55)
    print('[4] 앙상블 & 제출 파일 저장')
    print('=' * 55)

    lgb_oof,  lgb_pred,  lgb_cv,  _ = lgb_result
    xgb_oof,  xgb_pred,  xgb_cv    = xgb_result
    cat_oof,  cat_pred,  cat_cv    = cat_result

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
    filename  = f'{SAVE_PATH}submission_{timestamp}_clean_auc{auc_str}.csv'

    sub['probability'] = ensemble_pred
    sub.to_csv(filename, index=False)
    print(f'\n  저장 완료 : {filename}')
    print('\n  ✅ 다음 단계: 피처 중요도 보고 하위 피처 제거 or 새 피처 하나씩 추가')


# ====================================================
# 메인 실행
# ====================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  난임 환자 임신 성공 여부 예측 (v_clean)')
    print('  피처 수: 18개 (의학적 근거 있는 것만)')
    print('=' * 55)

    train, test, sub                       = load_data()
    X_train, y_train, X_test, feature_cols = preprocess(train, test)

    lgb_result = train_lgb(X_train, y_train, X_test)
    xgb_result = train_xgb(X_train, y_train, X_test)
    cat_result = train_cat(X_train, y_train, X_test)

    ensemble_and_save(y_train, sub, lgb_result, xgb_result, cat_result)