import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold

def feature_engineering(df):
    df = df.copy()
    
    # 1. 횟수 컬럼 처리
    cols_to_fix = ['총 출산 횟수', '총 임신 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 시술 횟수', 
                   '클리닉 내 총 시술 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']
    for col in cols_to_fix:
        if col in df.columns:
            df[f'{col}_is_max'] = df[col].astype(str).str.contains('이상').astype(int)
            df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)

    # 2. 나이 수치화
    age_map = {'만18-34세': 26, '만35-37세': 36, '만38-39세': 38.5, '만40-42세': 41, '만43-44세': 43.5, '만45-50세': 47.5}
    if '시술 당시 나이' in df.columns:
        df['age_numeric'] = df['시술 당시 나이'].map(age_map).fillna(36)

    # 3. 파생 변수
    df['Egg_to_Embryo_Yield'] = (df['총 생성 배아 수'].fillna(0) / (df['수집된 신선 난자 수'].fillna(0) + df['해동 난자 수'].fillna(0) + 1e-5)).clip(0, 1)
    df['Implant_Efficiency'] = (df['이식된 배아 수'].fillna(0) / (df['총 생성 배아 수'].fillna(0) + 1e-5)).clip(0, 1)
    df['ICSI_Ratio'] = (df['미세주입된 난자 수'].fillna(0) / (df['혼합된 난자 수'].fillna(0) + 1e-5)).clip(0, 1)
    df['Frozen_Ratio'] = (df['해동된 배아 수'].fillna(0) / (df['이식된 배아 수'].fillna(0) + 1e-5)).clip(0, 1)

    # 4. 상호작용 (V6 검증된 3개 유지)
    df['Age_x_Implant_Efficiency'] = df['age_numeric'] * df['Implant_Efficiency'].fillna(0)
    df['Age_x_Embryo_Count'] = df['age_numeric'] * df['이식된 배아 수'].fillna(0)
    df['Age_x_Transfer_Day'] = df['age_numeric'] * df['배아 이식 경과일'].fillna(0)
    # 🚨 팀원 피드백 반영: p-value 높은 'Age_x_Stored_Embryo'는 생성 안 함

    # 5. 결측 플래그
    high_null_cols = ['PGS 시술 여부', 'PGD 시술 여부', '난자 해동 경과일', '착상 전 유전 검사 사용 여부', '임신 시도 또는 마지막 임신 경과 연수']
    for col in high_null_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    # 6. 노이즈 삭제
    cols_to_drop = [
        'ID', '시술 시기 코드', '시술 당시 나이', '불임 원인 - 여성 요인', '난자 채취 경과일', '난자 해동 경과일', '난자 혼합 경과일', 
        '저장된 신선 난자 수', '불임 원인 - 자궁경부 문제', '미세주입 후 저장된 배아 수', '배아 해동 경과일', '총 생성 배아 수', 
        '미세주입에서 생성된 배아 수', 'DI 출산 횟수', 'PGD 시술 여부_is_missing', '불임 원인 - 정자 면역학적 요인', 
        '총 출산 횟수_is_max', '총 임신 횟수_is_max', 'DI 출산 횟수_is_max', 'IVF 임신 횟수_is_max', 'DI 임신 횟수_is_max', 
        'IVF 출산 횟수_is_max', 'PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부', '불임 원인 - 정자 운동성', 
        '기증 배아 사용 여부', '대리모 여부', '총 시술 횟수_is_max', '클리닉 내 총 시술 횟수_is_max', '착상 전 유전 진단 사용 여부', 
        '부부 부 불임 원인', '여성 부 불임 원인', '임신 시도 또는 마지막 임신 경과 연수_is_missing', '착상 전 유전 검사 사용 여부_is_missing', 
        'DI 임신 횟수', '해동 난자 수', '불임 원인 - 정자 형태'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df

def build_strict_pipeline(train_df, test_df, target_col='임신 성공 여부'):
    train_eng = feature_engineering(train_df)
    test_eng = feature_engineering(test_df)
    X_train, y_train, X_test = train_eng.drop(columns=[target_col]), train_eng[target_col], test_eng.copy()

    # ✅ Target Encoding
    te_cols = [c for c in ['시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '난자 출처', '정자 출처'] if c in X_train.columns]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y_train.mean()
    for col in te_cols:
        oof, test_preds = np.full(len(X_train), global_mean), np.zeros(len(X_test))
        for tr_idx, val_idx in kf.split(X_train):
            mean_map = y_train.iloc[tr_idx].groupby(X_train[col].iloc[tr_idx]).mean()
            oof[val_idx] = X_train[col].iloc[val_idx].map(mean_map).fillna(global_mean)
            test_preds += X_test[col].map(mean_map).fillna(global_mean) / 5
        X_train[f'{col}_te'] = oof
        X_test[f'{col}_te'] = test_preds

    # 결측치/인코딩 처리
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    
    return X_train[X_train.columns], y_train, X_test[X_train.columns]