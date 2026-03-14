import os
import shutil
import pandas as pd
from src.utils import get_data_path
from src.preprocess import build_strict_pipeline
from autogluon.tabular import TabularPredictor

train = pd.read_csv(get_data_path("train.csv"))
test = pd.read_csv(get_data_path("test.csv"))

print(train['임신 성공 여부'].value_counts(normalize=True))

X_train, y_train, X_test = build_strict_pipeline(train, test)
print(f"✅ 전처리 완료! 피처 개수: {X_train.shape[1]}개")

train_data = pd.concat([X_train, y_train], axis=1)

# 이전 폴더 초기화
if os.path.exists('Autogluon_Models_v7'):
    shutil.rmtree('Autogluon_Models_v7')

predictor = TabularPredictor(
    label='임신 성공 여부',
    eval_metric='roc_auc',
    path='Autogluon_Models_v7'
).fit(
    train_data,
    presets='best_quality',       # 프리셋이 최적화 담당, 건드리지 않음
    time_limit=7200,              # 2시간으로 연장
    ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
    ag_args_fit={'num_gpus': 1},  # GPU 사용 (LightGBM 경고는 무시해도 됨)
    excluded_model_types=['KNN', 'FASTAI'],  # RF, XT는 유지
    save_space=True
)

lb = predictor.leaderboard(silent=False)
print(lb)

proba_df = predictor.predict_proba(X_test)
target_label = 1 if 1 in proba_df.columns else True
preds_proba = proba_df[target_label]

submission = pd.read_csv(get_data_path("sample_submission.csv"))
submission[submission.columns[1]] = preds_proba

os.makedirs('./submissions', exist_ok=True)
submission.to_csv('./submissions/autogluon_gpu_submission_v7.csv', index=False)
print(f"✅ 제출 파일 저장 완료")
print(f"probability 범위: {preds_proba.min():.4f} ~ {preds_proba.max():.4f}")
print(f"결측: {preds_proba.isnull().sum()}")

try:
    importance = predictor.feature_importance(
        data=train_data,
        subsample_size=5000,
        num_shuffle_sets=3
    )
    print(importance)
    importance.to_csv('./submissions/feature_importance_v7.csv')
except Exception as e:
    print(f"❌ 중요도 분석 오류: {e}")