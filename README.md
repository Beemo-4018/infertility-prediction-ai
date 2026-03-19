# 난임 환자 임신 성공 여부 예측 AI 해커톤

> **DACON** 난임 환자 대상 임신 성공 여부 예측 AI 해커톤  
> **Team Beemo** | 평가지표: ROC-AUC | **최고 LB: 0.74216**

---

## 📊 최종 성과

| 지표 | 값 |
|------|-----|
| 최고 LB AUC | **0.74216** |
| 최고 CV AUC | **0.74083** |
| 1위와의 차이 | 0.00019 |

---

## 🗂️ 프로젝트 구조

```
infertility-prediction-ai/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── submissions/
├── src/
│   ├── train_v5.py                  # LB 0.74198
│   ├── train_v8.py                  # LB 0.74195 (LGB Optuna)
│   ├── train_v8_tuned.py            # LB 0.74200 (전체 Optuna)
│   ├── train_v8plus_v7b_final.py    # LB 0.74216 ← 팀 최고
│   ├── train_vclean7.py             # CV 0.73184 (Clean Baseline)
│   ├── train_vclean12.py            # CV 0.73999
│   ├── feature_search.py            # 피처 기여도 자동 측정 v1
│   ├── feature_search_v2.py         # 피처 기여도 자동 측정 v2
│   ├── feature_search_v3.py         # 피처 기여도 자동 측정 v3
│   ├── feature_search_clinic.py     # 클리닉 조합 TE 측정
│   ├── fold_analysis.py             # Fold 분포 분석
│   ├── blend.py                     # 블렌딩 스크립트
│   ├── rank_blend.py                # Rank Averaging 블렌딩
│   ├── check_corr.py                # 제출 파일 상관관계 확인
│   └── train_autogluon.py           # AutoGluon (코랩용)
└── README.md
```

---

## 🔧 환경 설정

```bash
conda create -n bmo python=3.10
conda activate bmo
pip install lightgbm xgboost catboost optuna scikit-learn pandas numpy scipy
```

---

## 🚀 실행 방법

```bash
# 팀 최고 성능 버전
python src/train_v8plus_v7b_final.py

# Clean Baseline 버전
python src/train_vclean12.py

# 피처 기여도 자동 측정
python src/feature_search.py

# 제출 파일 상관관계 확인
python src/check_corr.py
```

---

## 🏗️ 모델 아키텍처

```
Raw Data (256,351 × 69)
        ↓
   Preprocessing
   ├── 결측치 처리 (train median)
   ├── 횟수 컬럼 수치화 ('회' 제거)
   └── 나이 수치화 (age_map)
        ↓
 Feature Engineering (130개)
   ├── 배아 비율 피처 (9개)
   ├── 시간 간격 피처 (5개) ← 가장 큰 기여
   ├── 교호작용 피처 (9개)
   ├── 클리닉 집계 피처 (7개, K-Fold OOF)
   └── Target Encoding (12개 컬럼, K-Fold OOF)
        ↓
  LightGBM Optuna (50 trials, 3-Fold)
        ↓
  3-Model Ensemble
   ├── LightGBM (Optuna 튜닝)
   ├── XGBoost
   └── CatBoost
        ↓
  11가지 앙상블 조합 자동 비교
  (단순평균 / 성능가중 / Rank Normalization 등)
        ↓
    최고 CV 조합 제출
```

---

## 🔬 핵심 피처

### 피처 중요도 Top 10

| 순위 | 피처 | 설명 |
|------|------|------|
| 1 | `배아_이식비율` | 이식배아 / 총생성배아 |
| 2 | `미세주입_성공률` | ICSI 성공률 |
| 3 | `시술시기코드_시술건수` | 클리닉 규모 (log1p) |
| 4 | `시술시기코드_성공률` | 클리닉별 성공률 |
| 5 | `나이_수치` | 시술 당시 나이 수치화 |
| 6 | `과거_임신성공률` | 총임신 / 총시술 |
| 7 | `이식된 배아 수` | 이번 시술 이식 배아 수 |
| 8 | `총_불임원인_수` | 불임 원인 복잡도 |
| 9 | `혼합_이식_간격` | 배아 배양 기간 |
| 10 | `failure_streak` | 연속 실패 횟수 |

### v7b 핵심 피처

```python
# 고령(38세+) 환자가 단일 배아 이식 선택 = 의사가 좋은 배아라고 판단한 신호
df['고령_x_단일배아이식'] = df['고령_여부'] * df['단일 배아 이식 여부']
```

### 시간 간격 피처 (+0.00675 점프)

```python
df['혼합_이식_간격'] = df['배아 이식 경과일'] - df['난자 혼합 경과일']
df['해동_이식_간격'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
df['배반포_이식추정'] = (df['혼합_이식_간격'] >= 5).astype(int)
```

---

## 📈 버전별 성능

| 버전 | CV AUC | LB AUC | 핵심 변경 |
|------|--------|--------|---------|
| v3 | 0.74060 | 0.74198 | Target Encoding + 도메인 피처 |
| v5 | 0.74062 | 0.74198 | 클리닉 집계 피처 |
| v8 | 0.74067 | 0.74195 | 미사용 컬럼 발굴 + LGB Optuna |
| v8_tuned | 0.74073 | 0.74200 | XGB/CatBoost Optuna 추가 |
| blend v5+v8 | — | 0.74205 | 50:50 블렌딩 |
| **v8plus_v7b** | **0.74083** | **0.74216** | **고령×단일배아이식 + Rank Norm** |
| vclean7 | 0.73184 | — | Clean Baseline + 시간간격 피처 |
| vclean12 | 0.73999 | — | feature_search 상위 피처 추가 |

---

## 🔀 앙상블 전략

### Rank Normalization

```python
# 확률값 대신 순위값으로 정규화 → 각 모델의 score 분포 차이 제거
r_lgb = rankdata(oof_lgb) / len(oof_lgb)
r_xgb = rankdata(oof_xgb) / len(oof_xgb)
r_cat = rankdata(oof_cat) / len(oof_cat)
```

### 블렌딩 상관관계 분석

```python
# 제출 전 상관관계 확인 필수
# 0.998 이상이면 블렌딩 효과 거의 없음
python src/check_corr.py
```

---

## 🛡️ Data Leakage 방지 원칙

| 처리 | 방법 |
|------|------|
| Target Encoding | K-Fold OOF (val은 tr 통계만 사용) |
| 결측치 보간 | train median → test 적용 |
| LabelEncoder | train만 fit, Unknown 처리 |
| **Pseudo-Labeling** | **절대 금지 (규정 위반 → 실격)** |

---

## 🧪 실패 교훈

| 실험 | 결과 | 교훈 |
|------|------|------|
| 풀 스태킹 | CV 0.73481 폭락 | 메타 피처는 OOF만 |
| TabNet + GPU | 불안정 | 25만행에서 트리 모델 우위 |
| 교호작용으로 원본 대체 | CV 하락 | 교호작용은 원본에 추가 |
| Pseudo-Labeling | 규정 위반 | 절대 사용 금지 |
| 피처 183개로 확장 | CV 하락 | 피처 수보다 피처 품질 |
| L2 스태킹 | L1보다 낮음 | 모델 다양성 부족 시 무의미 |

---

## 👥 Team 3
