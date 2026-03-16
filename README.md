# 난임 환자 임신 성공 여부 예측 AI

> 🏥 난임 환자의 시술 데이터를 분석하여 임신 성공 여부를 예측하는 AI 모델

---

## 🎯 프로젝트 목표

난임 시술(IVF, DI 등)을 받는 환자의 다양한 의료 정보를 바탕으로, 임신 성공 가능성을 사전에 예측합니다. 정확한 예측을 통해 환자 맞춤형 시술 계획 수립에 기여하는 것을 목표로 합니다.

- **평가 지표**: ROC-AUC
- **타겟**: 임신 성공 여부 (0: 실패, 1: 성공)
- **클래스 비율**: 성공 25.83% / 실패 74.17% (불균형)

---

## 📊 데이터 설명

> ⚠️ 데이터는 대회 규정상 저장소에 포함되지 않습니다. `data/` 폴더에 직접 배치해주세요.

| 파일 | 행 수 | 컬럼 수 | 설명 |
|------|-------|--------|------|
| `train.csv` | 256,351 | 69 | 학습 데이터 (타겟 포함) |
| `test.csv` | 90,067 | 68 | 추론 데이터 |
| `sample_submission.csv` | 90,067 | 2 | 제출 양식 |

**주요 피처 그룹**

| 그룹 | 피처 예시 |
|------|----------|
| 환자 정보 | 시술 당시 나이, 임신 시도 경과 연수 |
| 시술 정보 | 시술 유형(IVF/DI), 특정 시술 유형, 배란 자극 여부 |
| 불임 원인 | 남성/여성/부부 불임 원인 플래그 18개 |
| 배아 정보 | 총 생성·이식·저장 배아 수, 미세주입 관련 수치 |
| 과거 이력 | 총 시술 횟수, IVF/DI 임신·출산 횟수 |
| 경과일 | 난자 채취·혼합·이식·해동 경과일 |

**결측치 주의 컬럼**

| 컬럼 | 결측률 |
|------|--------|
| 임신 시도 또는 마지막 임신 경과 연수 | 96.3% |
| 난자 해동 경과일 | 99.4% |
| PGD/PGS 시술 여부 | ~99% |

---

## 🧪 모델 실험 결과

### Data Leakage 방지 원칙

모든 실험에서 아래 원칙을 준수합니다.

- LabelEncoder / Target Encoding은 **train 데이터로만 fit**, test는 transform만 수행
- 결측치 보간 통계값은 **train 기준**으로만 계산 후 test에 적용
- 파생 변수는 **각 행(row) 내 연산만** 수행
- Target Encoding은 **K-Fold OOF 방식**으로 train 내부 leakage 차단

### 버전별 실험 요약

| 버전 | 주요 변경사항 | 피처 수 | LightGBM | XGBoost | CatBoost | 앙상블 CV AUC |
|------|-------------|--------|----------|---------|---------|--------------|
| v1 | 베이스라인 (기본 파생 피처, lr=0.05) | 87 | 0.73868 | 0.73967 | 0.73999 | 0.74018 |
| v2 | 피처 강화 + lr=0.02 + Optuna 튜닝 | 106 | 0.74037 | 0.73989 | 0.74000 | 0.74047 |
| v3 | Target Encoding + 교호작용 + 도메인 피처 | 121 | 0.74047 | 0.73960 | 0.74041 | **0.74060** |

### 버전별 핵심 변경 내용

**v1 → v2**
- `learning_rate` 0.05 → 0.02, `n_estimators` 3,000 → 5,000
- 시간 간격 피처 추가 (채취→이식, 혼합→이식 경과일 차이, 배반포 이식 추정)
- 불임 원인 조합 피처 추가 (총 불임 원인 수, 복합 불임 여부)
- Optuna 50 trials 하이퍼파라미터 탐색

**v2 → v3**
- K-Fold OOF Target Encoding 추가 (시술 시기 코드, 시술 유형 등 6개 컬럼)
- 교호작용 피처 추가 (시술유형 × 나이, 시술유형 × 불임 주원인)
- 도메인 지식 기반 치료 적합성 피처 추가 (남성요인-ICSI 매칭, 배란장애-자극 매칭 등)

---

## 🏆 최종 모델 선택 이유

**현재 최고 성능: v3 앙상블 (CV AUC 0.74060)**

v3를 현재 최고 버전으로 선택한 이유는 다음과 같습니다.

1. **일관된 성능 향상**: v1 → v2 → v3로 CV AUC가 단조 증가
2. **Leakage 안전성**: 모든 피처 엔지니어링이 Leakage 규정을 준수
3. **앙상블 안정성**: 세 모델의 OOF 예측을 CV AUC 기반 가중 평균으로 결합해 단일 모델 대비 분산 감소
4. **도메인 지식 반영**: 단순 통계 피처 외에 의학적 치료 적합성을 피처로 반영해 모델의 해석 가능성 향상

> LB(리더보드) 점수 확인 후 CV-LB 갭 분석 예정. 갭이 클 경우 오버피팅 의심 및 피처 재검토 진행.

---

## ▶️ 실행 방법

### 1. 환경 설정

```bash
git clone https://github.com/Beemo-4018/infertility-prediction-ai.git
cd infertility-prediction-ai

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 데이터 배치

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. 학습 실행

```bash
# 최신 버전 (v3)
python src/train_v3.py

# 이전 버전
python src/train_v2.py
python src/train.py      # v1 베이스라인
```

### 4. 주요 설정값 (`train_v3.py` 상단)

```python
USE_OPTUNA    = True   # False로 변경 시 Optuna 생략, 빠른 실행 가능
OPTUNA_TRIALS = 50     # Optuna 탐색 횟수 (시간 여유 있으면 100 권장)
DATA_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/'
SAVE_PATH     = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'
```

### 5. 출력 파일

제출 파일은 `data/submissions/` 에 자동 저장됩니다.

```
submission_MMDD_HHMM_auc0pXXXXX.csv
```

---

## 📁 프로젝트 구조

```
infertility-prediction-ai/
│
├── data/                        ← 데이터 폴더 (.gitignore 처리)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── submissions/             ← 제출 파일 자동 저장
│
├── src/                         ← 학습 스크립트
│   ├── train.py                 ← v1 베이스라인
│   ├── train_v2.py              ← v2 피처 강화 + Optuna
│   └── train_v3.py              ← v3 Target Encoding + 도메인 피처 (최신)
│
├── notebooks/                   ← EDA 및 실험용 노트북
│
├── requirements.txt
└── README.md
```

---

## 📦 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| lightgbm | 그래디언트 부스팅 모델 |
| xgboost | 그래디언트 부스팅 모델 |
| catboost | 그래디언트 부스팅 모델 |
| scikit-learn | 교차 검증, 전처리 |
| optuna | 하이퍼파라미터 자동 탐색 |
| pandas / numpy | 데이터 처리 |

---

## 👥 팀원

| 이름 | GitHub |
|------|--------|
| 이정결 | - |
| 안병준 | - |
| 이승연 | - |
