# 난임 임신 성공 여부 예측 프로젝트

난임 치료 관련 정형 데이터를 기반으로 임신 성공 여부를 예측한 머신러닝 프로젝트입니다.  
대회형 환경에서 ROC-AUC를 기준으로 모델을 개선했고, 피처 엔지니어링, 검증 체계, 앙상블, 리더보드 해석까지 전 과정을 직접 설계하며 발전시켰습니다.

이 저장소에는 다음이 포함되어 있습니다.

- 최종 학습 파이프라인
- 버전별 피처 엔지니어링 실험 코드
- 빠른 스크리닝용 검증 스크립트
- 세미풀 검증 및 피처 다이어트 도구
- 제출 파일과 중요도/실험 로그 산출물

## 프로젝트 개요

- 문제 유형: 이진 분류
- 목표: 난임 치료 후 임신 성공 여부 예측
- 평가 지표: ROC-AUC
- 데이터 규모
  - Train: `256,351 x 69`
  - Test: `90,067 x 68`
  - 양성 비율: `25.83%`
- 실행 환경: Windows / Python / Conda
- 원본 데이터 위치: [data/raw](c:\fertility-prediction-ai\data\raw)

## 핵심 결과

- 프로젝트 진행 중 확인한 최고 리더보드 점수: `0.7421840512`
- 최종 후보군의 로컬 CV 범위: 약 `0.7408x`
- 핵심 인사이트: 로컬 CV가 높다고 반드시 리더보드 점수가 높아지지는 않았음

대표 실험 파일:

| 파일 | 역할 | 설명 |
|---|---|---|
| `train_v8plus_v8_final.py` | 안정적인 최종 베이스라인 | 메인 엔트리에서 사용하는 기본 파이프라인 |
| `train_v8plus_v8_final_seedavg.py` | 시드 평균 앙상블 실험 | CV는 올랐지만 LB 개선은 일관되지 않았음 |
| `train_v12_final.py` | 후반부 고비용 실험 | 더 강한 피처 엔지니어링과 넓은 앙상블 구성 |
| `train_v12_lite_check.py` | 빠른 수동 피처 다이어트 검증 | 한 번 실행으로 CV와 중요도를 함께 확인 |
| `train_v12_semifull_eval.py` | 세미풀 검증 | 스크린보다 현실적이고 final보다 가벼운 검증 |

## 폴더 / 파일 구조

주요 학습 및 실험 파일:

- [main.py](c:\fertility-prediction-ai\main.py): 현재 안정 버전 실행용 엔트리
- [train_v8plus_v8_final.py](c:\fertility-prediction-ai\train_v8plus_v8_final.py): v8 최종 파이프라인
- [train_v8plus_v8_final_seedavg.py](c:\fertility-prediction-ai\train_v8plus_v8_final_seedavg.py): 시드 평균 실험
- [train_v8plus_v8_lgb_screen.py](c:\fertility-prediction-ai\train_v8plus_v8_lgb_screen.py): 빠른 LightGBM 스크리닝
- [train_v12_final.py](c:\fertility-prediction-ai\train_v12_final.py): 후반부 최종 실험 파이프라인
- [train_v12_lite_check.py](c:\fertility-prediction-ai\train_v12_lite_check.py): 중요도 기반 수동 피처 다이어트용 체크 스크립트
- [train_v12_semifull_eval.py](c:\fertility-prediction-ai\train_v12_semifull_eval.py): 세미풀 비교/검증 스크립트

전처리 및 보조 모듈:

- [src/preprocess.py](c:\fertility-prediction-ai\src\preprocess.py)
- [src/preprocess_v25.py](c:\fertility-prediction-ai\src\preprocess_v25.py)
- [src/preprocess_v26a.py](c:\fertility-prediction-ai\src\preprocess_v26a.py)
- [src/preprocess_v27.py](c:\fertility-prediction-ai\src\preprocess_v27.py)
- [src/utils.py](c:\fertility-prediction-ai\src\utils.py)

데이터 및 결과물:

- [data/raw](c:\fertility-prediction-ai\data\raw): 원본 대회 데이터
- [data/submissions](c:\fertility-prediction-ai\data\submissions): 제출 파일, 중요도, 실험 산출물
- [submissions](c:\fertility-prediction-ai\submissions): 초기 제출/분석 산출물

## 접근 방식

프로젝트는 아래 사이클을 반복하며 발전했습니다.

1. 도메인 해석 기반 피처 엔지니어링
2. 범주형/조합 피처에 대한 OOF 타깃 인코딩
3. LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest 비교
4. 빠른 스크린 실험으로 후보 압축
5. 세미풀 검증과 최종 파이프라인 검증 후 제출

핵심 피처 유형:

- 클리닉 단위 집계 성공률 피처
- 나이, 시술 유형, 불임 원인 조합 피처
- 배아/난자/정자 관련 비율 및 효율 피처
- 시술 이력 및 시기 기반 파생 피처

## 잘 먹힌 전략

- 클리닉 단위 집계 피처가 지속적으로 강한 신호를 보였습니다.
- 고카디널리티 범주형 변수에 OOF 타깃 인코딩이 효과적이었습니다.
- 한 번에 큰 변경보다, 해석 가능한 작은 변경이 더 안정적으로 성능을 올렸습니다.
- 가벼운 스크리닝 스크립트로 실험 후보를 미리 압축한 것이 시간 절약에 도움이 됐습니다.
- 개별 모델 차이가 작아도 OOF 기반 가중 앙상블은 안정적으로 도움이 됐습니다.

## 잘 안 먹힌 전략

- 더 높은 CV가 항상 더 높은 리더보드 점수로 이어지지는 않았습니다.
- 시드 평균 앙상블은 CV를 올렸지만 실제 LB는 오히려 떨어지는 경우가 있었습니다.
- 공격적인 피처 다이어트는 로컬 검증에선 안전해 보여도 최종 LB에서 손해를 줄 수 있었습니다.
- GPU가 당연히 더 좋을 것 같았지만, 이 프로젝트에서는 CPU가 더 빠르고 더 강한 경우가 많았습니다.
- 여러 피처를 한꺼번에 바꾸는 방식은 해석과 검증이 어려워 신뢰도가 낮았습니다.

## 트러블슈팅 요약

### 1. CPU와 GPU 결과가 다르게 나왔던 문제

처음에는 GPU가 더 빠르고 성능도 좋을 것으로 기대했지만, 실제로는 이 데이터와 설정에서 CPU가 더 빠르고 더 좋은 경우가 반복적으로 나왔습니다.  
그래서 하드웨어에 대한 가정을 믿기보다, 같은 조건에서 CPU/GPU를 직접 비교하고 이후 실험은 CPU 기준으로 고정했습니다.

### 2. CV는 올랐는데 리더보드 점수는 떨어진 문제

후반부 실험에서 여러 번 겪은 문제였습니다.  
특히 seed averaging 같은 기법은 로컬 CV는 분명히 좋아졌지만 실제 LB는 떨어지는 결과가 나왔습니다.

이후에는:

- 로컬 CV는 후보 선별용으로 사용
- 반복적으로 확인된 LB 결과를 더 신뢰
- 아주 작은 CV 개선에는 과도한 의미를 부여하지 않음

방식으로 실험 프로세스를 바꿨습니다.

### 3. 피처 다이어트가 로컬에서는 괜찮아 보였지만 최종 성능을 해친 문제

LightGBM 기준 importance가 낮거나 0인 피처를 제거하는 방식은 빠르게 후보를 줄이는 데는 유용했습니다.  
하지만 실제 final 앙상블과 LB에서는 일부 저중요도 피처가 미세하게 기여하고 있었습니다.

그래서 이후에는:

- screen 결과는 후보 생성용
- semi-full 결과는 중간 검증용
- final 결과만 제출 판단 기준

으로 역할을 분리했습니다.

### 4. 모델을 많이 넣는다고 앙상블이 좋아지지 않았던 문제

모델 수를 늘리면 다양성이 생길 것이라 기대했지만, 실제 최적 가중치를 보면 기여도가 거의 0에 가까운 모델도 있었습니다.  
결국 중요한 것은 모델 개수보다, 서로 다른 신호를 진짜로 제공하는지 여부였습니다.

## 빠른 실행 방법

의존성 설치:

```powershell
pip install -r requirements.txt
```

현재 안정 버전 실행:

```powershell
python main.py
```

후반부 최종 실험 실행:

```powershell
python train_v12_final.py
```

빠른 중요도 + CV 체크:

```powershell
python train_v12_lite_check.py
```

드롭 리스트를 반영한 재실행:

```powershell
python train_v12_lite_check.py --drop-file data/submissions/v12_lite_check/drop_cols_zero_only.csv
```

## 산출물

주요 산출물:

- 제출용 CSV 파일
- 피처 중요도 CSV 파일
- zero importance / low importance 후보 목록
- 세미풀 비교 결과 CSV
- 피처 다이어트 로그

## 포트폴리오 문서

포트폴리오용 서술형 문서는 아래 파일에 정리했습니다.

- [PORTFOLIO_CASE_STUDY.md](c:\fertility-prediction-ai\PORTFOLIO_CASE_STUDY.md)

이 문서에는 다음 내용을 담았습니다.

- 프로젝트 배경
- 내가 맡은 역할
- 모델링 전략
- 성과
- 시행착오와 트러블슈팅
- 이 프로젝트를 통해 보여줄 수 있는 역량
