
# 난임 환자 임신 성공 여부 예측 AI

> 해커톤 | 평가 지표: ROC-AUC

---

## 📁 폴더 구조

```
infertility-prediction-ai/
├── data/                      ← Git 제외 (별도 공유)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb           ← 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb ← 전처리 & 피처 엔지니어링
│   └── 03_modeling.ipynb      ← 모델링 & 앙상블
├── src/
│   ├── preprocess.py          ← 전처리 함수 모음
│   └── model.py               ← 모델 학습/예측 함수 모음
├── submissions/               ← 제출 파일 버전 관리
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ 환경 세팅

```bash
# 1. 레포 클론
git clone https://github.com/Beemo-4018/infertility-prediction-ai.git
cd infertility-prediction-ai

# 2. 가상환경 생성 & 활성화
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. data/ 폴더 생성 후 공유받은 데이터 넣기
mkdir data
# train.csv, test.csv, sample_submission.csv 를 data/ 에 복사
```

---

## 🌿 브랜치 전략

```
main        ← 최종 제출용 (검증된 코드만 merge)
dev         ← 통합 브랜치
├── feat/eda
├── feat/preprocessing
├── feat/modeling
└── feat/ensemble
```

```bash
# 작업 시작 전 항상
git pull origin dev

# 작업 후
git add .
git commit -m "feat: 피처 엔지니어링 추가"
git push origin feat/본인브랜치
```

---

## 📊 데이터 개요

| 항목 | 내용 |
|------|------|
| 피처 수 | 68개 |
| 평가 지표 | ROC-AUC |
| 시술 유형 | IVF, DI |
| 타겟 | 임신 성공 여부 (이진 분류) |

---

## 🏆 모델 전략

1. **전처리**: 결측치 처리, 나이 수치화, 카테고리 인코딩
2. **피처 엔지니어링**: 임신 성공률, 배아 비율 등 파생 피처
3. **모델**: LightGBM + XGBoost + CatBoost 앙상블
4. **검증**: Stratified K-Fold (5-Fold)
5. **튜닝**: Optuna
