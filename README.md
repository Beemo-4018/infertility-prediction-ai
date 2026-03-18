# Fertility Prediction AI

난임 환자 임신 성공 여부 예측 대회용 실험 저장소입니다. 현재는 `v8plus` 계열 피처 엔지니어링과 `LightGBM + XGBoost + CatBoost` 앙상블을 중심으로 실험하고 있습니다.

## 현재 상태

- 현재 기준 베스트 로컬 CV: `0.74078`
- 현재 베스트 실험: `v8plus_v5b`
- 현재 메인 베이스라인으로 보는 코드: [`train_v8plus_v5.py`](./train_v8plus_v5.py)
- 다음 검증 후보: [`train_v8plus_v6a.py`](./train_v8plus_v6a.py)

`v5b`에서 유지하기로 판단한 핵심 파생변수 3개:

- `난자정자출처조합_te`
- `총배아대비_해동비율`
- `IVF_실패부담`

빠른 스크리닝 기준으로는 추가 후보 3개 중 `IVF_출산전환율`이 가장 유망하게 나왔고, 이를 단일 추가한 정식 검증 버전이 `v6a`입니다.

## 버전 히스토리

최근 주요 버전 성능 기록:

- `v8plus_cpu`: `0.74071`
- `v8plus_gpu`: `0.74066`
- `v8plus_v2`: `0.74073`
- `v8plus_v3`: `0.74076`
- `v8plus_v4`: `0.74065`
- `v8plus_v4a`: `0.74063`
- `v8plus_v5`(6개 동시 추가): `0.74069`
- `v8plus_v5a`: `0.74073`
- `v8plus_v5b`: `0.74078`
- `v8plus_v5c`(조합 TE 제거 검증): `0.74073`

해석 요약:

- `v3` 이후 추가 피처 다이어트는 성능 개선으로 이어지지 않았습니다.
- `v5a`의 3개 추가보다 `v5b`의 3개 추가가 더 효과적이었습니다.
- 현재는 `v5b` 3개를 유지한 상태에서 단일 파생변수를 하나씩 추가 검증하는 전략이 가장 합리적입니다.

## 현재 실험 전략

1. `v5`를 실질 베이스라인으로 유지
2. `v5b` 핵심 3개는 유지
3. 추가 후보는 한 번에 많이 넣지 않고 단일 변수 기준으로 검증
4. 빠른 스크리닝은 `LightGBM 3-fold`
5. 최종 판단은 정식 `5-fold + 3모델 앙상블`

현재 우선순위:

1. `IVF_출산전환율`
2. `출산전환율`
3. `기증자정자_비율`

## 주요 파일

- [`train_v8plus.py`](./train_v8plus.py)
  초기 `v8plus` 베이스라인

- [`train_v8plus_v2.py`](./train_v8plus_v2.py)
  중요도 기반 1차 피처 다이어트 반영

- [`train_v8plus_v3.py`](./train_v8plus_v3.py)
  `v2` 이후 추가 정리와 확장 실험의 기준점

- [`train_v8plus_v5.py`](./train_v8plus_v5.py)
  현재 실질 메인 베이스. `v5b`에서 검증된 3개 파생변수 반영

- [`train_v8plus_v6a.py`](./train_v8plus_v6a.py)
  `v5 + IVF_출산전환율` 단일 추가 정식 검증 버전

- [`screen_v5_feature_ablation.py`](./screen_v5_feature_ablation.py)
  `v5` 핵심 피처 제거 영향 빠른 비교용 스크립트

- [`screen_v5_single_additions.py`](./screen_v5_single_additions.py)
  `v5` 기준 단일 추가 후보 빠른 비교용 스크립트

- [`feature_diet_search_v8plus.py`](./feature_diet_search_v8plus.py)
  중요도 기반 iterative feature diet 실험용 스크립트

## 디렉토리 구조

```text
data/
  raw/                원본 train/test/sample_submission

submissions/
  *.csv               제출 파일, 중요도 파일, 스크리닝 결과
  oof/                모델별 OOF / prediction numpy 저장소

src/
  기존 전처리 코드

train_v8plus*.py      실험 버전별 학습 스크립트
screen_*.py           빠른 스크리닝용 스크립트
```

## 실행 예시

정식 학습:

```bash
python train_v8plus_v5.py
python train_v8plus_v6a.py
```

빠른 스크리닝:

```bash
python screen_v5_feature_ablation.py
python screen_v5_single_additions.py
```

## 정리 계획

학습이 끝난 뒤 아래 순서로 정리하는 것을 권장합니다.

### 1. 유지할 파일

- 각 주요 버전의 최고 제출 파일 1개
- 각 주요 버전의 `feature_importance_*.csv`
- 최근 스크리닝 결과 CSV
- 현재 사용 중인 `train_v8plus_v*.py`
- 현재 기준 베스트와 다음 검증 후보 코드

### 2. 아카이브할 파일

- 오래된 제출 CSV 중 같은 버전의 하위 성능 파일
- 과거 세대(`v26`, `v27` 등) OOF 산출물
- 더 이상 참조하지 않는 중간 실험 CSV

권장 아카이브 위치:

- `submissions/archive/`
- `submissions/oof/archive/`

### 3. 지금은 건드리지 말 것

학습이 실행 중일 때는 아래 파일을 이동/삭제하지 않습니다.

- 가장 최근 생성 중인 `sub_v8plus_*.csv`
- 가장 최근 생성 중인 `feature_importance_*.csv`
- 현재 러닝 중인 버전의 `submissions/oof/*.npy`

### 4. GitHub 브랜치 업로드 전 체크리스트

- `README.md` 최신화
- 현재 기준 베스트 버전과 점수 명시
- 불필요한 대용량 산출물 제외 여부 확인
- `.gitignore`에 캐시/임시 파일 점검
- 커밋 메시지에 실험 기준점 기록

## 다음 할 일

- `v6a` 결과 확인
- `v6a`가 상승하면 새 베이스로 채택
- 이후 `출산전환율`, `기증자정자_비율` 순서로 단일 추가 검증
- 학습 종료 후 `submissions/`와 `submissions/oof/` 아카이브 정리
