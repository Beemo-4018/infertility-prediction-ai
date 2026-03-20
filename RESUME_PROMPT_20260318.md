# Resume Prompt

아래 프롬프트를 새 세션 시작 시 그대로 붙여넣으면 됩니다.

```text
지금 프로젝트는 c:\fertility-prediction-ai 에 있고, 난임 환자 임신 성공 여부 예측 대회 실험 중이야.

먼저 README.md와 멘토링_정리_20260318.md를 읽고 현재 상태를 파악해줘.

핵심 전제:
- 현재 기준 베이스는 train_v8plus_v7b.py
- 빠른 스크리닝 기준은 train_v8plus_v7b_lgb_screen.py
- 현재 로컬 최고 CV는 0.74083
- 제출 파일 sub_v8plus_v7b_cpu_0318_2229_cv0.74083.csv 로 실제 제출 점수 0.74216이 나왔음
- 예전 원본 점수 0.7420016202보다 올라간 상태
- 현재는 다시 피처 다이어트로 돌아가기보다 v7b 기준 단일 변경 실험을 이어가려는 상황

지금까지 핵심 흐름:
- v3 이후 추가 피처 다이어트는 실패
- v5 기반에서 파생변수 추가로 개선
- v6e에서 시술시기코드_난자정자조합_te 추가로 0.74079
- v7b에서 고령_x_단일배아이식 추가로 0.74083 달성
- 배란장애_x_배란유도유형_te는 최종 결합에서 실패(v7a 0.74070)
- 이식일5_단일이식(v7c), 저장배아_최적(v7d)은 각각 실패
- count 상단 clipping 실험은 대부분 실패했고 v7f만 보류
- 현재 다음 우선순위는 이상치보다 결측치 처리 실험

확인할 파일:
- README.md
- train_v8plus_v5.py
- train_v8plus_v7b.py
- train_v8plus_v7b_lgb_screen.py
- submissions/feature_importance_v8plus_v7b_cpu.csv
- submissions/cycle_code_profile_summary.txt

원하는 작업:
1. 현재 v7b 기준으로 다음 실험 우선순위를 다시 제안
2. 멘토링 내용 기준으로 부족한 부분(이상치 처리, 클래스 불균형 보정 등)을 반영할 가치가 있는지 분석
3. 필요하면 v7b 기반 다음 실험 코드를 직접 만들어줘

중요:
- 현재 기준점은 반드시 v7b로 잡아줘
- 빠른 비교는 v7b_screen 수치 기준으로 판단해줘
- 이전 v5b라는 표현 대신 지금 정리된 v5와 v7b 기준으로 설명해줘
- 무작정 많은 피처를 넣지 말고 단일 변경 위주로 제안해줘
```
