# -*- coding: utf-8 -*-
"""
Rank Averaging 블렌딩
- probability 평균 대신 rank 평균 사용
- 각 모델의 예측 분포 차이를 정규화해서 블렌딩 효과 극대화
실행 방법: python src/rank_blend.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'

# ====================================================
# 블렌딩할 파일 목록 (LB 기준 좋은 것들)
# ====================================================
files = {
    'v5'   : SAVE_PATH + 'submission_0313_2308_auc0p74060_fixed.csv',  # LB 0.74198
    'v8'   : SAVE_PATH + 'submission_0314_1838_auc0p74067.csv',         # LB 0.74195
    'v9'   : SAVE_PATH + 'submission_0316_1207_auc0p74067.csv',         # CV 0.74067
    'v11t' : './submissions/submission_0316_1533_tree_only_auc0p74065.csv',  # CV 0.74065
}

# ====================================================
# Rank Averaging 함수
# ====================================================
def rank_avg(preds_dict, weights=None):
    """
    각 예측값을 rank로 변환 후 가중 평균
    - 분포 차이를 정규화해서 블렌딩 효과 극대화
    - probability 평균보다 LB 안정화 효과 있음
    """
    names = list(preds_dict.keys())
    n     = len(preds_dict[names[0]])

    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}

    ranked = {}
    for name, pred in preds_dict.items():
        # rank를 0~1 사이로 정규화
        ranked[name] = pd.Series(pred).rank(method='average').values / n

    result = np.zeros(n)
    for name in names:
        result += weights[name] * ranked[name]

    return result

# ====================================================
# 파일 로드
# ====================================================
print('=' * 55)
print('  Rank Averaging 블렌딩')
print('=' * 55)

dfs   = {}
preds = {}
for name, path in files.items():
    try:
        df          = pd.read_csv(path)
        dfs[name]   = df
        preds[name] = df['probability'].values
        print(f'  ✅ {name}: {path.split("/")[-1]}')
    except FileNotFoundError:
        print(f'  ❌ {name}: 파일 없음 → 스킵')

print(f'\n  로드된 파일: {list(preds.keys())}')

# ID 순서 일치 확인
base_ids = dfs[list(dfs.keys())[0]]['ID']
for name, df in dfs.items():
    assert (df['ID'] == base_ids).all(), f'{name} ID 순서 불일치!'
print('  ✅ ID 순서 일치 확인\n')

# ====================================================
# 다양한 조합 생성
# ====================================================
os.makedirs(SAVE_PATH, exist_ok=True)
timestamp = datetime.now().strftime('%m%d_%H%M')
base_df   = dfs[list(dfs.keys())[0]].copy()

combos = []

# [1] v5 + v8 Rank Avg (현재 최고 prob avg 0.74205와 비교)
if 'v5' in preds and 'v8' in preds:
    combo = rank_avg({'v5': preds['v5'], 'v8': preds['v8']})
    fname = f'{SAVE_PATH}submission_{timestamp}_rank_v5_v8.csv'
    out   = base_df.copy()
    out['probability'] = combo
    out.to_csv(fname, index=False)
    combos.append(('rank_v5_v8', fname))
    print(f'  생성: rank_v5_v8')

# [2] v5 + v8 + v9 Rank Avg
if all(k in preds for k in ['v5', 'v8', 'v9']):
    combo = rank_avg({'v5': preds['v5'], 'v8': preds['v8'], 'v9': preds['v9']})
    fname = f'{SAVE_PATH}submission_{timestamp}_rank_v5_v8_v9.csv'
    out   = base_df.copy()
    out['probability'] = combo
    out.to_csv(fname, index=False)
    combos.append(('rank_v5_v8_v9', fname))
    print(f'  생성: rank_v5_v8_v9')

# [3] v5 + v8 + v11t Rank Avg
if all(k in preds for k in ['v5', 'v8', 'v11t']):
    combo = rank_avg({'v5': preds['v5'], 'v8': preds['v8'], 'v11t': preds['v11t']})
    fname = f'{SAVE_PATH}submission_{timestamp}_rank_v5_v8_v11t.csv'
    out   = base_df.copy()
    out['probability'] = combo
    out.to_csv(fname, index=False)
    combos.append(('rank_v5_v8_v11t', fname))
    print(f'  생성: rank_v5_v8_v11t')

# [4] v5 + v8 + v9 + v11t Rank Avg (전체)
if all(k in preds for k in ['v5', 'v8', 'v9', 'v11t']):
    combo = rank_avg(preds)
    fname = f'{SAVE_PATH}submission_{timestamp}_rank_all.csv'
    out   = base_df.copy()
    out['probability'] = combo
    out.to_csv(fname, index=False)
    combos.append(('rank_all', fname))
    print(f'  생성: rank_all')

print('\n' + '=' * 55)
print('  생성된 파일 목록')
print('=' * 55)
for name, fname in combos:
    print(f'  {name:<25} : {fname.split("/")[-1]}')

print('\n  추천 제출 순서:')
print('  1. rank_v5_v8      ← 현재 최고(prob avg 0.74205)와 직접 비교')
print('  2. rank_v5_v8_v9   ← 3개 모델 rank avg')
print('  3. rank_all        ← 4개 모델 전체')