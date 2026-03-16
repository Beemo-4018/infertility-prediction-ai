# -*- coding: utf-8 -*-
"""
v5 + v8 제출 파일 블렌딩
실행 방법: python src/blend.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'

# ====================================================
# 블렌딩할 파일 경로 설정
# ====================================================
FILE_V5 = SAVE_PATH + 'submission_0313_2308_auc0p74060_fixed.csv'  # LB 0.74198
FILE_V8 = SAVE_PATH + 'submission_0314_1838_auc0p74067.csv'         # LB 0.74195

# ====================================================
# 블렌딩 비율 설정
# ====================================================
# v5가 LB 기준 살짝 더 좋으므로 v5에 가중치 더 줌
BLEND_CONFIGS = [
    ('50_50',  0.50, 0.50),
    ('60_40',  0.60, 0.40),   # v5 가중치 더 높게
    ('40_60',  0.40, 0.60),   # v8 가중치 더 높게
]

# ====================================================
# 실행
# ====================================================
print('=' * 55)
print('  v5 + v8 블렌딩')
print('=' * 55)

v5 = pd.read_csv(FILE_V5)
v8 = pd.read_csv(FILE_V8)

print(f'  v5 shape : {v5.shape}')
print(f'  v8 shape : {v8.shape}')
print(f'  v5 prob  : mean={v5["probability"].mean():.5f}')
print(f'  v8 prob  : mean={v8["probability"].mean():.5f}')

# ID 순서 맞는지 확인
assert (v5['ID'] == v8['ID']).all(), 'ID 순서 불일치!'
print('\n  ✅ ID 순서 일치 확인')

os.makedirs(SAVE_PATH, exist_ok=True)
timestamp = datetime.now().strftime('%m%d_%H%M')

print('\n  생성된 블렌딩 파일:')
print('  ' + '-' * 45)

for name, w5, w8 in BLEND_CONFIGS:
    blended = v5.copy()
    blended['probability'] = w5 * v5['probability'] + w8 * v8['probability']
    filename = f'{SAVE_PATH}submission_{timestamp}_blend_{name}.csv'
    blended.to_csv(filename, index=False)
    print(f'  {name}: v5×{w5} + v8×{w8}  →  {filename.split("/")[-1]}')

print('\n  추천 제출 순서: 50_50 → 60_40 → 40_60')
print('  제출 후 LB 점수 기록해서 최적 비율 확인하세요!')