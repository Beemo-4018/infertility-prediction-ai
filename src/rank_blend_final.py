# -*- coding: utf-8 -*-
"""
rank_blend_final.py
best(0.74205) 65% + v_clean10(0.73841) 35% Rank Averaging
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SAVE_PATH = '/Users/admin/Downloads/infertility-prediction-ai/data/submissions/'

# ── 파일 경로 ──────────────────────────────────────────────
BEST_FILE    = SAVE_PATH + 'submission_0316_0909_blend_50_50.csv'
CLEAN10_FILE = SAVE_PATH + 'submission_0317_1151_vclean10_auc0p73841.csv'

# ── 비율 ───────────────────────────────────────────────────
W_BEST    = 0.65
W_CLEAN10 = 0.35

# ── 로드 ───────────────────────────────────────────────────
best    = pd.read_csv(BEST_FILE)
clean10 = pd.read_csv(CLEAN10_FILE)

print(f'best    shape: {best.shape}')
print(f'clean10 shape: {clean10.shape}')

# ── Rank Averaging ─────────────────────────────────────────
n = len(best)
rank_best    = rankdata(best['probability'])    / n
rank_clean10 = rankdata(clean10['probability']) / n

blended = W_BEST * rank_best + W_CLEAN10 * rank_clean10

# ── 저장 ───────────────────────────────────────────────────
out = best.copy()
out['probability'] = blended

filename = SAVE_PATH + 'submission_0317_blend65_35_final.csv'
out.to_csv(filename, index=False)

print(f'\n저장 완료 : {filename}')
print(f'비율      : best {W_BEST*100:.0f}% + v_clean10 {W_CLEAN10*100:.0f}%')
print(f'prob 범위 : {blended.min():.4f} ~ {blended.max():.4f}')
print(f'\n→ 이 파일 제출하세요!')