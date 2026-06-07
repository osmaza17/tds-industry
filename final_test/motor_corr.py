"""
final_test/motor_corr.py — do motor faults co-occur? (does motor X breaking imply Y breaking?)

Measures, on the TRAINING labels:
  1. Row-level phi correlation between the 6 fault-label columns.
  2. Conditional co-occurrence P(Y=1 | X=1) for every ordered pair.
  3. Lagged co-occurrence: P(Y=1 at t+k | X=1 at t) to catch "X breaks, then Y breaks".
  4. Per-sequence: which motors fail in the same sequence.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(ROOT, 'final_test'))
import calib

LAB = [f'data_motor_{m}_label' for m in range(1, 7)]


def main():
    train_df, test_df = calib.load_all()
    df = train_df
    # only rows where ALL motor labels are present (aligned)
    present = df[LAB].notna().all(axis=1)
    L = df.loc[present, LAB].astype(int)
    print(f'rows with all 6 labels present: {len(L)}')
    print('per-motor fault rate:')
    print((L.mean()*100).round(2).to_string())

    print('\n=== phi (Pearson) correlation between fault labels ===')
    print(L.corr().round(2).to_string())

    print('\n=== P(row Y=1 | X=1)  (rows: X given, cols: Y) ===')
    M = np.zeros((6, 6))
    for i in range(6):
        xi = L.iloc[:, i].values == 1
        nX = xi.sum()
        for j in range(6):
            if nX == 0:
                M[i, j] = np.nan
            else:
                M[i, j] = (L.iloc[:, j].values[xi] == 1).mean()
    print(pd.DataFrame((M*100).round(1), index=[f'M{i}' for i in range(1, 7)],
                       columns=[f'M{j}' for j in range(1, 7)]).to_string())

    print('\n=== lagged: P(Y=1 at t+k | X=1 at t), within sequence, k=+10 (1s) ===')
    K = 10
    Mlag = np.zeros((6, 6))
    cntX = np.zeros(6)
    accum = np.zeros((6, 6))
    for seq, g in df.groupby('test_condition', sort=False):
        if not g[LAB].notna().all(axis=1).all():
            continue
        arr = g[LAB].astype(int).values
        n = len(arr)
        if n <= K:
            continue
        for i in range(6):
            xi = arr[:n-K, i] == 1
            cntX[i] += xi.sum()
            for j in range(6):
                accum[i, j] += (arr[K:, j][xi] == 1).sum()
    for i in range(6):
        for j in range(6):
            Mlag[i, j] = accum[i, j] / cntX[i] if cntX[i] > 0 else np.nan
    print(pd.DataFrame((Mlag*100).round(1), index=[f'M{i}' for i in range(1, 7)],
                       columns=[f'M{j}' for j in range(1, 7)]).to_string())

    print('\n=== per-sequence: motors with >=1 fault (TRAIN) ===')
    for seq, g in df.groupby('test_condition', sort=False):
        present_motors = [m for m in range(1, 7)
                          if g[f'data_motor_{m}_label'].notna().any()
                          and (g[f'data_motor_{m}_label'] == 1).any()]
        if present_motors:
            print(f'  {seq}: {present_motors}')


if __name__ == '__main__':
    main()
