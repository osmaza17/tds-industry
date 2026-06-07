"""
final_test/heat_analysis.py — classify each TRAINING fault row as heating vs cooling.

Hypothesis (Martin / user): a real fault should coincide with a TEMPERATURE RISE.
Fault rows where temperature is FALLING may be mislabels/noise. Here we measure, per motor
and per sequence, how many labelled-fault rows are heating vs cooling, using a centred local
temperature slope. This both tests the hypothesis and tells us how much we'd remove.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(ROOT, 'final_test'))
import calib  # reuse the exact harness data loading

W = 10  # centred half-window (10 samples = 1.0 s at 10 Hz) for the slope sign


def centred_slope(temp, w=W):
    """temp[t+w]-temp[t-w] within a sequence; sign = heating(+)/cooling(-)."""
    x = np.asarray(temp, dtype=float)
    n = len(x)
    fwd = np.empty(n); fwd[:] = np.nan
    fwd[:n-w] = x[w:]
    bwd = np.empty(n); bwd[:] = np.nan
    bwd[w:] = x[:n-w]
    s = fwd - bwd
    # edges: fall back to one-sided diff
    s[:w] = x[min(w, n-1)] - x[0] if n > w else 0.0
    s[n-w:] = x[n-1] - x[max(n-1-w, 0)]
    return s


def main():
    train_df, _ = calib.load_all()
    print('train rows', len(train_df))
    DEAD = 0.0   # slope threshold; |s|<=DEAD counts as flat
    for m in range(1, 7):
        lc = f'data_motor_{m}_label'
        tc = f'data_motor_{m}_temperature'
        if lc not in train_df.columns:
            continue
        df = train_df.dropna(subset=[lc])
        # compute centred slope per sequence
        slopes = np.concatenate([centred_slope(g[tc].values)
                                 for _, g in df.groupby('test_condition', sort=False)])
        lab = df[lc].values.astype(int)
        fault = lab == 1
        nf = int(fault.sum())
        if nf == 0:
            print(f'M{m}: no faults'); continue
        sf = slopes[fault]
        heat = int((sf > DEAD).sum()); cool = int((sf < -DEAD).sum()); flat = int((sf == 0).sum())
        print(f'M{m}: {nf} fault rows  | heating={heat} ({heat/nf:.0%})  cooling={cool} ({cool/nf:.0%})  flat={flat}')
        # per-sequence breakdown for this motor
        tmp = df.assign(_s=slopes, _f=fault)
        for seq, g in tmp[tmp['_f']].groupby('test_condition', sort=False):
            h = int((g['_s'] > DEAD).sum()); c = int((g['_s'] < -DEAD).sum())
            print(f'      {seq}: faults={len(g):5d}  heating={h:5d}  cooling={c:5d}  ({h/(h+c+1e-9):.0%} heat)')


if __name__ == '__main__':
    main()
