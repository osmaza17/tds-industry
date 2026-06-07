"""
final_test/make_sub.py — build submission CSVs from final_test/outputs/test_probs.csv.

Replicates exp_C's submission assembly (per-sequence: threshold -> binary_closing -> remove_short_runs)
so we can rapidly try per-motor thresholds / rank-calibration / post-processing WITHOUT retraining,
then submit to read the private score.

API:
  build(thr, postproc=None, rank=None, name=...) -> writes final_test/outputs/<name>.csv
    thr:    dict {motor: absolute_threshold}   (used when rank is None for that motor)
    rank:   dict {motor: fraction}             (predict top-`fraction` by prob; overrides thr)
    postproc: {'min_run': {m:int}|int, 'close': {m:int}|int, 'use_closing': {m:bool}}
"""
import os, json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'final_test', 'outputs')
SS   = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
PR   = None
META = None


def set_source(probs_csv='test_probs.csv', meta_json='train_prev.json'):
    """Point the builder at a given probabilities/meta pair (default = champion)."""
    global PR, META
    PR   = pd.read_csv(os.path.join(OUT, probs_csv))
    META = json.load(open(os.path.join(OUT, meta_json)))
    assert (SS['idx'].values == PR['idx'].values).all()
    return PR, META


set_source()  # default champion source


def remove_short_runs(y, min_run):
    y = np.asarray(y).astype(int).copy()
    if min_run <= 1: return y
    n = len(y); i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1: j += 1
            if (j - i) < min_run: y[i:j] = 0
            i = j
        else:
            i += 1
    return y


def _rank_threshold(probs, frac):
    frac = min(max(frac, 0.0), 1.0)
    if frac <= 0: return np.inf          # predict nothing
    return float(np.quantile(probs, 1.0 - frac))


def build(name, thr=None, rank=None, postproc=None, exclude=0):
    thr      = thr or {}
    rank     = rank or {}
    postproc = postproc or {}
    min_run  = postproc.get('min_run', 0)
    close    = postproc.get('close', None)
    use_clo  = postproc.get('use_closing', {})
    out = SS.copy()

    # per-motor global rank thresholds use the WHOLE test distribution (like exp_C 'rank')
    motor_thr = {}
    for m in range(1, 7):
        col = f'prob_m{m}'
        if m in rank:
            motor_thr[m] = _rank_threshold(PR[col].values, rank[m])
        elif m in thr:
            motor_thr[m] = thr[m]
        else:
            motor_thr[m] = META[str(m)]['champ_threshold']

    for test_id in SS['test_condition'].unique():
        mask = (PR['test_condition'] == test_id).values
        for m in range(1, 7):
            p = PR.loc[mask, f'prob_m{m}'].values
            yp = (p >= motor_thr[m]).astype(int)
            cs = close.get(m, None) if isinstance(close, dict) else close
            uc = use_clo.get(m, META[str(m)]['champ_closing']) if isinstance(use_clo, dict) else use_clo
            if cs:
                yp = binary_closing(yp, structure=np.ones(cs)).astype(int)
            elif uc:
                yp = binary_closing(yp, structure=np.ones(3)).astype(int)
            mr = min_run.get(m, 0) if isinstance(min_run, dict) else min_run
            if mr > 0:
                yp = remove_short_runs(yp, mr)
            out.loc[mask, f'data_motor_{m}_label'] = yp
    for m in range(1, 7):
        out[f'data_motor_{m}_label'] = out[f'data_motor_{m}_label'].astype(int)
    if exclude:
        out[f'data_motor_{exclude}_label'] = -1
    path = os.path.join(OUT, f'{name}.csv')
    out.to_csv(path, index=False)
    counts = {f'M{m}': int((out[f'data_motor_{m}_label'] == 1).sum()) for m in range(1, 7)}
    return path, counts


# Champion replica config (== exp_C c_win_m1): val thresholds + PP_CHAMP3
CHAMP_PP = {'min_run': {4: 40, 6: 20}, 'close': {4: 5}}


if __name__ == '__main__':
    # validation: reproduce champion and diff against the real champion CSV
    path, counts = build('repro_champ', postproc=CHAMP_PP)
    champ = pd.read_csv(os.path.join(ROOT, 'pruebas', 'outputs', 'C', 'sub_C_c_win_m1_FULL_seed42.csv'))
    mine  = pd.read_csv(path)
    diffs = {f'M{m}': int((champ[f'data_motor_{m}_label'].values != mine[f'data_motor_{m}_label'].values).sum())
             for m in range(1, 7)}
    print('repro counts:', counts)
    print('row diffs vs real champion:', diffs)
    print('TOTAL diff rows:', sum(diffs.values()))
