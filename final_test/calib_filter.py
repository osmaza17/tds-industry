"""
final_test/calib_filter.py — retrain dropping "cooling" fault labels, dump test probabilities.

Tests the hypothesis: a real fault coincides with a TEMPERATURE RISE; fault labels on cooling
temperature are suspect and should not be trained on. We relabel the suspect fault rows to 0
(normal) BEFORE training, keeping the synthetic heating injections unchanged.

Modes (argv[1]):
  slope     per-ROW: keep fault rows with centred slope > 0 (rising edge only); drop falling+flat
  slope_nz  per-ROW: keep fault rows with slope >= 0 (rising + plateau); drop only the falling edge
  block     per-BLOCK: keep a contiguous fault block only if its mean temp is ABOVE the sequence's
            normal (non-fault) baseline by >= margin; drop "depressed/cooling-type" blocks entirely

Usage: python final_test/calib_filter.py block   -> writes outputs/test_probs_block.csv (+ meta)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.ndimage import binary_closing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(ROOT, 'final_test'))
sys.path.insert(1, os.path.join(ROOT, 'pruebas'))
import calib
import exp_C as H

SEED = 42
OUT  = calib.OUT
MODE = sys.argv[1] if len(sys.argv) > 1 else 'block'
W = 10                 # centred half-window for slope sign
BLOCK_MARGIN = 0.5     # °C above baseline to call a block "heating-type"


def centred_slope(temp, w=W):
    x = np.asarray(temp, dtype=float); n = len(x)
    fwd = np.full(n, np.nan); fwd[:n-w] = x[w:]
    bwd = np.full(n, np.nan); bwd[w:] = x[:n-w]
    s = fwd - bwd
    s[:w] = (x[min(w, n-1)] - x[0]) if n > w else 0.0
    s[n-w:] = x[n-1] - x[max(n-1-w, 0)]
    return s


def keep_mask_for_sequence(temp, lab):
    """Return boolean array: which fault rows to KEEP as label=1 (others -> 0)."""
    lab = np.asarray(lab).astype(int)
    temp = np.asarray(temp, dtype=float)
    keep = lab.copy()                       # start from current labels
    fault = lab == 1
    if fault.sum() == 0:
        return keep
    if MODE in ('slope', 'slope_nz'):
        s = centred_slope(temp)
        if MODE == 'slope':
            drop = fault & (s <= 0)
        else:
            drop = fault & (s < 0)
        keep[drop] = 0
        return keep
    if MODE == 'block':
        if (~fault).sum() < 30:        # ~all-fault sequence: no usable normal baseline -> keep all
            return keep
        base = temp[~fault].mean()
        # contiguous fault blocks
        i, n = 0, len(lab)
        while i < n:
            if lab[i] == 1:
                j = i
                while j < n and lab[j] == 1:
                    j += 1
                block_mean = temp[i:j].mean()
                if block_mean < base + BLOCK_MARGIN:      # depressed / not clearly heating -> drop
                    keep[i:j] = 0
                i = j
            else:
                i += 1
        return keep
    raise ValueError(MODE)


def apply_filter(train_df):
    train_df = train_df.reset_index(drop=True)
    removed = {}
    for m in range(1, 7):
        lc = f'data_motor_{m}_label'; tc = f'data_motor_{m}_temperature'
        if lc not in train_df.columns:
            continue
        before = int((train_df[lc] == 1).sum())
        new_lab = train_df[lc].values.copy()
        for seq, g in train_df.groupby('test_condition', sort=False):
            mask = g[lc].notna().values
            if mask.sum() == 0:
                continue
            km = keep_mask_for_sequence(g[tc].values[mask], g[lc].values[mask])
            rows_idx = g.index.values[mask]
            new_lab[rows_idx] = km
        train_df[lc] = new_lab
        after = int((train_df[lc] == 1).sum())
        removed[m] = (before, after)
    return train_df, removed


def main():
    t0 = time.time()
    print(f'[mode={MODE}] loading...')
    train_df, test_df = calib.load_all()
    train_df, removed = apply_filter(train_df)
    print('  fault rows kept per motor (before -> after):')
    for m, (b, a) in removed.items():
        print(f'    M{m}: {b} -> {a}  (dropped {b-a}, {100*(b-a)/max(b,1):.0f}%)')

    prob_cols, meta = {}, {}
    for motor_id in range(1, 7):
        features = [f'data_motor_{motor_id}_{feat}' for feat in calib.MOTOR_FEATURES[motor_id]]
        label_col = f'data_motor_{motor_id}_label'
        motor_df = train_df.dropna(subset=[label_col]).copy()
        real_prev = float((motor_df[label_col] == 1).mean())
        seqs = motor_df['test_condition'].unique()
        fail_seqs = [s for s in seqs if (motor_df[motor_df['test_condition'] == s][label_col] == 1).any()]
        norm_seqs = [s for s in seqs if s not in fail_seqs]
        if len(fail_seqs) > 1:
            tr_f, va_f = train_test_split(fail_seqs, test_size=max(1, int(0.2*len(fail_seqs))), random_state=42)
        else:
            tr_f, va_f = fail_seqs, []
        if len(norm_seqs) > 1:
            tr_n, va_n = train_test_split(norm_seqs, test_size=max(1, int(0.2*len(norm_seqs))), random_state=42)
        else:
            tr_n, va_n = norm_seqs, []
        raw_train = motor_df[motor_df['test_condition'].isin(tr_f + tr_n)].copy()
        raw_val   = motor_df[motor_df['test_condition'].isin(va_f + va_n)].copy()

        np.random.seed(SEED * 1000 + motor_id)
        pl, ph = calib.PEAK[motor_id]
        s = H.synthesize_for_motor(raw_train, motor_id, n_sequences=calib.INJ[motor_id], peak_low=pl, peak_high=ph)
        aug = pd.concat([raw_train, s], ignore_index=True) if not s.empty else raw_train
        X_train, y_train = aug[features], aug[label_col]
        sw = np.ones(len(y_train)); sw[y_train == 1] = 2.0
        X_val = raw_val[features] if (va_f + va_n) else None
        y_val = raw_val[label_col] if (va_f + va_n) else None

        grid = H.PARAM_GRIDS['histgb']
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        best_params, best_score, model0 = grid[0], -1.0, None
        for params in grid:
            model = H.build_model('histgb', params)
            model.fit(X_train, y_train, sample_weight=sw)
            if X_val is not None and len(np.unique(y_val)) > 1:
                p = model.predict_proba(X_val)[:, 1]
                sc = max(f1_score(y_val, (p >= t).astype(int), pos_label=1, zero_division=0) for t in thresholds)
            else:
                sc = 0.0
            if sc > best_score:
                best_score, best_params, model0 = sc, params, model
            if X_val is None:
                break
        thr, closing, vf1 = 0.5, True, 0.0
        if X_val is not None and len(np.unique(y_val)) > 1:
            pv = model0.predict_proba(X_val)[:, 1]
            for t in thresholds:
                base = (pv >= t).astype(int)
                for uc in (False, True):
                    pred = binary_closing(base, structure=np.ones(5)).astype(int) if uc else base
                    sc = f1_score(y_val, pred, pos_label=1, zero_division=0)
                    if sc > vf1: vf1, thr, closing = sc, t, uc

        probs = model0.predict_proba(test_df[features])[:, 1]
        prob_cols[f'prob_m{motor_id}'] = probs
        meta[motor_id] = {'prevalence': real_prev, 'champ_threshold': thr, 'champ_closing': bool(closing),
                          'val_f1': vf1, 'best_params': best_params}
        print(f'  M{motor_id}: prev={real_prev:.4f} thr={thr} valF1={vf1:.3f} pos@thr={(probs>=thr).sum()}')

    res = pd.DataFrame({'idx': test_df['idx'] if 'idx' in test_df else np.arange(len(test_df)),
                        'test_condition': test_df['test_condition'].values})
    for k, v in prob_cols.items():
        res[k] = v
    res.to_csv(os.path.join(OUT, f'test_probs_{MODE}.csv'), index=False)
    json.dump({str(k): v for k, v in meta.items()},
              open(os.path.join(OUT, f'train_prev_{MODE}.json'), 'w'), indent=2)
    print(f'[done] outputs/test_probs_{MODE}.csv in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
