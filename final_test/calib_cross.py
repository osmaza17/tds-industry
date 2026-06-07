"""
final_test/calib_cross.py — retrain each motor WITH cross-motor features, dump test probs.

Motivation: M2<->M4 fault labels are strongly correlated (phi=0.77) in training; M1 weakly with
M2/M4/M6. So another motor's thermal signature may help detect a motor's fault. We give every motor
its own champion features PLUS a compact summary of the OTHER motors (temperature + temperature_dev_20).

Modes (argv[1]):
  all   every motor sees all 5 other motors' [temperature, temperature_dev_20]
  pair  only the strong M2<->M4 coupling (M2 gets M4's, M4 gets M2's); others = champion

Writes outputs/test_probs_cross_<mode>.csv (+ meta).
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
MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
CROSS_FEATS = ['temperature', 'temperature_dev_20']


def feature_columns(motor_id):
    own = [f'data_motor_{motor_id}_{feat}' for feat in calib.MOTOR_FEATURES[motor_id]]
    if MODE == 'all':
        others = [o for o in range(1, 7) if o != motor_id]
    elif MODE == 'pair':
        pair = {2: [4], 4: [2]}
        others = pair.get(motor_id, [])
    else:
        raise ValueError(MODE)
    cross = [f'data_motor_{o}_{f}' for o in others for f in CROSS_FEATS]
    return own + cross


def main():
    t0 = time.time()
    print(f'[cross mode={MODE}] loading...')
    train_df, test_df = calib.load_all()
    prob_cols, meta = {}, {}
    for motor_id in range(1, 7):
        features = feature_columns(motor_id)
        features = [f for f in features if f in train_df.columns]
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

        # synthesize on OWN features only (injection touches this motor's temperature);
        # cross columns come along unchanged from the source normal sequence.
        np.random.seed(SEED * 1000 + motor_id)
        pl, ph = calib.PEAK[motor_id]
        s = H.synthesize_for_motor(raw_train, motor_id, n_sequences=calib.INJ[motor_id], peak_low=pl, peak_high=ph)
        aug = pd.concat([raw_train, s], ignore_index=True) if not s.empty else raw_train
        # synthetic rows may miss cross columns -> fill 0
        for f in features:
            if f not in aug.columns:
                aug[f] = 0.0
        aug[features] = aug[features].fillna(0.0)
        X_train, y_train = aug[features], aug[label_col]
        sw = np.ones(len(y_train)); sw[y_train == 1] = 2.0
        X_val = raw_val[features].fillna(0.0) if (va_f + va_n) else None
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

        Xtest = test_df[features].fillna(0.0)
        probs = model0.predict_proba(Xtest)[:, 1]
        prob_cols[f'prob_m{motor_id}'] = probs
        meta[motor_id] = {'prevalence': real_prev, 'champ_threshold': thr, 'champ_closing': bool(closing),
                          'val_f1': vf1, 'best_params': best_params, 'n_cross': len(features)}
        print(f'  M{motor_id}: nfeat={len(features)} thr={thr} valF1={vf1:.3f} pos@thr={(probs>=thr).sum()}')

    res = pd.DataFrame({'idx': test_df['idx'] if 'idx' in test_df else np.arange(len(test_df)),
                        'test_condition': test_df['test_condition'].values})
    for k, v in prob_cols.items():
        res[k] = v
    res.to_csv(os.path.join(OUT, f'test_probs_cross_{MODE}.csv'), index=False)
    json.dump({str(k): v for k, v in meta.items()},
              open(os.path.join(OUT, f'train_prev_cross_{MODE}.json'), 'w'), indent=2)
    print(f'[done] outputs/test_probs_cross_{MODE}.csv in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
