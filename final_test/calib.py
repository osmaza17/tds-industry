"""
final_test/calib.py — train the champion per-motor models and DUMP TEST-SET PROBABILITIES.

Goal of the final_test campaign: the champion scores public 0.866 / private 0.304.
Diagnosis (final_test/LOG.md) = over-prediction on a low-prevalence private subset.
To design conservative / prevalence-calibrated submissions we need the model's
per-row probability for each motor on the test set, not just the 0/1 decision.

This reuses the harness functions from pruebas/exp_C.py (no edits to the originals) and
replicates its data loading + per-motor training, but instead of writing a binary
submission it writes:
    final_test/outputs/test_probs.csv   (idx, test_condition, prob_m1..prob_m6)
    final_test/outputs/train_prev.json  (per-motor training prevalence + chosen champ threshold)

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe final_test/calib.py
"""
import os, sys, json, time, glob, shutil
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(ROOT, 'pruebas'))
sys.path.insert(1, os.path.join(ROOT, 'kaggle_data_challenge'))
import exp_C as H
from utility import read_all_test_data_from_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.ndimage import binary_closing

SEED = 42
OUT  = os.path.join(ROOT, 'final_test', 'outputs')
os.makedirs(OUT, exist_ok=True)

# Champion config (== exp_C 'c_win_m1')
MOTOR_FEATURES = H.FEATS_WIN_M1
INJ            = H.INJ_V2
PEAK           = H.PEAK_FULL


def load_all():
    np.random.seed(SEED)
    train_path = os.path.join(ROOT, 'data', 'training_data') + '/'
    test_path  = os.path.join(ROOT, 'data', 'testing_data') + '/'
    train_df = read_all_test_data_from_path(train_path, H.pre_processing, is_plot=False)

    additional_base = os.path.join(ROOT, 'data', 'additional_data')
    groups = ['additional_data_20240524_group_6', 'additional_training_data_group_1',
              'additional_training_data_group_7']
    additional_dfs = []
    for group in groups:
        group_path = os.path.join(additional_base, group)
        xlsx_path  = os.path.join(group_path, 'Test conditions.xlsx')
        copy_xlsx  = os.path.join(group_path, 'Test conditions copy.xlsx')
        if not os.path.exists(xlsx_path) and os.path.exists(copy_xlsx):
            try: shutil.copy(copy_xlsx, xlsx_path)
            except Exception: pass
        if not os.path.exists(group_path):
            continue
        temp_dir = os.path.join(additional_base, f"{group}_temp_FT")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        if os.path.exists(xlsx_path):
            shutil.copy(xlsx_path, os.path.join(temp_dir, 'Test conditions.xlsx'))
        for subdir in [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]:
            sp = os.path.join(group_path, subdir)
            csvs = [f for f in os.listdir(sp) if f.endswith('.csv')]
            ok = bool(csvs)
            for cf in csvs:
                try:
                    with open(os.path.join(sp, cf)) as f:
                        if len([l for l in f if l.strip()]) <= 1:
                            ok = False; break
                except Exception:
                    ok = False; break
            if ok:
                shutil.copytree(sp, os.path.join(temp_dir, subdir), dirs_exist_ok=True)
        try:
            additional_dfs.append(read_all_test_data_from_path(temp_dir + '/', H.pre_processing, is_plot=False))
        except Exception as e:
            print('  add load fail', group, e)
        shutil.rmtree(temp_dir, ignore_errors=True)
    if additional_dfs:
        train_df = pd.concat([train_df] + additional_dfs, ignore_index=True)

    test_df = read_all_test_data_from_path(test_path, H.pre_processing, is_plot=False)

    # movement descriptions -> per-motor desc one-hots
    tc_files  = glob.glob(os.path.join(ROOT, 'data') + '/**/Test conditions*.xlsx', recursive=True)
    tc_files += glob.glob(additional_base + '/**/Test conditions*.xlsx', recursive=True)
    tc_dfs = []
    for f in set(tc_files):
        try: tc_dfs.append(pd.read_excel(f))
        except Exception: pass
    all_tc = pd.concat(tc_dfs, ignore_index=True)
    all_tc['Description'] = all_tc['Description'].fillna('').astype(str).str.lower()
    desc_map = all_tc.set_index('Test id')['Description'].to_dict()
    out = {}
    for label, df in [('train', train_df), ('test', test_df)]:
        df['raw_desc']        = df['test_condition'].map(desc_map).fillna('')
        df['desc_transfer']   = df['raw_desc'].str.contains('transfer').astype(int)
        df['desc_not_moving'] = df['raw_desc'].str.contains('not moving').astype(int)
        df['desc_turn_motor'] = df['raw_desc'].str.contains('turn motor').astype(int)
        df['desc_chute_cube'] = df['raw_desc'].str.contains('chute cube').astype(int)
        new_cols = pd.DataFrame({
            f'data_motor_{m}_{feat}': df[feat].values
            for m in range(1, 7)
            for feat in ['desc_transfer', 'desc_not_moving', 'desc_turn_motor', 'desc_chute_cube']
        }, index=df.index)
        out[label] = pd.concat([df, new_cols], axis=1)
    return out['train'], out['test']


def main():
    t0 = time.time()
    print('[1] loading...')
    train_df, test_df = load_all()
    print(f'  train {train_df.shape}, test {test_df.shape}')

    prob_cols = {}
    meta = {}
    print('[2] training per motor (champion config)...')
    for motor_id in range(1, 7):
        features  = [f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]]
        label_col = f'data_motor_{motor_id}_label'
        motor_df  = train_df.dropna(subset=[label_col]).copy()
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
        peak_low, peak_high = PEAK[motor_id]
        s = H.synthesize_for_motor(raw_train, motor_id, n_sequences=INJ[motor_id],
                                   peak_low=peak_low, peak_high=peak_high)
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

        # champion threshold (val-tuned, with optional closing)
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
        meta[motor_id] = {'prevalence': real_prev, 'champ_threshold': thr,
                          'champ_closing': bool(closing), 'val_f1': vf1, 'best_params': best_params}
        print(f'  M{motor_id}: prev={real_prev:.4f} thr={thr} close={closing} valF1={vf1:.3f} '
              f'meanP={probs.mean():.3f} pos@thr={(probs>=thr).sum()}')

    res = pd.DataFrame({'idx': test_df['idx'] if 'idx' in test_df else np.arange(len(test_df)),
                        'test_condition': test_df['test_condition'].values})
    for k, v in prob_cols.items():
        res[k] = v
    res.to_csv(os.path.join(OUT, 'test_probs.csv'), index=False)
    with open(os.path.join(OUT, 'train_prev.json'), 'w') as f:
        json.dump({str(k): v for k, v in meta.items()}, f, indent=2)
    print(f'[done] wrote test_probs.csv ({len(res)} rows) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
