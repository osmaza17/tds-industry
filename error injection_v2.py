import os
import sys
import time
import itertools
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from scipy.ndimage import binary_closing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Add utility path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(_PROJECT_ROOT, 'kaggle_data_challenge'))
from utility import read_all_test_data_from_path

# ── Injection config per motor ────────────────────────────────────────────────
# M3 and M5 have very few real faults → more injections with subtle peaks
N_INJECT   = {1: 4,  2: 4,  3: 16, 4: 4,  5: 16, 6: 4}
PEAK_RANGE = {1: (2, 10), 2: (2, 10), 3: (1, 4), 4: (2, 10), 5: (1, 4), 6: (2, 10)}
MIN_FAULT_LEN = 120
MAX_FAULT_LEN = 400


def inject_failure(temperature, label, *, peak_low=2, peak_high=10):
    """Apply MATLAB-style thermal rise-and-fall to samples where label==1.

    Operates on the full sequence array; only the masked (fault) portion is modified.
    Uses the global numpy RNG — call np.random.seed() before synthesizing for reproducibility.
    """
    temperature = np.asarray(temperature, dtype=float).copy()
    label = np.asarray(label)
    mask = label == 1
    tmp = temperature[mask].copy()
    n_seq = len(tmp)
    if n_seq == 0:
        return temperature
    n_rise = n_seq // 3
    if n_rise < 1:
        return temperature
    temp_start = tmp[0]
    temp_end   = tmp[-1]
    temp_high  = max(temp_start, temp_end) + np.random.randint(peak_low, peak_high + 1)

    # Rise phase
    rise_span = int(round(temp_high - temp_start))
    if rise_span >= 1:
        step_rise = (n_rise // (rise_span + 1)) or 1
        i = 0
        for i in range(1, rise_span + 1):
            lo = (i - 1) * step_rise
            hi = min(i * step_rise, n_rise)
            if lo >= n_rise:
                break
            tmp[lo:hi] = temp_start + (i - 1)
        filled = i * step_rise
        if filled < n_rise:
            tmp[filled:n_rise] = temp_high
    else:
        tmp[:n_rise] = temp_high

    # Decay phase
    down_span = int(round(temp_high - temp_end))
    if down_span >= 1:
        step_down = (2 * n_rise // down_span) or 1
        i = 0
        for i in range(1, down_span):
            lo = n_rise + (i - 1) * step_down
            hi = min(n_rise + i * step_down, n_seq)
            if lo >= n_seq:
                break
            tmp[lo:hi] = temp_high - i
        filled = n_rise + i * step_down
        if filled < n_seq:
            tmp[filled:] = temp_end
    else:
        tmp[n_rise:] = temp_end

    temperature[mask] = tmp
    return temperature


def make_injected_sequence(seq_df, motor_id, start, length, new_id, *, peak_low=2, peak_high=10):
    """Clone a normal sequence, place a fault window at [start, start+length), and inject."""
    out = seq_df.copy().reset_index(drop=True)
    n   = len(out)
    start = max(0, min(start, n - 1))
    end   = min(start + length, n)
    if end - start < 3:
        return pd.DataFrame()

    lab_col  = f'data_motor_{motor_id}_label'
    temp_col = f'data_motor_{motor_id}_temperature'

    label = np.zeros(n, dtype=int)
    label[start:end] = 1
    out[temp_col] = inject_failure(out[temp_col].to_numpy(), label,
                                   peak_low=peak_low, peak_high=peak_high)
    out[lab_col] = label

    # Void other motors' labels so the synthetic row doesn't interfere with their training
    for j in range(1, 7):
        if j != motor_id:
            out[f'data_motor_{j}_label'] = np.nan

    out['test_condition'] = new_id
    return out


def synthesize_for_motor(train_df, motor_id, *, n_sequences,
                          peak_low=2, peak_high=10,
                          min_len=MIN_FAULT_LEN, max_len=MAX_FAULT_LEN):
    """Generate n_sequences synthetic fault sequences for motor_id from normal sequences."""
    lab_col  = f'data_motor_{motor_id}_label'
    temp_col = f'data_motor_{motor_id}_temperature'
    pos_col  = f'data_motor_{motor_id}_position'

    normal_seqs = [seq for seq, g in train_df.groupby('test_condition', sort=False)
                   if (g[lab_col] == 1).sum() == 0]
    if not normal_seqs:
        return pd.DataFrame()

    frames, made, attempts = [], 0, 0
    max_attempts = n_sequences * 20

    while made < n_sequences and attempts < max_attempts:
        attempts += 1
        src_seq = np.random.choice(normal_seqs)
        seq_df  = train_df[train_df['test_condition'] == src_seq]
        n = len(seq_df)

        effective_max = min(max_len, n - 10)
        if effective_max < min_len:
            continue  # sequence too short for a meaningful fault window

        fault_len = np.random.randint(min_len, effective_max + 1)
        start     = np.random.randint(0, max(1, n - fault_len))
        new_id    = f'{src_seq}_synth_m{motor_id}_{made}'

        frame = make_injected_sequence(seq_df, motor_id, start, fault_len, new_id,
                                        peak_low=peak_low, peak_high=peak_high)
        if len(frame) == 0:
            continue

        # Recompute derived features for the synthetic sequence
        t = frame[temp_col]
        if pos_col in frame.columns:
            frame[f'data_motor_{motor_id}_position_diff'] = frame[pos_col].diff().fillna(0)
        frame[f'data_motor_{motor_id}_temperature_diff'] = t.diff(20).fillna(0)
        for w in [5, 20]:
            frame[f'data_motor_{motor_id}_temperature_roll_mean_{w}'] = t.rolling(w, min_periods=1).mean()
            frame[f'data_motor_{motor_id}_temperature_roll_max_{w}']  = t.rolling(w, min_periods=1).max()
            frame[f'data_motor_{motor_id}_temperature_roll_std_{w}']  = t.rolling(w, min_periods=1).std().fillna(0)

        frames.append(frame)
        made += 1

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def crop_normals_near_overheating(X_train, y_train, multiplier=3):
    pos_mask = y_train == 1
    num_pos  = pos_mask.sum()
    if num_pos == 0:
        return X_train, y_train
    target_neg = num_pos * multiplier
    neg_mask   = y_train == 0
    if neg_mask.sum() <= target_neg:
        return X_train, y_train
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_train)
    centroid_1 = X_scaled[pos_mask].mean(axis=0)
    dists      = np.linalg.norm(X_scaled[neg_mask] - centroid_1, axis=1)
    closest    = np.argsort(dists)[:target_neg]
    X_balanced = pd.concat([X_train[pos_mask], X_train[neg_mask].iloc[closest]], ignore_index=True)
    y_balanced = pd.concat([y_train[pos_mask], y_train[neg_mask].iloc[closest]], ignore_index=True)
    idx = np.random.permutation(len(X_balanced))
    return X_balanced.iloc[idx], y_balanced.iloc[idx]


def pre_processing(df: pd.DataFrame):
    if len(df) == 0:
        return
    # Remove outliers
    df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
    df['temperature'] = df['temperature'].where(df['temperature'] >= 0,   np.nan)
    df['temperature'] = df['temperature'].ffill()

    df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
    df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
    df['voltage'] = df['voltage'].ffill()

    df['position'] = df['position'].where(df['position'] >= 0,    np.nan)
    df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
    df['position'] = df['position'].ffill()

    # Relative to first sample
    for col in ['temperature', 'voltage', 'position']:
        df[col] -= df[col].iloc[0]

    # Difference features
    df['temperature_diff'] = df['temperature'].diff(20)
    df['voltage_diff']     = df['voltage'].diff(20)
    df['position_diff']    = df['position'].diff(20)

    # Rolling features
    df['temperature_roll_mean_5']  = df['temperature'].rolling(5,  min_periods=1).mean()
    df['temperature_roll_max_5']   = df['temperature'].rolling(5,  min_periods=1).max()
    df['temperature_roll_std_5']   = df['temperature'].rolling(5,  min_periods=1).std().fillna(0)
    df['temperature_roll_mean_20'] = df['temperature'].rolling(20, min_periods=1).mean()
    df['temperature_roll_max_20']  = df['temperature'].rolling(20, min_periods=1).max()
    df['temperature_roll_std_20']  = df['temperature'].rolling(20, min_periods=1).std().fillna(0)

    df.fillna(0, inplace=True)


def main():
    parser = argparse.ArgumentParser(description="Motor overheating prediction (v2)")
    parser.add_argument('--exclude-motor', type=int, choices=range(1, 7), default=3,
                        help="Set this motor's predictions to -1 (Kaggle probe submission). "
                             "Hardcoded default: 3 (always skip motor 3 unless overridden).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    rolling_feats = ['temperature_roll_mean_5', 'temperature_roll_max_5', 'temperature_roll_std_5',
                     'temperature_roll_mean_20', 'temperature_roll_max_20', 'temperature_roll_std_20']
    desc_feats    = ['desc_transfer', 'desc_not_moving', 'desc_turn_motor', 'desc_chute_cube']

    MOTOR_FEATURES = {
        1: ['temperature', 'position', 'voltage', 'temperature_diff'] + rolling_feats + desc_feats,
        2: ['temperature', 'position', 'voltage'] + rolling_feats + desc_feats,
        3: ['position', 'temperature_diff', 'voltage', 'temperature'] + rolling_feats + desc_feats,
        4: ['position', 'temperature', 'voltage'] + rolling_feats + desc_feats,
        5: ['position', 'voltage', 'temperature', 'temperature_diff'] + rolling_feats + desc_feats,
        6: ['position', 'temperature', 'temperature_diff', 'position_diff'] + rolling_feats + desc_feats,
    }

    train_path = os.path.join(_PROJECT_ROOT, 'data', 'training_data') + '/'
    test_path  = os.path.join(_PROJECT_ROOT, 'data', 'testing_data')  + '/'

    t0_total = time.time()
    print("=" * 60)
    print("  MOTOR OVERHEATING PREDICTION — TDS INDUSTRY  (v2)")
    print("=" * 60)
    print(f"  Random seed : {args.seed}")
    print(f"  Injection   : random window ({MIN_FAULT_LEN}–{MAX_FAULT_LEN} samples), fixed counts per motor")

    # ── [1/4] Training data ───────────────────────────────────────────────────
    print("\n[1/4] Loading and preprocessing training data...")
    t0 = time.time()
    train_df = read_all_test_data_from_path(train_path, pre_processing, is_plot=False)
    print(f"  Base training set: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols  ({time.time()-t0:.1f}s)")

    additional_dfs  = []
    additional_base = os.path.join(_PROJECT_ROOT, 'data', 'additional_data')
    groups = [
        'additional_data_20240524_group_6',
        'additional_training_data_group_1',
        'additional_training_data_group_7',
    ]
    for group in groups:
        group_path     = os.path.join(additional_base, group)
        xlsx_path      = os.path.join(group_path, 'Test conditions.xlsx')
        copy_xlsx_path = os.path.join(group_path, 'Test conditions copy.xlsx')
        if not os.path.exists(xlsx_path) and os.path.exists(copy_xlsx_path):
            import shutil
            try:
                shutil.copy(copy_xlsx_path, xlsx_path)
            except Exception as e:
                print(f"  Failed to copy Test conditions: {e}")

        if os.path.exists(group_path):
            print(f"  Loading {group}...")
            import shutil
            temp_group_dir = os.path.join(additional_base, f"{group}_temp_valid")
            if os.path.exists(temp_group_dir):
                try:
                    shutil.rmtree(temp_group_dir)
                except Exception:
                    pass
            os.makedirs(temp_group_dir, exist_ok=True)
            if os.path.exists(xlsx_path):
                shutil.copy(xlsx_path, os.path.join(temp_group_dir, 'Test conditions.xlsx'))

            subdirs      = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
            copied_count = 0
            for subdir in subdirs:
                sub_path  = os.path.join(group_path, subdir)
                csv_files = [f for f in os.listdir(sub_path) if f.endswith('.csv')]
                is_valid  = True
                for csv_file in csv_files:
                    try:
                        with open(os.path.join(sub_path, csv_file), 'r') as f:
                            lines = [l.strip() for l in f if l.strip()]
                        if len(lines) <= 1:
                            is_valid = False
                            break
                    except Exception:
                        is_valid = False
                        break
                if is_valid and csv_files:
                    shutil.copytree(sub_path, os.path.join(temp_group_dir, subdir), dirs_exist_ok=True)
                    copied_count += 1

            print(f"    {copied_count}/{len(subdirs)} valid subdirectories")
            if copied_count > 0:
                try:
                    group_df = read_all_test_data_from_path(temp_group_dir + '/', pre_processing, is_plot=False)
                    additional_dfs.append(group_df)
                except Exception as e:
                    print(f"  Failed to load {group}: {e}")
            try:
                shutil.rmtree(temp_group_dir)
            except Exception:
                pass

    if additional_dfs:
        train_df = pd.concat([train_df] + additional_dfs, ignore_index=True)
        print(f"  Combined dataset: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols")

    # ── [2/4] Test data ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading and preprocessing test data...")
    t0 = time.time()
    test_df = read_all_test_data_from_path(test_path, pre_processing, is_plot=False)
    print(f"  Test set: {test_df.shape[0]:,} rows  ({time.time()-t0:.1f}s)")

    # ── [3/4] Movement descriptions ──────────────────────────────────────────
    print(f"\n[3/4] Loading test conditions to extract movement descriptions...")
    import glob
    tc_files  = glob.glob(os.path.join(_PROJECT_ROOT, 'data') + '/**/Test conditions*.xlsx', recursive=True)
    tc_files += glob.glob(additional_base + '/**/Test conditions*.xlsx', recursive=True)

    tc_dfs = []
    for f in set(tc_files):
        try:
            tc_dfs.append(pd.read_excel(f))
        except Exception:
            pass

    if tc_dfs:
        all_tc = pd.concat(tc_dfs, ignore_index=True)
        if 'Description' in all_tc.columns and 'Test id' in all_tc.columns:
            all_tc['Description'] = all_tc['Description'].fillna('').astype(str).str.lower()
            desc_map = all_tc.set_index('Test id')['Description'].to_dict()
            print(f"  Mapped descriptions for {len(desc_map)} test conditions.")

            for label, df in [('train', train_df), ('test', test_df)]:
                df['raw_desc']        = df['test_condition'].map(desc_map).fillna('')
                df['desc_transfer']   = df['raw_desc'].str.contains('transfer').astype(int)
                df['desc_not_moving'] = df['raw_desc'].str.contains('not moving').astype(int)
                df['desc_turn_motor'] = df['raw_desc'].str.contains('turn motor').astype(int)
                df['desc_chute_cube'] = df['raw_desc'].str.contains('chute cube').astype(int)

                # Build all 24 per-motor desc columns at once (avoids DataFrame fragmentation)
                new_cols = pd.DataFrame({
                    f'data_motor_{m_id}_{feat}': df[feat].values
                    for m_id in range(1, 7)
                    for feat in ['desc_transfer', 'desc_not_moving', 'desc_turn_motor', 'desc_chute_cube']
                }, index=df.index)
                merged = pd.concat([df, new_cols], axis=1)
                if label == 'train':
                    train_df = merged
                else:
                    test_df = merged

    # ── [4/4] Train per-motor models ─────────────────────────────────────────
    print(f"\n[4/4] Training individual models per motor...")
    models = {}

    for motor_id in range(1, 7):
        print(f"\n{'─'*50}")
        print(f"  Motor {motor_id}/6")

        features = [f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]]
        missing_feats = [f for f in features if f not in train_df.columns]
        if missing_feats:
            print(f"  SKIP: missing columns: {missing_feats}")
            continue

        label_col = f'data_motor_{motor_id}_label'
        if label_col not in train_df.columns:
            print(f"  SKIP: no label column found.")
            continue

        motor_df  = train_df.dropna(subset=[label_col]).copy()
        sequences = motor_df['test_condition'].unique()
        failure_seqs = [s for s in sequences if (motor_df[motor_df['test_condition'] == s][label_col] == 1).any()]
        normal_seqs  = [s for s in sequences if s not in failure_seqs]

        # 80/20 sequence split stratified by fault presence
        if len(failure_seqs) > 1:
            n_val = max(1, int(0.2 * len(failure_seqs)))
            train_fail_seqs, val_fail_seqs = train_test_split(failure_seqs, test_size=n_val, random_state=42)
        else:
            train_fail_seqs, val_fail_seqs = failure_seqs, []

        if len(normal_seqs) > 1:
            n_val = max(1, int(0.2 * len(normal_seqs)))
            train_norm_seqs, val_norm_seqs = train_test_split(normal_seqs, test_size=n_val, random_state=42)
        else:
            train_norm_seqs, val_norm_seqs = normal_seqs, []

        train_seqs = train_fail_seqs + train_norm_seqs
        val_seqs   = val_fail_seqs   + val_norm_seqs

        raw_motor_train_df = motor_df[motor_df['test_condition'].isin(train_seqs)].copy()
        raw_motor_val_df   = motor_df[motor_df['test_condition'].isin(val_seqs)].copy()

        print(f"  Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}")
        print(f"  Failure seqs in train: {len(train_fail_seqs)}  |  normal: {len(train_norm_seqs)}")

        # Synthesize fault sequences: random window, fixed count, motor-specific peak
        n_inj              = N_INJECT[motor_id]
        peak_low, peak_high = PEAK_RANGE[motor_id]
        print(f"  Synthesizing {n_inj} fault sequences "
              f"(random window {MIN_FAULT_LEN}–{MAX_FAULT_LEN} samples, peak +{peak_low}–{peak_high}°C)...")
        synth_df = synthesize_for_motor(raw_motor_train_df, motor_id,
                                         n_sequences=n_inj,
                                         peak_low=peak_low, peak_high=peak_high)
        if not synth_df.empty:
            raw_motor_train_df = pd.concat([raw_motor_train_df, synth_df], ignore_index=True)

        X_train = raw_motor_train_df[features]
        y_train = raw_motor_train_df[label_col]

        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        print(f"  Augmented train set — normal: {num_neg:,}  failure: {num_pos:,}  ratio: {num_pos/(num_neg+num_pos):.1%}")

        X_val, y_val = None, None
        if val_seqs:
            X_val = raw_motor_val_df[features]
            y_val = raw_motor_val_df[label_col]

        best_model      = None
        best_val_score  = -1.0
        best_params_info = {}
        best_X_train    = None
        best_y_train    = None

        param_grid = {
            'learning_rate':    [0.02, 0.05],
            'max_iter':         [100, 150],
            'max_depth':        [3, 4, 5],
            'min_samples_leaf': [200, 600],
        }
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        n_combos = len(param_combinations)

        thresholds     = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        closing_options = [False, True]

        sample_weight = np.ones(len(y_train))
        sample_weight[y_train == 1] = 2.0

        print(f"  Grid search: {n_combos} combos × {len(thresholds)} thresholds × 2 closing options")
        t0_grid = time.time()

        for i, params in enumerate(param_combinations, 1):
            print(f"  [{i:2d}/{n_combos}] lr={params['learning_rate']}  depth={params['max_depth']}  "
                  f"iter={params['max_iter']}  min_leaf={params['min_samples_leaf']} ...", end='', flush=True)
            t_fit = time.time()
            model = HistGradientBoostingClassifier(random_state=42, **params)
            model.fit(X_train, y_train, sample_weight=sample_weight)

            if X_val is not None:
                y_prob_val      = model.predict_proba(X_val)[:, 1]
                best_score_this = 0.0

                for t in thresholds:
                    y_pred_base = (y_prob_val >= t).astype(int)
                    for use_closing in closing_options:
                        y_pred_val = (binary_closing(y_pred_base, structure=np.ones(5)).astype(int)
                                      if use_closing else y_pred_base)
                        score = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
                        if score > best_val_score:
                            best_val_score  = score
                            best_model      = model
                            best_params_info = {'model_params': params, 'threshold': t, 'use_closing': use_closing}
                            best_X_train    = X_train
                            best_y_train    = y_train
                        if score > best_score_this:
                            best_score_this = score

                print(f"  best F1={best_score_this:.3f}  ({time.time()-t_fit:.1f}s)")
            else:
                print(f"  (no val set, using defaults)  ({time.time()-t_fit:.1f}s)")
                best_model       = model
                best_params_info = {'model_params': params, 'threshold': 0.5, 'use_closing': True}
                best_X_train     = X_train
                best_y_train     = y_train
                break  # only need one combo if there is no validation set

        print(f"  Grid search done in {time.time()-t0_grid:.1f}s")

        model          = best_model
        best_threshold = best_params_info.get('threshold', 0.5)
        best_closing   = best_params_info.get('use_closing', True)
        models[motor_id] = {'model': model, 'threshold': best_threshold, 'use_closing': best_closing}

        if X_val is not None:
            print(f"\n  Best val F1 (class 1): {best_val_score:.4f}")
            bp = best_params_info['model_params']
            print(f"  Best params: lr={bp['learning_rate']}  depth={bp['max_depth']}  "
                  f"iter={bp['max_iter']}  min_leaf={bp['min_samples_leaf']}  "
                  f"threshold={best_threshold}  closing={best_closing}")

        y_prob_train = model.predict_proba(best_X_train)[:, 1]
        y_pred_train = (y_prob_train >= best_threshold).astype(int)
        if best_closing:
            y_pred_train = binary_closing(y_pred_train, structure=np.ones(5)).astype(int)
        print("\n  Train report:")
        print(classification_report(best_y_train, y_pred_train, zero_division=0))

        if X_val is not None:
            y_prob_val = model.predict_proba(X_val)[:, 1]
            y_pred_val = (y_prob_val >= best_threshold).astype(int)
            if best_closing:
                y_pred_val = binary_closing(y_pred_val, structure=np.ones(5)).astype(int)
            print("  Validation report:")
            print(classification_report(y_val, y_pred_val, zero_division=0))

    # ── Submission ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating predictions...")
    print(f"{'='*60}")
    sample_sub_path = os.path.join(_PROJECT_ROOT, 'sample_submission.csv')
    submission_df   = pd.read_csv(sample_sub_path)
    final_sub       = submission_df.copy()

    for test_id in submission_df['test_condition'].unique():
        sub_mask       = submission_df['test_condition'] == test_id
        motor_test_data = test_df[test_df['test_condition'] == test_id].sort_values('time')
        expected_len   = int(sub_mask.sum())

        if len(motor_test_data) == 0:
            print(f"  WARNING: no data found for test_condition={test_id}, filling 0s.")
            for mid in range(1, 7):
                final_sub.loc[sub_mask, f'data_motor_{mid}_label'] = 0
            continue

        for motor_id in range(1, 7):
            features = [f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]]
            if motor_id not in models or not all(f in motor_test_data.columns for f in features):
                final_sub.loc[sub_mask, f'data_motor_{motor_id}_label'] = 0
                continue

            model_info = models[motor_id]
            y_prob     = model_info['model'].predict_proba(motor_test_data[features])[:, 1]
            y_pred     = (y_prob >= model_info['threshold']).astype(int)
            if model_info['use_closing']:
                y_pred = binary_closing(y_pred, structure=np.ones(3)).astype(int)

            if len(y_pred) != expected_len:
                print(f"  Length mismatch {test_id} motor {motor_id}: {len(y_pred)} vs {expected_len}")
                y_pred = (y_pred[:expected_len] if len(y_pred) > expected_len
                          else np.pad(y_pred, (0, expected_len - len(y_pred)), 'constant'))

            final_sub.loc[sub_mask, f'data_motor_{motor_id}_label'] = y_pred

    for motor_id in range(1, 7):
        final_sub[f'data_motor_{motor_id}_label'] = final_sub[f'data_motor_{motor_id}_label'].astype(int)

    submissions_dir = os.path.join(_PROJECT_ROOT, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    if args.exclude_motor is not None:
        print(f"\n  Kaggle probe: motor {args.exclude_motor} -> -1  (submission will fail on this motor)")
        final_sub[f'data_motor_{args.exclude_motor}_label'] = -1
        out_path = os.path.join(submissions_dir, f'motor_excluded_{args.exclude_motor}_submission.csv')
    else:
        out_path = os.path.join(submissions_dir, 'motor_submission.csv')

    final_sub.to_csv(out_path, index=False)

    total_time = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"  Done in {total_time/60:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"  Shape : {final_sub.shape}")
    print()
    print(f"  {'Motor':<8} {'Algorithm':<12} {'0 (normal)':>12} {'1 (fault)':>10} {'% fault':>9}")
    print(f"  {'-'*56}")
    for mid in range(1, 7):
        col   = f'data_motor_{mid}_label'
        algo  = models.get(mid, {}).get('type', 'HistGB')
        c0    = (final_sub[col] == 0).sum()
        c1    = (final_sub[col] == 1).sum()
        total = c0 + c1
        pct   = 100 * c1 / total if total > 0 else 0
        excl  = '  <- EXCLUDED (-1)' if mid == args.exclude_motor else ''
        print(f"  {mid:<8} {algo:<12} {c0:>12,} {c1:>10,} {pct:>8.1f}%{excl}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
