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


def inject_failure_sequence_matlab(temperature_series):
    import math
    tmp_temp = temperature_series.values.copy()
    n_seq = len(tmp_temp)
    if n_seq == 0:
        return temperature_series
        
    n_rise = math.floor(n_seq / 3)
    if n_rise == 0:
        n_rise = 1
        
    temp_start = tmp_temp[0]
    temp_end = tmp_temp[-1]
    
    temp_high = max(temp_start, temp_end) + np.random.randint(2, 11)
    
    # Rise
    step_size_rise = math.floor(n_rise / max(1, (temp_high - temp_start + 1)))
    if step_size_rise == 0:
        step_size_rise = 1
        
    for i in range(1, int(temp_high - temp_start) + 1):
        start_idx = (i - 1) * step_size_rise
        end_idx = i * step_size_rise
        if start_idx < len(tmp_temp):
            tmp_temp[start_idx:end_idx] = temp_start + i - 1
            
    # Cap remaining rise
    last_rise_idx = int((temp_high - temp_start) * step_size_rise)
    if last_rise_idx < n_rise:
        tmp_temp[last_rise_idx:n_rise] = temp_high
        
    # Decrease
    step_size_down = math.floor((2 * n_rise) / max(1, (temp_high - temp_end)))
    if step_size_down == 0:
        step_size_down = 1
        
    for i in range(1, int(temp_high - temp_end)):
        start_idx = n_rise + (i - 1) * step_size_down
        end_idx = n_rise + i * step_size_down
        if start_idx < len(tmp_temp):
            tmp_temp[start_idx:end_idx] = temp_high - i
            
    last_down_idx = n_rise + int((temp_high - temp_end - 1) * step_size_down)
    if last_down_idx < len(tmp_temp):
        tmp_temp[last_down_idx:] = temp_end
        
    return pd.Series(tmp_temp, index=temperature_series.index)

def balance_dataset_sequences(train_df_subset, motor_id, target_ratio=0.5):
    # This operates on the full training dataframe for a specific motor split
    # and synthesizes new sequences by duplicating normal sequences and injecting
    label_col = f'data_motor_{motor_id}_label'
    temp_col = f'data_motor_{motor_id}_temperature'
    
    seqs = train_df_subset['test_condition'].unique()
    num_neg = (train_df_subset[label_col] == 0).sum()
    num_pos = (train_df_subset[label_col] == 1).sum()
    
    if num_pos == 0 or target_ratio >= 1.0:
        return train_df_subset
        
    total_pos_target = int(target_ratio * num_neg / (1 - target_ratio))
    needed = total_pos_target - num_pos
    if needed <= 0:
        return train_df_subset
        
    # Find normal sequences
    normal_seqs = []
    for s in seqs:
        if (train_df_subset[train_df_subset['test_condition'] == s][label_col] == 1).sum() == 0:
            normal_seqs.append(s)
            
    if not normal_seqs:
        return train_df_subset
        
    synthetic_dfs = []
    current_synthetic_pos = 0
    seq_counter = 0
    
    while current_synthetic_pos < needed:
        src_seq = np.random.choice(normal_seqs)
        seq_df = train_df_subset[train_df_subset['test_condition'] == src_seq].copy()
        
        # Mark final 20% as failure
        failure_length = max(5, int(len(seq_df) * 0.2))
        failure_indices = seq_df.index[-failure_length:]
        seq_df.loc[failure_indices, label_col] = 1
        
        # Apply MATLAB injection to temperature
        original_temp = seq_df.loc[failure_indices, temp_col]
        synthetic_temp = inject_failure_sequence_matlab(original_temp)
        seq_df.loc[failure_indices, temp_col] = synthetic_temp
        
        # Add slight voltage noise so it isn't completely identical
        volt_col = f'data_motor_{motor_id}_voltage'
        if volt_col in seq_df.columns:
            seq_df.loc[failure_indices, volt_col] += np.random.normal(0, 0.05, size=failure_length)
            
        seq_df['test_condition'] = f"{src_seq}_synth_{seq_counter}"
        seq_counter += 1
        
        synthetic_dfs.append(seq_df)
        current_synthetic_pos += failure_length
        
    if not synthetic_dfs:
        return train_df_subset
        
    combined_synth = pd.concat(synthetic_dfs, ignore_index=True)
    
    pos_col = f'data_motor_{motor_id}_position'
    if pos_col in combined_synth.columns:
        combined_synth[f'data_motor_{motor_id}_position_diff'] = combined_synth.groupby('test_condition')[pos_col].diff().fillna(0)
    combined_synth[f'data_motor_{motor_id}_temperature_diff'] = combined_synth.groupby('test_condition')[temp_col].diff().fillna(0)
    
    for w in [5, 20]:
        combined_synth[f'data_motor_{motor_id}_temperature_roll_mean_{w}'] = combined_synth.groupby('test_condition')[temp_col].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
        combined_synth[f'data_motor_{motor_id}_temperature_roll_max_{w}'] = combined_synth.groupby('test_condition')[temp_col].rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True).fillna(0)
        combined_synth[f'data_motor_{motor_id}_temperature_roll_std_{w}'] = combined_synth.groupby('test_condition')[temp_col].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        
    return pd.concat([train_df_subset, combined_synth], ignore_index=True)

def crop_normals_near_overheating(X_train, y_train, multiplier=3):
    from sklearn.preprocessing import StandardScaler
    pos_mask = y_train == 1
    num_pos = pos_mask.sum()
    
    if num_pos == 0:
        return X_train, y_train
        
    target_neg = num_pos * multiplier
    neg_mask = y_train == 0
    
    if neg_mask.sum() <= target_neg:
        return X_train, y_train
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    centroid_1 = X_scaled[pos_mask].mean(axis=0)
    
    X_neg_scaled = X_scaled[neg_mask]
    dists = np.linalg.norm(X_neg_scaled - centroid_1, axis=1)
    
    closest_indices = np.argsort(dists)[:target_neg]
    
    X_neg_cropped = X_train[neg_mask].iloc[closest_indices]
    y_neg_cropped = y_train[neg_mask].iloc[closest_indices]
    
    X_pos = X_train[pos_mask]
    y_pos = y_train[pos_mask]
    
    X_balanced = pd.concat([X_pos, X_neg_cropped], ignore_index=True)
    y_balanced = pd.concat([y_pos, y_neg_cropped], ignore_index=True)
    
    idx = np.random.permutation(len(X_balanced))
    return X_balanced.iloc[idx], y_balanced.iloc[idx]

def pre_processing(df: pd.DataFrame):
    if len(df) == 0:
        return
    ''' # Description
    Preprocess the data:
    - remove outliers
    - add new features about the difference between the current and previous n data point.
    '''
    # Remove outliers
    df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
    df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
    df['temperature'] = df['temperature'].ffill()        

    df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
    df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
    df['voltage'] = df['voltage'].ffill()        

    df['position'] = df['position'].where(df['position'] >= 0, np.nan)
    df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
    df['position'] = df['position'].ffill()

    # Calculate differences
    n_int = 20
    # Transform features relative to the first data point
    df['temperature'] = df['temperature'] - df['temperature'].iloc[0]
    df['voltage'] = df['voltage'] - df['voltage'].iloc[0]
    df['position'] = df['position'] - df['position'].iloc[0]

    # Difference features
    df['temperature_diff'] = df['temperature'].diff(n_int)
    df['voltage_diff'] = df['voltage'].diff(n_int)
    df['position_diff'] = df['position'].diff(n_int)

    # Rolling sequence features
    df['temperature_roll_mean_5'] = df['temperature'].rolling(window=5, min_periods=1).mean()
    df['temperature_roll_max_5'] = df['temperature'].rolling(window=5, min_periods=1).max()
    df['temperature_roll_std_5'] = df['temperature'].rolling(window=5, min_periods=1).std().fillna(0)
    
    df['temperature_roll_mean_20'] = df['temperature'].rolling(window=20, min_periods=1).mean()
    df['temperature_roll_max_20'] = df['temperature'].rolling(window=20, min_periods=1).max()
    df['temperature_roll_std_20'] = df['temperature'].rolling(window=20, min_periods=1).std().fillna(0)

    # Fill NaNs created by diff
    df.fillna(0, inplace=True)


def main():
    parser = argparse.ArgumentParser(description="Motor overheating prediction")
    parser.add_argument('--exclude-motor', type=int, choices=range(1, 7), default=None,
                        help="If set, outputs predictions for all motors EXCEPT this one, which is set to -1")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"  Random seed: {args.seed}")

    rolling_feats = ['temperature_roll_mean_5', 'temperature_roll_max_5', 'temperature_roll_std_5',
                     'temperature_roll_mean_20', 'temperature_roll_max_20', 'temperature_roll_std_20']
                     
    desc_feats = ['desc_transfer', 'desc_not_moving', 'desc_turn_motor', 'desc_chute_cube']
                     
    MOTOR_FEATURES = {
        1: ['temperature', 'position', 'voltage', 'temperature_diff'] + rolling_feats + desc_feats,
        2: ['temperature', 'position', 'voltage'] + rolling_feats + desc_feats,
        3: ['position', 'temperature_diff', 'voltage', 'temperature'] + rolling_feats + desc_feats,
        4: ['position', 'temperature', 'voltage'] + rolling_feats + desc_feats,
        5: ['position', 'voltage', 'temperature', 'temperature_diff'] + rolling_feats + desc_feats,
        6: ['position', 'temperature', 'temperature_diff', 'position_diff'] + rolling_feats + desc_feats
    }
    train_path = os.path.join(_PROJECT_ROOT, 'data', 'training_data') + '/'
    test_path = os.path.join(_PROJECT_ROOT, 'data', 'testing_data') + '/'
    
    t0_total = time.time()
    print("=" * 60)
    print("  MOTOR OVERHEATING PREDICTION — TDS INDUSTRY")
    print("=" * 60)

    print("\n[1/4] Loading and preprocessing training data...")
    t0 = time.time()
    train_df = read_all_test_data_from_path(train_path, pre_processing, is_plot=False)
    print(f"  Base training set: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols  ({time.time()-t0:.1f}s)")

    # Load additional data groups
    additional_dfs = []
    additional_base = os.path.join(_PROJECT_ROOT, 'data', 'additional_data')
    groups = [
        'additional_data_20240524_group_6',
        'additional_training_data_group_1',
        'additional_training_data_group_7'
    ]
    for group in groups:
        group_path = os.path.join(additional_base, group)
        # Ensure Test conditions.xlsx exists (copy for group 7 if missing)
        xlsx_path = os.path.join(group_path, 'Test conditions.xlsx')
        copy_xlsx_path = os.path.join(group_path, 'Test conditions copy.xlsx')
        if not os.path.exists(xlsx_path) and os.path.exists(copy_xlsx_path):
            import shutil
            try:
                shutil.copy(copy_xlsx_path, xlsx_path)
            except Exception as e:
                print(f"Failed to copy Test conditions: {e}")
                
        if os.path.exists(group_path):
            print(f"  Loading {group}...")
            # Filter out empty/invalid subdirectories to prevent "cannot set a frame with no defined index and a scalar"
            import shutil
            temp_group_dir = os.path.join(additional_base, f"{group}_temp_valid")
            if os.path.exists(temp_group_dir):
                try:
                    shutil.rmtree(temp_group_dir)
                except Exception:
                    pass
            os.makedirs(temp_group_dir, exist_ok=True)
            
            # Copy Test conditions.xlsx
            if os.path.exists(xlsx_path):
                shutil.copy(xlsx_path, os.path.join(temp_group_dir, 'Test conditions.xlsx'))
            
            subdirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
            copied_count = 0
            for subdir in subdirs:
                sub_path = os.path.join(group_path, subdir)
                csv_files = [f for f in os.listdir(sub_path) if f.endswith('.csv')]
                is_valid = True
                for csv_file in csv_files:
                    csv_path = os.path.join(sub_path, csv_file)
                    try:
                        with open(csv_path, 'r') as f:
                            lines = [line.strip() for line in f if line.strip()]
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
                    # Add trailing slash as expected by read_all_test_data_from_path
                    group_df = read_all_test_data_from_path(temp_group_dir + '/', pre_processing, is_plot=False)
                    additional_dfs.append(group_df)
                except Exception as e:
                    print(f"Failed to load {group}: {e}")
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_group_dir)
            except Exception:
                pass
                
    if additional_dfs:
        combined_additional = pd.concat(additional_dfs, ignore_index=True)
        train_df = pd.concat([train_df, combined_additional], ignore_index=True)
        print(f"  Combined dataset: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols")

    print(f"\n[2/4] Loading and preprocessing test data...")
    t0 = time.time()
    test_df = read_all_test_data_from_path(test_path, pre_processing, is_plot=False)
    print(f"  Test set: {test_df.shape[0]:,} rows  ({time.time()-t0:.1f}s)")
    
    print(f"\n[3/4] Loading test conditions to extract movement descriptions...")
    import glob
    tc_files = glob.glob(os.path.join(_PROJECT_ROOT, 'data') + '/**/Test conditions*.xlsx', recursive=True)
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
                df['raw_desc'] = df['test_condition'].map(desc_map).fillna('')
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
                    
    print(f"\n[4/4] Training individual models per motor...")
    models = {}

    for motor_id in range(1, 7):
        print(f"\n{'─'*50}")
        print(f"  Motor {motor_id}/6")
        # Define features for this motor
        features = [
            f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]
        ]
        
        # Check if all features exist
        missing_feats = [f for f in features if f not in train_df.columns]
        if missing_feats:
            print(f"  SKIP: missing columns: {missing_feats}")
            continue

        label_col = f'data_motor_{motor_id}_label'
        if label_col not in train_df.columns:
            print(f"  SKIP: no label column found.")
            continue
            
        motor_df = train_df.dropna(subset=[label_col]).copy()
        
        # Split sequences 80:20 (stratified by presence of failure)
        sequences = motor_df['test_condition'].unique()
        failure_seqs = []
        normal_seqs = []
        for seq in sequences:
            seq_df = motor_df[motor_df['test_condition'] == seq]
            if (seq_df[label_col] == 1).any():
                failure_seqs.append(seq)
            else:
                normal_seqs.append(seq)
                
        # Handle cases with very few sequences
        if len(failure_seqs) > 1:
            n_val = max(1, int(0.2 * len(failure_seqs)))
            train_fail_seqs, val_fail_seqs = train_test_split(failure_seqs, test_size=n_val, random_state=42)
        else:
            train_fail_seqs = failure_seqs
            val_fail_seqs = []
            
        if len(normal_seqs) > 1:
            n_val = max(1, int(0.2 * len(normal_seqs)))
            train_norm_seqs, val_norm_seqs = train_test_split(normal_seqs, test_size=n_val, random_state=42)
        else:
            train_norm_seqs = normal_seqs
            val_norm_seqs = []
            
        train_seqs = train_fail_seqs + train_norm_seqs
        val_seqs = val_fail_seqs + val_norm_seqs
        
        raw_motor_train_df = motor_df[motor_df['test_condition'].isin(train_seqs)].copy()
        raw_motor_val_df = motor_df[motor_df['test_condition'].isin(val_seqs)].copy()
        
        print(f"  Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}")
        print(f"  Failure seqs in train: {len(train_fail_seqs)}  |  normal: {len(train_norm_seqs)}")

        print("  Synthesizing failure sequences to 20% ratio...")
        raw_motor_train_df = balance_dataset_sequences(raw_motor_train_df, motor_id, target_ratio=0.2)

        X_train = raw_motor_train_df[features]
        y_train = raw_motor_train_df[label_col]

        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        print(f"  Balanced train set — normal: {num_neg:,}  failure: {num_pos:,}  ratio: {num_pos/(num_neg+num_pos):.1%}")
        
        X_val = None
        y_val = None
        if len(val_seqs) > 0:
            X_val = raw_motor_val_df[features]
            y_val = raw_motor_val_df[label_col]
            
        best_model = None
        best_val_score = -1.0
        best_params_info = {}
        best_X_train = None
        best_y_train = None
        
        param_grid = {
            'learning_rate': [0.02, 0.05],
            'max_iter': [100, 150],
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [200, 600]
        }
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        n_combos = len(param_combinations)

        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
                y_prob_val = model.predict_proba(X_val)[:, 1]
                best_score_this = 0.0

                for t in thresholds:
                    y_pred_base = (y_prob_val >= t).astype(int)

                    for use_closing in closing_options:
                        if use_closing:
                            y_pred_val = binary_closing(y_pred_base, structure=np.ones(5)).astype(int)
                        else:
                            y_pred_val = y_pred_base

                        score = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
                        if score > best_val_score:
                            best_val_score = score
                            best_model = model
                            best_params_info = {
                                'model_params': params, 'threshold': t,
                                'use_closing': use_closing
                            }
                            best_X_train = X_train
                            best_y_train = y_train
                        if score > best_score_this:
                            best_score_this = score

                print(f"  best F1={best_score_this:.3f}  ({time.time()-t_fit:.1f}s)")
            else:
                print(f"  (no val set, using defaults)  ({time.time()-t_fit:.1f}s)")
                best_model = model
                best_params_info = {
                    'model_params': params, 'threshold': 0.5,
                    'use_closing': True
                }
                best_X_train = X_train
                best_y_train = y_train
                break  # only need one if no val

        print(f"  Grid search done in {time.time()-t0_grid:.1f}s")
                
        model = best_model
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
    
    print(f"\n{'='*60}")
    print("  Generating predictions...")
    print(f"{'='*60}")
    sample_sub_path = os.path.join(_PROJECT_ROOT, 'sample_submission.csv')
    submission_df = pd.read_csv(sample_sub_path)
    
    # Make a copy to hold our results
    final_sub = submission_df.copy()
    
    for test_id in submission_df['test_condition'].unique():
        sub_mask = submission_df['test_condition'] == test_id
        
        # Test data for this condition
        test_mask = test_df['test_condition'] == test_id
        motor_test_data = test_df[test_mask].sort_values('time')
        expected_len = sub_mask.sum()
        
        if len(motor_test_data) == 0:
            print(f"  WARNING: no data found for test_condition={test_id}, filling 0s.")
            for motor_id in range(1, 7):
                final_sub.loc[sub_mask, f'data_motor_{motor_id}_label'] = 0
            continue
            
        for motor_id in range(1, 7):
            features = [
                f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]
            ]
            
            if motor_id not in models or not all(f in motor_test_data.columns for f in features):
                final_sub.loc[sub_mask, f'data_motor_{motor_id}_label'] = 0
                continue
                
            X_test_motor = motor_test_data[features]
            model_info = models[motor_id]
            model = model_info['model']
            threshold = model_info['threshold']
            use_closing = model_info['use_closing']
            
            y_prob = model.predict_proba(X_test_motor)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            
            # Post-process dynamically based on optimal validation strategy
            if use_closing:
                y_pred = binary_closing(y_pred, structure=np.ones(3)).astype(int)
            
            if len(y_pred) != expected_len:
                print(f"Length mismatch for {test_id} motor {motor_id}! Expected {expected_len}, got {len(y_pred)}. Merging by truncation or padding.")
                if len(y_pred) > expected_len:
                    y_pred = y_pred[:expected_len]
                else:
                    y_pred = np.pad(y_pred, (0, expected_len - len(y_pred)), 'constant')
                    
            final_sub.loc[sub_mask, f'data_motor_{motor_id}_label'] = y_pred

    for motor_id in range(1, 7):
        col = f'data_motor_{motor_id}_label'
        final_sub[col] = final_sub[col].astype(int)

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
    print(f"  Shape   : {final_sub.shape}")
    print()
    print(f"  {'Motor':<8} {'Algorithm':<12} {'0 (normal)':>12} {'1 (fault)':>10} {'% fault':>9}")
    print(f"  {'-'*56}")
    for mid in range(1, 7):
        col  = f'data_motor_{mid}_label'
        algo = models.get(mid, {}).get('type', 'HistGB')
        c0   = (final_sub[col] == 0).sum()
        c1   = (final_sub[col] == 1).sum()
        total = c0 + c1
        pct  = 100 * c1 / total if total > 0 else 0
        excl = '  <- EXCLUDED (-1)' if mid == args.exclude_motor else ''
        print(f"  {mid:<8} {algo:<12} {c0:>12,} {c1:>10,} {pct:>8.1f}%{excl}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
