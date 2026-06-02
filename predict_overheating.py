import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from scipy.ndimage import binary_closing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Add utility path
sys.path.insert(1, r'C:\Users\oscar\Desktop\TDS INDUSTRY\kaggle_data_challenge')
from utility import read_all_test_data_from_path

def inject_failure(temperature_values, label_values):
    failure_mask = (label_values == 1)
    if failure_mask.sum() == 0:
        return temperature_values.copy()

    tmp_temp = temperature_values[failure_mask]
    n_seq = len(tmp_temp)
    n_rise = n_seq // 3
    n_fall = n_seq - n_rise

    temp_start = tmp_temp[0]
    temp_end   = tmp_temp[-1]
    temp_high  = max(temp_start, temp_end) + np.random.randint(2, 11)

    rise     = np.linspace(temp_start, temp_high, n_rise + 1)[:-1]
    fall     = np.linspace(temp_high,  temp_end,  n_fall)
    injected = np.concatenate([rise, fall])

    result = temperature_values.copy()
    result[failure_mask] = injected
    return result


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
    args = parser.parse_args()

    rolling_feats = ['temperature_roll_mean_5', 'temperature_roll_max_5', 'temperature_roll_std_5',
                     'temperature_roll_mean_20', 'temperature_roll_max_20', 'temperature_roll_std_20']
                     
    MOTOR_FEATURES = {
        1: ['temperature', 'position', 'voltage', 'temperature_diff'] + rolling_feats,
        2: ['temperature', 'position', 'voltage'] + rolling_feats,
        3: ['position', 'temperature_diff', 'voltage', 'temperature'] + rolling_feats,
        4: ['position', 'temperature', 'voltage'] + rolling_feats,
        5: ['position', 'voltage', 'temperature', 'temperature_diff'] + rolling_feats,
        6: ['position', 'temperature', 'temperature_diff', 'position_diff'] + rolling_feats
    }
    base_dir = r'C:\Users\oscar\Desktop\TDS INDUSTRY\kaggle_data_challenge\kaggle_data_challenge'
    train_path = os.path.join(base_dir, 'training_data/')
    test_path = os.path.join(base_dir, 'testing_data/')
    
    print("Loading and preprocessing training data...")
    train_df = read_all_test_data_from_path(train_path, pre_processing, is_plot=False)
    
    # Load additional data groups
    additional_dfs = []
    additional_base = r'C:\Users\oscar\Desktop\TDS INDUSTRY\kaggle_data_challenge\additional_data'
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
            print(f"Loading additional data from {group}...")
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
                    shutil.copytree(sub_path, os.path.join(temp_group_dir, subdir))
                    copied_count += 1
            
            print(f"Group {group}: {copied_count} out of {len(subdirs)} subdirectories are valid.")
            
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
        print(f"Combined original and additional dataset shape: {train_df.shape}")
        
    print("Loading and preprocessing testing data...")
    test_df = read_all_test_data_from_path(test_path, pre_processing, is_plot=False)
    
    print("\nTraining individual models per motor...")
    models = {}
    val_f1_per_motor = {}

    for motor_id in range(1, 7):
        print(f"\n--- Motor {motor_id} ---")
        # Define features for this motor
        features = [
            f'data_motor_{motor_id}_{feat}' for feat in MOTOR_FEATURES[motor_id]
        ]
        
        # Check if all features exist
        missing_feats = [f for f in features if f not in train_df.columns]
        if missing_feats:
            print(f"Skipping Motor {motor_id} because missing columns: {missing_feats}")
            continue

        label_col = f'data_motor_{motor_id}_label'
        if label_col not in train_df.columns:
            print(f"Skipping Motor {motor_id} because no label column.")
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

        print(f"Split: {len(train_seqs)} training sequences, {len(val_seqs)} validation sequences.")

        # Inject failures into training sequences only (val stays with real data)
        temp_col = f'data_motor_{motor_id}_temperature'
        if temp_col in raw_motor_train_df.columns:
            train_inj = raw_motor_train_df.copy()
            for seq in train_seqs:
                seq_mask = train_inj['test_condition'] == seq
                if (train_inj.loc[seq_mask, label_col] == 1).any():
                    t_vals = train_inj.loc[seq_mask, temp_col].values
                    l_vals = train_inj.loc[seq_mask, label_col].values
                    train_inj.loc[seq_mask, temp_col] = inject_failure(t_vals, l_vals)
                    t = train_inj.loc[seq_mask, temp_col]
                    diff_col_name = f'data_motor_{motor_id}_temperature_diff'
                    if diff_col_name in train_inj.columns:
                        train_inj.loc[seq_mask, diff_col_name] = t.diff(20).fillna(0).values
                    for win, suffix in [(5, '5'), (20, '20')]:
                        for stat in ['mean', 'max', 'std']:
                            col = f'data_motor_{motor_id}_temperature_roll_{stat}_{suffix}'
                            if col in train_inj.columns:
                                if stat == 'mean':
                                    train_inj.loc[seq_mask, col] = t.rolling(win, min_periods=1).mean().values
                                elif stat == 'max':
                                    train_inj.loc[seq_mask, col] = t.rolling(win, min_periods=1).max().values
                                else:
                                    train_inj.loc[seq_mask, col] = t.rolling(win, min_periods=1).std().fillna(0).values
            raw_motor_train_df = train_inj

        X_train = raw_motor_train_df[features]
        y_train = raw_motor_train_df[label_col]
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        
        print(f"Train set -> Majority class (0): {num_neg}, Minority class (1): {num_pos}")
        
        # Moderate weighting to help rare events without destroying precision
        if motor_id in [3, 4] and num_pos > 0:
            print("Applying custom distance-based similarity weighting...")
            sample_weight = np.ones(len(y_train))
            sample_weight[y_train == 1] = 5.0  # Moderate minority weight
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            centroid_1 = X_scaled[y_train == 1].mean(axis=0)
            majority_mask = (y_train == 0)
            dists_0 = np.linalg.norm(X_scaled[majority_mask] - centroid_1, axis=1)
            min_d = dists_0.min()
            max_d = dists_0.max()
            if max_d > min_d:
                sim_0 = 1.0 - ((dists_0 - min_d) / (max_d - min_d))
            else:
                sim_0 = np.zeros_like(dists_0)
            weight_0 = 0.5 + sim_0 * 1.0
            sample_weight[majority_mask] = weight_0
        else:
            if num_pos > 0:
                raw_weight = num_neg / num_pos
                scale_pos_weight = min(raw_weight, 5.0)  # Capped moderate weight
            else:
                scale_pos_weight = 1.0
            sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
            
        X_val = None
        y_val = None
        if len(val_seqs) > 0:
            X_val = raw_motor_val_df[features]
            y_val = raw_motor_val_df[label_col]
            
        best_model = None
        best_val_score = -1.0
        best_params_info = {}
        
        print(f"Tuning model hyperparams and threshold via predict_proba...")
        import itertools
        
        if motor_id == 1:
            # LogisticRegression Grid
            C_values = [0.01, 0.1, 1.0, 10.0]
            param_combinations = [{'C': c} for c in C_values]
        else:
            # XGBoost Grid
            param_grid = {
                'learning_rate': [0.02, 0.05],
                'n_estimators': [100, 150],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 10]
            }
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        closing_options = [False, True]
        
        for params in param_combinations:
            if motor_id == 1:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(C=params['C'], random_state=42, max_iter=1000))
                ])
                model.fit(X_train, y_train, lr__sample_weight=sample_weight)
            else:
                model = xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0,
                    tree_method='hist',
                    **params
                )
                model.fit(X_train, y_train, sample_weight=sample_weight)
                
            if X_val is not None:
                # Use predict_proba for tuning
                if motor_id == 1:
                    y_prob_val = model.predict_proba(X_val)[:, 1]
                else:
                    y_prob_val = model.predict_proba(X_val)[:, 1]
                    
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
                            best_params_info = {'model_params': params, 'threshold': t, 'use_closing': use_closing}
            else:
                best_model = model
                best_params_info = {'model_params': params, 'threshold': 0.5, 'use_closing': True}
                break # Only need one if no val
                
        model = best_model
        models[motor_id] = {'model': model, 'threshold': best_params_info.get('threshold', 0.5), 'use_closing': best_params_info.get('use_closing', True)}
        
        if X_val is not None:
            print(f"Best Validation F1 (Class 1): {best_val_score:.4f}")
            print(f"Best Configuration: {best_params_info}")
            val_f1_per_motor[motor_id] = best_val_score
        else:
            val_f1_per_motor[motor_id] = None
            
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_prob_train >= best_params_info.get('threshold', 0.5)).astype(int)
        if best_params_info.get('use_closing', True):
            y_pred_train = binary_closing(y_pred_train, structure=np.ones(5)).astype(int)
        print("Training Classification Report (Best Configuration):")
        print(classification_report(y_train, y_pred_train, zero_division=0))
        
        if X_val is not None:
            y_prob_val = model.predict_proba(X_val)[:, 1]
            y_pred_val = (y_prob_val >= best_params_info.get('threshold', 0.5)).astype(int)
            if best_params_info.get('use_closing', True):
                y_pred_val = binary_closing(y_pred_val, structure=np.ones(5)).astype(int)
            print("Validation Classification Report (Best Configuration):")
            print(classification_report(y_val, y_pred_val, zero_division=0))
    
    scored = {m: s for m, s in val_f1_per_motor.items() if s is not None}
    print("\n========== VALIDATION F1 SUMMARY ==========")
    for m in range(1, 7):
        s = val_f1_per_motor.get(m)
        print(f"  Motor {m}: {s:.4f}" if s is not None else f"  Motor {m}: no val set")
    if scored:
        global_f1 = sum(scored.values()) / len(scored)
        print(f"  ---")
        print(f"  Global mean F1 ({len(scored)} motors): {global_f1:.4f}")
    print("============================================\n")

    print("\nGenerating predictions for sample_submission.csv format...")
    sample_sub_path = os.path.join(base_dir, 'sample_submission.csv')
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
            # No data for this test condition? Fill 0s
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
                y_pred = binary_closing(y_pred, structure=np.ones(5)).astype(int)
            
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
        
    if args.exclude_motor is not None:
        print(f"\nExcluding Motor {args.exclude_motor}: Setting its predictions to -1.")
        final_sub[f'data_motor_{args.exclude_motor}_label'] = -1
        out_path = os.path.join(r'C:\Users\oscar\Desktop\TDS INDUSTRY', f'motor_excluded_{args.exclude_motor}_submission.csv')
    else:
        out_path = r'C:\Users\oscar\Desktop\TDS INDUSTRY\motor_submission.csv'
        
    final_sub.to_csv(out_path, index=False)
    print(f"\nSaved final submission to {out_path}")

if __name__ == '__main__':
    main()
