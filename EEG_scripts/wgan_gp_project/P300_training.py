# =================================================================================================
#
#       P300 TRANSFER LEARNING & EVALUATION PIPELINE (v3 - Corrected Data Loading)
#
# This script implements a leave-one-subject-out transfer learning paradigm for P300
# classification.
#
# V3 CORRECTIONS:
# - Reverted the data loading and preprocessing logic to be a direct, line-for-line
#   copy of the original n_p300.py script to resolve any data handling errors.
# - The custom helper function has been removed in favor of the original, proven code block.
#   This block is now used within the loops for both source and target data aggregation.
#
# =================================================================================================

import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import sem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector
import warnings
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")

# --- Configuration ---
SEED_VALUE = 42
np.random.seed(SEED_VALUE)

SESSION_IDS = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
CH_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']

# Preprocessing Parameters
FS = 80
ARTIFTH = 50
TS_START_SEC, TS_END_SEC = -0.1, 0.6
BASELINE_START_SEC, BASELINE_END_SEC = -0.1, 0.0

# Data Splitting Parameters (Identical to original GAN script)
NUM_TRAIN_TARGET_TRIALS_BASE = 60
NUM_VALID_TARGET_TRIALS_BASE = 15
NUM_TRAIN_NONTARGET_TRIALS_BASE = 120
NUM_VALID_NONTARGET_TRIALS_BASE = 45

# For TL, we combine train and valid sets of the target subject
NUM_TRAIN_TARGET_TRIALS = NUM_TRAIN_TARGET_TRIALS_BASE + NUM_VALID_TARGET_TRIALS_BASE
NUM_TRAIN_NONTARGET_TRIALS = NUM_TRAIN_NONTARGET_TRIALS_BASE + NUM_VALID_NONTARGET_TRIALS_BASE

# Global variable for log file path
current_log_file_path = None

# =============================================================================
# PART 1: UTILITY AND LOGGING FUNCTIONS (Unchanged)
# =============================================================================

def log_message(message):
    print(message)
    global current_log_file_path
    if current_log_file_path:
        with open(current_log_file_path, 'a') as f: f.write(str(message) + "\n")

# =============================================================================
# PART 2: P300 CLASSIFIER & FEATURE ENGINEERING (Unchanged)
# =============================================================================

def create_averaged_features_from_trials(single_trials_data, single_trials_labels, n_average=15):
    feature_list, label_list = [], []
    for label_class in [1, 2]:
        class_indices = np.where(single_trials_labels == label_class)[0]
        class_trials = single_trials_data[class_indices]
        num_features_for_class = len(class_trials) // n_average
        for i in range(num_features_for_class):
            trial_group = class_trials[i * n_average : (i + 1) * n_average]
            averaged_feature = np.mean(trial_group, axis=0)
            feature_list.append(averaged_feature)
            label_list.append(label_class)
    if not feature_list:
        return {'x': np.empty((single_trials_data.shape[1], single_trials_data.shape[2], 0)), 'y': np.empty((0,))}
    p300_features = np.array(feature_list).transpose(1, 2, 0)
    p300_labels = np.array(label_list)
    return {'x': p300_features, 'y': p300_labels}

def train_p300_lda(training_features):
    p300_feature, p300_labels = training_features['x'], training_features['y'].ravel()
    if p300_feature.shape[2] == 0: return None, None
    num_channels, num_samples, num_trials = p300_feature.shape
    p300_feature_reshaped = p300_feature.transpose(2, 1, 0).reshape(num_trials, -1)
    sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), k_features=5, forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=-1)
    sfs.fit(p300_feature_reshaped, p300_labels)
    selected_feature_indices = list(sfs.k_feature_idx_)
    train_data_selected = p300_feature_reshaped[:, selected_feature_indices]
    final_lda_model = LinearDiscriminantAnalysis()
    final_lda_model.fit(train_data_selected, p300_labels)
    return final_lda_model, selected_feature_indices

def evaluate_p300_lda(model, feature_indices, evaluation_features):
    if model is None or feature_indices is None: return 0.0
    p300_feature, true_labels = evaluation_features['x'], evaluation_features['y'].ravel()
    if p300_feature.shape[2] == 0: return 0.0
    num_channels, num_samples, num_trials = p300_feature.shape
    p300_feature_reshaped = p300_feature.transpose(2, 1, 0).reshape(num_trials, -1)
    test_data_selected = p300_feature_reshaped[:, feature_indices]
    predicted_labels = model.predict(test_data_selected)
    return np.mean(predicted_labels == true_labels) * 100

# =============================================================================
# PART 3: PLOTTING FUNCTIONS (Unchanged)
# =============================================================================

def plot_accuracy_chart(accuracies, subject_id_str, session_pair_str, output_dir):
    labels, values = list(accuracies.keys()), list(accuracies.values())
    plt.figure(figsize=(10, 7)); bars = plt.bar(labels, values, color=['skyblue', 'lightgreen'])
    plt.ylabel('Classification Accuracy (%)')
    plt.title(f'Target {subject_id_str} (Sessions {session_pair_str}): Accuracy Comparison')
    plt.ylim(0, 100); plt.xticks(rotation=15, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Accuracy_Comparison_{subject_id_str}_Sess{session_pair_str}.png"))
    plt.close()

# =============================================================================
# PART 4: MAIN ANALYSIS SCRIPT
# =============================================================================

def run_leave_one_out_experiment(all_subject_folders):
    base_results_dir = "results_TL_v3"
    os.makedirs(base_results_dir, exist_ok=True)

    for target_subject_folder in all_subject_folders:
        target_subject_id = os.path.basename(target_subject_folder)
        source_subject_folders = [f for f in all_subject_folders if f != target_subject_folder]

        print(f"\n{'='*30}\nSetting Target Subject: {target_subject_id}\n{'='*30}")

        # --- 1. AGGREGATE ALL SOURCE DOMAIN DATA ---
        print(f"--- Step 1: Loading and aggregating data from {len(source_subject_folders)} source subjects ---")
        all_source_trials_list, all_source_labels_list = [], []
        for source_folder in source_subject_folders:
            print(f"  Processing source subject: {os.path.basename(source_folder)}")
            for s_idx, s in enumerate(range(0, 7, 2)):
                try:
                    # === START: ORIGINAL DATA LOADING AND PREPROCESSING BLOCK ===
                    raw1 = mne.io.read_raw_gdf(os.path.join(source_folder, f"{SESSION_IDS[s]}.gdf"), preload=True, verbose=False)
                    raw2 = mne.io.read_raw_gdf(os.path.join(source_folder, f"{SESSION_IDS[s+1]}.gdf"), preload=True, verbose=False)
                    raw1.rename_channels({old: new for old, new in zip(raw1.ch_names, CH_NAMES)})
                    raw2.rename_channels({old: new for old, new in zip(raw2.ch_names, CH_NAMES)})
                    raw = mne.concatenate_raws([raw1, raw2], verbose=False)
                    original_fs = raw.info['sfreq']
                    data = raw.get_data()
                    b_hp, a_hp = butter(4, 0.4 / (original_fs / 2), 'high')
                    b_lp, a_lp = butter(4, 30 / (original_fs / 2), 'low')
                    b_n1, a_n1 = iirnotch(4.3 if s in [0,4] else 5.8, 5, original_fs)
                    b_n2, a_n2 = iirnotch((4.3 if s in [0,4] else 5.8)*2, 5, original_fs)
                    for j in range(data.shape[0]):
                        data[j, :] = filtfilt(b_hp, a_hp, filtfilt(b_lp, a_lp, data[j, :]))
                        data[j, :] = filtfilt(b_n1, a_n1, filtfilt(b_n2, a_n2, data[j, :]))
                    raw._data = data; raw.resample(FS, verbose=False)
                    events, event_id = mne.events_from_annotations(raw, verbose=False)
                    id_to_desc = {v: k for k, v in event_id.items()}
                    event_codes = np.array([int(id_to_desc[code]) for code in events[:, 2]])
                    stim_codes = np.zeros_like(event_codes)
                    stim_codes[event_codes == 33286] = 2; stim_codes[event_codes == 33285] = 1
                    flash_indices = np.where(stim_codes != 0)[0]
                    EEG = raw.get_data() * 1e6
                    ts_samples = np.arange(round(TS_START_SEC * FS), round(TS_END_SEC * FS))
                    bl_samples = np.arange(round(BASELINE_START_SEC * FS), round(BASELINE_END_SEC * FS))
                    clean_trials_target, clean_trials_nontarget = [], []
                    for i in flash_indices:
                        if i + 1 >= len(events): continue
                        latency = events[i + 1, 0]
                        epoch_indices = latency + ts_samples
                        if np.min(epoch_indices) < 0 or np.max(epoch_indices) >= EEG.shape[1]: continue
                        ep = EEG[:, epoch_indices]
                        bl_indices = events[i, 0] + bl_samples
                        if np.min(bl_indices) < 0 or np.max(bl_indices) >= EEG.shape[1]: continue
                        BLamp = np.mean(EEG[:, bl_indices], axis=1, keepdims=True)
                        ep_bc = ep - BLamp
                        if np.any(np.max(ep_bc, axis=1) - np.min(ep_bc, axis=1) > ARTIFTH): continue
                        if stim_codes[i] == 1: clean_trials_target.append(ep_bc)
                        elif stim_codes[i] == 2: clean_trials_nontarget.append(ep_bc)
                    if not clean_trials_target or not clean_trials_nontarget: continue
                    all_trials = np.array(clean_trials_target + clean_trials_nontarget)
                    all_labels = np.array([1]*len(clean_trials_target) + [2]*len(clean_trials_nontarget))
                    # === END: ORIGINAL DATA LOADING AND PREPROCESSING BLOCK ===
                    
                    all_source_trials_list.append(all_trials)
                    all_source_labels_list.append(all_labels)
                except Exception as e:
                    print(f"    Warning: Could not process session pair {s+1}-{s+2} for {os.path.basename(source_folder)}. Error: {e}")

        if not all_source_trials_list:
            print(f"CRITICAL: No source data could be loaded. Skipping target {target_subject_id}."); continue
        
        source_trials = np.concatenate(all_source_trials_list, axis=0)
        source_labels = np.concatenate(all_source_labels_list, axis=0)
        print(f"--- Source data aggregation complete. Total trials: {source_trials.shape[0]} ---")


        # --- 2. PROCESS EACH SESSION PAIR OF THE TARGET SUBJECT ---
        for s in range(0, 7, 2):
            session_pair_str = f"{s+1}-{s+2}"
            output_dir = os.path.join(base_results_dir, f"Target_{target_subject_id}_Sess_{session_pair_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            global current_log_file_path
            current_log_file_path = os.path.join(output_dir, "run_log.txt")
            with open(current_log_file_path, 'w') as f: f.write(f"Log for Target {target_subject_id}, Sessions {session_pair_str}\n{'='*40}\n")
            
            log_message(f"\n\n--- Processing Target: {target_subject_id}, Sessions: {session_pair_str} ---")

            try:
                # --- 3. LOAD AND SPLIT TARGET DATA USING ORIGINAL LOGIC ---
                log_message("\n--- Step 3: Loading and splitting target data ---")
                
                # === START: ORIGINAL DATA LOADING AND PREPROCESSING BLOCK ===
                raw1 = mne.io.read_raw_gdf(os.path.join(target_subject_folder, f"{SESSION_IDS[s]}.gdf"), preload=True)
                raw2 = mne.io.read_raw_gdf(os.path.join(target_subject_folder, f"{SESSION_IDS[s+1]}.gdf"), preload=True)
                raw1.rename_channels({old: new for old, new in zip(raw1.ch_names, CH_NAMES)})
                raw2.rename_channels({old: new for old, new in zip(raw2.ch_names, CH_NAMES)})
                raw = mne.concatenate_raws([raw1, raw2])
                original_fs = raw.info['sfreq']
                data = raw.get_data()
                b_hp, a_hp = butter(4, 0.4 / (original_fs / 2), 'high')
                b_lp, a_lp = butter(4, 30 / (original_fs / 2), 'low')
                b_n1, a_n1 = iirnotch(4.3 if s in [0,4] else 5.8, 5, original_fs)
                b_n2, a_n2 = iirnotch((4.3 if s in [0,4] else 5.8)*2, 5, original_fs)
                for j in range(data.shape[0]):
                    data[j, :] = filtfilt(b_hp, a_hp, filtfilt(b_lp, a_lp, data[j, :]))
                    data[j, :] = filtfilt(b_n1, a_n1, filtfilt(b_n2, a_n2, data[j, :]))
                raw._data = data; raw.resample(FS)
                log_message(f"Filtering and resampling complete. New Fs: {raw.info['sfreq']} Hz")
                events, event_id = mne.events_from_annotations(raw)
                id_to_desc = {v: k for k, v in event_id.items()}
                event_codes = np.array([int(id_to_desc[code]) for code in events[:, 2]])
                stim_codes = np.zeros_like(event_codes)
                stim_codes[event_codes == 33286] = 2; stim_codes[event_codes == 33285] = 1
                flash_indices = np.where(stim_codes != 0)[0]
                EEG = raw.get_data() * 1e6
                ts_samples = np.arange(round(TS_START_SEC * FS), round(TS_END_SEC * FS))
                bl_samples = np.arange(round(BASELINE_START_SEC * FS), round(BASELINE_END_SEC * FS))
                clean_trials_target, clean_trials_nontarget = [], []
                for i in flash_indices:
                    if i + 1 >= len(events): continue
                    latency = events[i + 1, 0]
                    epoch_indices = latency + ts_samples
                    if np.min(epoch_indices) < 0 or np.max(epoch_indices) >= EEG.shape[1]: continue
                    ep = EEG[:, epoch_indices]
                    bl_indices = events[i, 0] + bl_samples
                    if np.min(bl_indices) < 0 or np.max(bl_indices) >= EEG.shape[1]: continue
                    BLamp = np.mean(EEG[:, bl_indices], axis=1, keepdims=True)
                    ep_bc = ep - BLamp
                    if np.any(np.max(ep_bc, axis=1) - np.min(ep_bc, axis=1) > ARTIFTH): continue
                    if stim_codes[i] == 1: clean_trials_target.append(ep_bc)
                    elif stim_codes[i] == 2: clean_trials_nontarget.append(ep_bc)
                if not clean_trials_target or not clean_trials_nontarget:
                    log_message("CRITICAL: Not enough clean trials found. Skipping subject."); continue
                target_all_trials = np.array(clean_trials_target + clean_trials_nontarget)
                target_all_labels = np.array([1]*len(clean_trials_target) + [2]*len(clean_trials_nontarget))
                log_message(f"Found {len(clean_trials_target)} clean Target and {len(clean_trials_nontarget)} clean Non-Target trials.")
                # === END: ORIGINAL DATA LOADING AND PREPROCESSING BLOCK ===
                
                # Perform sequential split
                idx_c1 = np.where(target_all_labels == 1)[0]; idx_c2 = np.where(target_all_labels == 2)[0]
                total_needed_c1 = NUM_TRAIN_TARGET_TRIALS_BASE + NUM_VALID_TARGET_TRIALS_BASE
                total_needed_c2 = NUM_TRAIN_NONTARGET_TRIALS_BASE + NUM_VALID_NONTARGET_TRIALS_BASE
                if len(idx_c1) < total_needed_c1 or len(idx_c2) < total_needed_c2:
                    log_message(f"CRITICAL: Not enough trials for split. Have T:{len(idx_c1)}/NT:{len(idx_c2)}, need T:{total_needed_c1}/NT:{total_needed_c2}. Skipping."); continue
                
                train_base_idx = np.concatenate((idx_c1[:NUM_TRAIN_TARGET_TRIALS_BASE], idx_c2[:NUM_TRAIN_NONTARGET_TRIALS_BASE]))
                valid_idx = np.concatenate((idx_c1[NUM_TRAIN_TARGET_TRIALS_BASE:total_needed_c1], idx_c2[NUM_TRAIN_NONTARGET_TRIALS_BASE:total_needed_c2]))
                test_idx = np.concatenate((idx_c1[total_needed_c1:], idx_c2[total_needed_c2:]))

                target_train_idx = np.concatenate((train_base_idx, valid_idx))
                target_train_trials, target_train_labels = target_all_trials[target_train_idx], target_all_labels[target_train_idx]
                target_test_trials, target_test_labels = target_all_trials[test_idx], target_all_labels[test_idx]
                log_message(f"Target data split complete. Training set size: {len(target_train_trials)}, Test set size: {len(target_test_trials)}")

                # --- 4. NORMALIZATION ---
                log_message("\n--- Step 4: Normalizing Data ---")
                train_min_b, train_max_b = np.min(target_train_trials), np.max(target_train_trials)
                normalize_b = lambda x: 2 * (x - train_min_b) / (train_max_b - train_min_b) - 1
                target_train_norm_b = normalize_b(target_train_trials)
                target_test_norm_b = np.clip(normalize_b(target_test_trials), -1, 1)

                train_min_tl, train_max_tl = np.min(source_trials), np.max(source_trials)
                normalize_tl = lambda x: 2 * (x - train_min_tl) / (train_max_tl - train_min_tl) - 1
                source_trials_norm = normalize_tl(source_trials)
                target_train_norm_tl = np.clip(normalize_tl(target_train_trials), -1, 1)
                target_test_norm_tl = np.clip(normalize_tl(target_test_trials), -1, 1)
                
                # --- 5. CREATE AVERAGED FEATURES ---
                log_message("\n--- Step 5: Creating Averaged Features ---")
                train_features_b = create_averaged_features_from_trials(target_train_norm_b, target_train_labels)
                test_features_b = create_averaged_features_from_trials(target_test_norm_b, target_test_labels)

                transfer_train_trials = np.concatenate((source_trials_norm, target_train_norm_tl), axis=0)
                transfer_train_labels = np.concatenate((source_labels, target_train_labels))
                train_features_tl = create_averaged_features_from_trials(transfer_train_trials, transfer_train_labels)
                test_features_tl = create_averaged_features_from_trials(target_test_norm_tl, target_test_labels)
                
                # --- 6. BASELINE & TRANSFER LEARNING EVALUATION ---
                log_message("\n--- Step 6: Training and Evaluating Models ---")
                model_baseline, sf_baseline = train_p300_lda(train_features_b)
                accuracy_baseline = evaluate_p300_lda(model_baseline, sf_baseline, test_features_b)
                log_message(f"  Baseline Accuracy (Target Data Only): {accuracy_baseline:.2f}%")

                model_transfer, sf_transfer = train_p300_lda(train_features_tl)
                accuracy_transfer = evaluate_p300_lda(model_transfer, sf_transfer, test_features_tl)
                log_message(f"  Transfer Learning Accuracy (Source + Target Data): {accuracy_transfer:.2f}%")

                # --- 7. FINAL PLOTTING ---
                log_message("\n--- Step 7: Generating Final Plots ---")
                accuracies_summary = {
                    'Baseline (Target Only)': accuracy_baseline,
                    'Transfer Learning': accuracy_transfer
                }
                plot_accuracy_chart(accuracies_summary, target_subject_id, session_pair_str, output_dir)
                log_message(f"--- Finished Target: {target_subject_id}, Sessions: {session_pair_str} ---")

            except Exception as e:
                log_message(f"\n--- CRITICAL ERROR processing target {target_subject_id}, sessions {session_pair_str} ---\nError: {e}\n{traceback.format_exc()}---")

if __name__ == '__main__':
    data_folders_to_process = [f'H{i}' for i in range(1, 13)]
    
    for folder in data_folders_to_process:
        if not os.path.isdir(folder):
            print(f"Warning: Subject folder '{folder}' not found. Creating for demonstration.")
            os.makedirs(folder)
            for session_id in SESSION_IDS:
                with open(os.path.join(folder, f'{session_id}.gdf'), 'w') as f:
                    pass

    run_leave_one_out_experiment(data_folders_to_process)
    print("\n\n--- All subjects processed. ---")