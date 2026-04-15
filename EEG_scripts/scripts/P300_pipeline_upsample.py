#!/usr/bin/env python3
"""
P300 cWGAN-GP Classification Pipeline (Explicit Upsampling Edition)
=====================================================================
Methodology Highlights:
- Class Balancing: Explicitly duplicates target trials (Random Oversampling) to match non-targets.
- Mathematical Stability: Relies on LDA shrinkage='auto' to handle singular covariance matrices.
- No Data Leakage: Training strictly on Session 1 (QUIC), Testing strictly on Session 2 (JUMP).
- Strict Hyperparameter Selection: The best Strategy is chosen based entirely on internal 
  3-Fold Cross-Validation on the Training set. The Test set is kept completely blind.
"""

import scipy.io
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import os, csv, traceback, sys

try:
    from mlxtend.feature_selection import SequentialFeatureSelector
except ImportError:
    print("\n[!] WARNING: 'mlxtend' is not installed! Sequential Feature Selection will fail.")
    print("    Please run: pip install mlxtend\n")
    sys.exit(1)

import mne

SEED_VALUE = 42
np.random.seed(SEED_VALUE)

P300_ORIG_FS = 500
P300_TARGET_FS = 80
P300_NCHAN = 8
P300_CHAN_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']

P300_ERP_TMIN = -0.1
P300_ERP_TMAX = 0.8
P300_FEAT_TMIN = 0.15
P300_FEAT_TMAX = 0.5
P300_BL_TMIN = -0.1
P300_BL_TMAX = 0.0

P300_N_AVG = 15

EVT_TARGET = 33285
EVT_NONTARGET = 33286
EVT_ROW_END = 32777
EVT_COL_END = 32778

SESSION_IDS = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
CONDITIONS = [
    {'s': 1, 'matrix': '3x3', 'isi': 117, 'notch': [4.3, 8.6],  'n_rows': 3, 'n_cols': 3},
    {'s': 3, 'matrix': '3x3', 'isi': 175, 'notch': [5.8, 11.6], 'n_rows': 3, 'n_cols': 3},
    {'s': 5, 'matrix': '5x5', 'isi': 117, 'notch': [4.3, 8.6],  'n_rows': 5, 'n_cols': 5},
    {'s': 7, 'matrix': '5x5', 'isi': 175, 'notch': [5.8, 11.6], 'n_rows': 5, 'n_cols': 5},
]

QUICK_N_LETTERS = 5
JUMP_N_LETTERS = 4
N_BL_DRAWS = 10


def log_message(msg, log_path=None):
    print(msg)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(str(msg) + '\n')

def design_filters_at_orig_fs(orig_fs=P300_ORIG_FS):
    from scipy.signal import butter, buttord
    nyq = orig_fs / 2.0
    N_hp, Wn_hp = buttord(0.5 / nyq, 0.4 / nyq, 1, 40)
    sos_hp = butter(N_hp, Wn_hp, btype='high', output='sos')
    N_lp, Wn_lp = buttord(30 / nyq, 40 / nyq, 1, 80)
    sos_lp = butter(N_lp, Wn_lp, btype='low', output='sos')
    return sos_hp, sos_lp

def design_notch_filters(notch_freqs, orig_fs=P300_ORIG_FS, n_cascade=3):
    from scipy.signal import iirnotch
    filters = []
    for nf in notch_freqs:
        if nf < orig_fs / 2:
            b, a = iirnotch(nf, Q=5, fs=orig_fs)
            filters.append((b, a, n_cascade))
    return filters

def load_and_preprocess_gdf(gdf_path, sos_hp, sos_lp, notch_filts, nchan=P300_NCHAN):
    from scipy.signal import sosfiltfilt, filtfilt
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, eog=None, verbose=False)
    orig_fs = raw.info['sfreq']
    data = raw.get_data()[:nchan, :].copy()

    for ch in range(nchan):
        data[ch] = sosfiltfilt(sos_hp, data[ch])
        data[ch] = sosfiltfilt(sos_lp, data[ch])
        for b_n, a_n, n_casc in notch_filts:
            for _ in range(n_casc):
                data[ch] = filtfilt(b_n, a_n, data[ch])

    evs, eid = mne.events_from_annotations(raw, verbose=False)
    id2desc = {v: k for k, v in eid.items()}
    event_codes = np.array([int(id2desc[c]) for c in evs[:, 2]])
    event_latencies = evs[:, 0]
    return data, event_codes, event_latencies, orig_fs

def resample_data(data, orig_fs, target_fs=P300_TARGET_FS):
    import math
    from scipy.signal import resample_poly
    g = math.gcd(int(target_fs), int(orig_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    return resample_poly(data, up, down, axis=1).astype(np.float64)

def resample_latencies(latencies, orig_fs, target_fs=P300_TARGET_FS):
    return np.round(latencies * target_fs / orig_fs).astype(int)

def preprocess_condition(subj_path, cond, target_fs=P300_TARGET_FS):
    s = cond['s']
    s_idx = s - 1
    gdf1 = os.path.join(subj_path, f'{SESSION_IDS[s_idx]}.gdf')
    gdf2 = os.path.join(subj_path, f'{SESSION_IDS[s_idx + 1]}.gdf')
    if not os.path.exists(gdf1) or not os.path.exists(gdf2):
        return None

    sos_hp, sos_lp = design_filters_at_orig_fs()
    notch_filts = design_notch_filters(cond['notch'])

    data1, codes1, lats1, orig_fs = load_and_preprocess_gdf(gdf1, sos_hp, sos_lp, notch_filts)
    data2, codes2, lats2, _ = load_and_preprocess_gdf(gdf2, sos_hp, sos_lp, notch_filts)

    data1_rs = resample_data(data1, orig_fs, target_fs)
    data2_rs = resample_data(data2, orig_fs, target_fs)
    lats1_rs = resample_latencies(lats1, orig_fs, target_fs)
    lats2_rs = resample_latencies(lats2, orig_fs, target_fs)

    offset = data1_rs.shape[1]
    a2_lats = lats2_rs + offset

    EEG = np.concatenate([data1_rs, data2_rs], axis=1) * 1e6
    EEG = EEG - np.mean(EEG, axis=0, keepdims=True)

    stimes = np.round(np.concatenate([lats1_rs, a2_lats])).astype(int)

    return {
        'EEG': EEG, 'stimes': stimes,
        'a1_codes': codes1, 'a1_lats': lats1_rs,
        'a2_codes': codes2, 'a2_lats': a2_lats,
    }

def extract_row_col_flash_indices(event_codes, n_rows, n_cols):
    rc_indices = {}
    for r in range(n_rows):
        code = 33025 + r
        raw_idx = np.where(event_codes == code)[0]
        valid = [idx for idx in raw_idx if idx > 0 and event_codes[idx - 1] != EVT_ROW_END]
        rc_indices[('row', r)] = np.array(valid)
    col_start = 33025 + n_rows
    for c in range(n_cols):
        code = col_start + c
        raw_idx = np.where(event_codes == code)[0]
        valid = [idx for idx in raw_idx if idx + 1 < len(event_codes) and event_codes[idx + 1] != EVT_COL_END]
        rc_indices[('col', c)] = np.array(valid)
    return rc_indices

def build_15avg_classification_features(EEG, stimes, event_codes, all_lats,
                                       rc_indices, n_letters, n_rows, n_cols,
                                       fs=P300_TARGET_FS, nchan=P300_NCHAN):
    ts_f = np.arange(round(P300_FEAT_TMIN * fs), round(P300_FEAT_TMAX * fs))
    bl_start = round(P300_BL_TMIN * fs)
    bl_end = 0

    all_m_erp, all_labels = [], []

    for rc_type in ['row', 'col']:
        n_items = n_rows if rc_type == 'row' else n_cols
        for rc_idx in range(n_items):
            flash_indices = rc_indices.get((rc_type, rc_idx), np.array([]))
            if len(flash_indices) == 0: continue
            for letter_idx in range(n_letters):
                start = letter_idx * P300_N_AVG
                end = start + P300_N_AVG
                if end > len(flash_indices): continue
                
                block = flash_indices[start:end]
                epochs = []
                for idx_in_session_events in block:
                    flash_lat = all_lats[idx_in_session_events]
                    stimes_idx = np.argmin(np.abs(stimes - flash_lat))
                    if stimes_idx + 1 >= len(stimes): continue
                    
                    data_lat = stimes[stimes_idx + 1]
                    bl_samples = np.arange(stimes[stimes_idx] + bl_start, stimes[stimes_idx] + bl_end + 1)
                    ep_samples = data_lat + ts_f
                    
                    if ep_samples.min() < 0 or ep_samples.max() >= EEG.shape[1]: continue
                    ep = EEG[:nchan, ep_samples]
                    bl_amp = np.mean(EEG[:nchan, bl_samples], axis=1, keepdims=True)
                    epochs.append(ep - bl_amp)
                
                if len(epochs) == 0: continue
                m_erp = np.mean(np.stack(epochs, axis=2), axis=2)
                all_m_erp.append(m_erp)
                
                label_code = int(stats.mode(event_codes[block - 1], keepdims=False).mode)
                all_labels.append(2 if label_code == EVT_TARGET else 1)

    if not all_m_erp: return None, None
    return np.stack(all_m_erp, axis=2), np.array(all_labels, dtype=np.float32)

def normalize_per_channel(x, clip_sigma=3.0, norm_stats=None):
    if norm_stats is None:
        if x.ndim == 3:
            ch_mean = np.mean(x, axis=(1, 2), keepdims=True)
            ch_std = np.std(x, axis=(1, 2), keepdims=True) + 1e-8
        else:
            ch_mean = np.mean(x, axis=1, keepdims=True)
            ch_std = np.std(x, axis=1, keepdims=True) + 1e-8
    else:
        ch_mean, ch_std = norm_stats
    z = (x - ch_mean) / ch_std
    return np.clip(z / clip_sigma, -1., 1.).astype(np.float32), (ch_mean, ch_std)

def explicit_upsample(X, y, target_label=2, target_count=None):
    """
    Explicitly duplicates trials of 'target_label' until they equal 'target_count'.
    X should be a 2D array (Trials, Features). y should be (Trials,).
    """
    tgt_idx = np.where(y == target_label)[0]
    ntgt_idx = np.where(y != target_label)[0]
    
    if len(tgt_idx) == 0: return X, y
    
    # If target_count not provided, upsample to match the majority class
    if target_count is None:
        target_count = len(ntgt_idx)
        
    n_avail = len(tgt_idx)
    reps = target_count // n_avail
    rem = target_count % n_avail
    
    # Duplicate indices
    chosen_idx = np.concatenate([np.tile(tgt_idx, reps), tgt_idx[:rem]])
    
    X_up = np.concatenate([X[chosen_idx], X[ntgt_idx]], axis=0)
    y_up = np.concatenate([np.ones(target_count)*target_label, y[ntgt_idx]], axis=0)
    return X_up, y_up

def train_p300_lda_with_sfs(X_train, y_train, k_features=5):
    """ Trains LDA with internal feature selection. Expects pre-upsampled balanced data. """
    clf_base = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    try:
        sfs = SequentialFeatureSelector(
            clf_base, k_features=min(k_features, X_train.shape[1]), forward=True, floating=True,
            scoring='balanced_accuracy', cv=5, n_jobs=-1)
        sfs.fit(X_train, y_train)
        selected = list(sfs.k_feature_idx_)
    except Exception:
        selected = list(range(min(k_features, X_train.shape[1])))
        
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_train[:, selected], y_train)
    return clf, selected

def find_best_synth_combo(real_tgt_f, real_ntgt_f, synth_pool_f, n_synth_needed, ntgt_expected, n_search_iters=30):
    """
    Evaluates combinations of GAN data using 3-Fold Cross-Validation on the training set.
    Safely upsamples the targets *inside* the CV loop to prevent data leakage.
    Returns: best_synthetic_features, best_validation_cv_score
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED_VALUE)
    
    # Baseline fallback
    if n_synth_needed == 0:
        X_base = np.concatenate([real_tgt_f, real_ntgt_f], axis=2).transpose(2, 1, 0).reshape(real_tgt_f.shape[2] + real_ntgt_f.shape[2], -1)
        y_base = np.concatenate([np.ones(real_tgt_f.shape[2])*2, np.ones(real_ntgt_f.shape[2])*1])
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X_base, y_base):
            X_tr, y_tr = explicit_upsample(X_base[train_idx], y_base[train_idx], target_label=2)
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_tr, y_tr)
            cv_scores.append(balanced_accuracy_score(y_base[val_idx], clf.predict(X_base[val_idx])))
            
        return np.zeros((real_tgt_f.shape[0], real_tgt_f.shape[1], 0)), np.mean(cv_scores)
        
    n_avail_synth = synth_pool_f.shape[2]
    best_cv_acc = -1
    best_synth_f = None
    
    for _ in range(n_search_iters):
        rnd_idx = np.random.choice(n_avail_synth, size=min(n_synth_needed, n_avail_synth), replace=False)
        curr_synth = synth_pool_f[:, :, rnd_idx]
        
        combined_tgt = np.concatenate([real_tgt_f, curr_synth], axis=2) if real_tgt_f.shape[2] > 0 else curr_synth
        
        X_comb = np.concatenate([combined_tgt, real_ntgt_f], axis=2).transpose(2, 1, 0).reshape(combined_tgt.shape[2] + real_ntgt_f.shape[2], -1)
        y_comb = np.concatenate([np.ones(combined_tgt.shape[2])*2, np.ones(real_ntgt_f.shape[2])*1])
        
        cv_scores = []
        # Upsample strictly inside the training fold
        for train_idx, val_idx in skf.split(X_comb, y_comb):
            X_tr, y_tr = explicit_upsample(X_comb[train_idx], y_comb[train_idx], target_label=2)
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_tr, y_tr)
            cv_scores.append(balanced_accuracy_score(y_comb[val_idx], clf.predict(X_comb[val_idx])))
            
        cv_acc = np.mean(cv_scores)
        
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_synth_f = curr_synth
            
    return best_synth_f, best_cv_acc

def plot_accuracy(bl, so, best, sid, odir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    bars = ax.bar(['Baseline', 'SynthOnly', 'Best Mix'], [bl, so, best], color=colors, edgecolor='black')
    ax.set(ylabel='Balanced Test Accuracy (%)', title=f'P300 Test Accuracy - {sid}', ylim=(0, 110))
    ax.set_facecolor('#f8f8f8')
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'accuracy_upsampled_{sid}.png'), dpi=150)
    plt.close(fig)

def process_p300_upsample(subj_path, subject_name, cond, output_base, synth_mat_path):
    cond_label = f"{cond['matrix']}_{cond['isi']}ms"
    sid = f"{subject_name}_{cond_label}"
    odir = os.path.join(output_base, f'Subject_{sid}')
    os.makedirs(odir, exist_ok=True)
    log_path = os.path.join(odir, 'run_log.txt')
    with open(log_path, 'w') as f: f.write('')

    log_message(f"\n{'='*60}\n  P300 EXPLICIT UPSAMPLE PIPELINE {sid}\n{'='*60}", log_path)

    n_rows, n_cols = cond['n_rows'], cond['n_cols']
    ntgt_expected = (n_rows + n_cols - 2) * QUICK_N_LETTERS

    result = preprocess_condition(subj_path, cond)
    if result is None: 
        log_message(f"  ERROR: Could not preprocess condition data for {sid}", log_path)
        return None

    EEG, stimes = result['EEG'], result['stimes']
    synth_data = scipy.io.loadmat(synth_mat_path)
    synth_tgt_gan = synth_data['synthetic_tgt_x']
    ch_mean_s, ch_std_s = synth_data['norm_ch_mean'], synth_data['norm_ch_std']
    n_avail_synth = synth_tgt_gan.shape[0]

    a1_rc = extract_row_col_flash_indices(result['a1_codes'], n_rows, n_cols)
    a2_rc = extract_row_col_flash_indices(result['a2_codes'], n_rows, n_cols)

    train_feat, train_y = build_15avg_classification_features(
        EEG, stimes, result['a1_codes'], result['a1_lats'], a1_rc, QUICK_N_LETTERS, n_rows, n_cols)

    test_feat, test_y = build_15avg_classification_features(
        EEG, stimes, result['a2_codes'], result['a2_lats'], a2_rc, JUMP_N_LETTERS, n_rows, n_cols)

    if train_feat is None or test_feat is None: 
        log_message(f"  ERROR: Feature extraction returned None for {sid}", log_path)
        return None

    norm_stats = (ch_mean_s.reshape(-1, 1, 1), ch_std_s.reshape(-1, 1, 1))
    train_feat, _ = normalize_per_channel(train_feat, norm_stats=norm_stats)
    test_feat, _ = normalize_per_channel(test_feat, norm_stats=norm_stats)

    p300_win_start, p300_win_end = 12, 40
    if n_avail_synth > 0:
        synth_pool = synth_tgt_gan[:, :, p300_win_start:p300_win_end].transpose(1, 2, 0)
    else:
        synth_pool = np.zeros((train_feat.shape[0], p300_win_end-p300_win_start, 0))

    tgt_idx = np.where(train_y == 2)[0]
    ntgt_idx = np.where(train_y == 1)[0]
    
    actual_targets = len(tgt_idx)
    target_cap = min(10, actual_targets) 
    
    real_tgt_all = train_feat[:, :, tgt_idx[:target_cap]]
    real_ntgt_all = train_feat[:, :, ntgt_idx[:ntgt_expected]] 
    
    strategies = ['Baseline', 'SynthOnly', 'Mix+2', 'Mix+4', 'Mix+6', 'Mix+8']
    cv_scores_by_strat = {s: [] for s in strategies}
    test_scores_by_strat = {s: [] for s in strategies}

    def evaluate_combination_on_test(r_tgt, s_tgt, r_ntgt):
        # Combine Targets
        tgt_f = np.concatenate([r_tgt, s_tgt], axis=2) if r_tgt.shape[2] > 0 else s_tgt
        
        # Flatten to 2D
        X_tgt = tgt_f.transpose(2, 1, 0).reshape(tgt_f.shape[2], -1)
        X_ntgt = r_ntgt.transpose(2, 1, 0).reshape(r_ntgt.shape[2], -1)
        
        X_unbal = np.concatenate([X_tgt, X_ntgt], axis=0)
        y_unbal = np.concatenate([np.ones(X_tgt.shape[0])*2, np.ones(X_ntgt.shape[0])*1])
        
        # Explicitly upsample training data here
        X_train_up, y_train_up = explicit_upsample(X_unbal, y_unbal, target_label=2)
        
        # Train and Evaluate
        clf, sel = train_p300_lda_with_sfs(X_train_up, y_train_up)
        if clf:
            X_test = test_feat.transpose(2, 1, 0).reshape(test_feat.shape[2], -1)[:, sel]
            y_pred = clf.predict(X_test)
            return balanced_accuracy_score(test_y, y_pred) * 100
        return 0.0

    for draw in range(N_BL_DRAWS):
        rng = np.random.RandomState(SEED_VALUE + draw)
        shuffled_real_tgt = real_tgt_all[:, :, rng.permutation(real_tgt_all.shape[2])]

        for s_name in strategies:
            if s_name == 'Baseline':
                n_real, n_synth = target_cap, 0
            elif s_name == 'SynthOnly':
                n_real, n_synth = 0, 10
            else:
                n_synth = int(s_name.split('+')[1])
                n_real = max(0, target_cap - n_synth)

            curr_real = shuffled_real_tgt[:, :, :n_real] if n_real > 0 else np.zeros((train_feat.shape[0], train_feat.shape[1], 0))
            
            # Select best synthetic combo
            curr_synth, validation_cv_score = find_best_synth_combo(curr_real, real_ntgt_all, synth_pool, n_synth_needed=n_synth, ntgt_expected=ntgt_expected)
            cv_scores_by_strat[s_name].append(validation_cv_score * 100)
            
            # Evaluate on independent Test data
            test_acc = evaluate_combination_on_test(curr_real, curr_synth, real_ntgt_all)
            test_scores_by_strat[s_name].append(test_acc)

    mean_cv = {s: np.mean(cv_scores_by_strat[s]) for s in strategies}
    mix_strategies = [s for s in strategies if s.startswith('Mix')]
    best_mix = max(mix_strategies, key=lambda s: mean_cv[s])

    mean_test = {s: np.mean(test_scores_by_strat[s]) for s in strategies}

    log_message(f"\n  === SUMMARY {sid} ===", log_path)
    log_message(f"  Selection Criteria: Strategy with highest internal Validation (CV) score", log_path)
    log_message(f"  Chosen Strategy:    {best_mix} (CV Acc = {mean_cv[best_mix]:.2f}%)", log_path)
    log_message(f"\n  --- Independent Test Accuracy ---", log_path)
    log_message(f"  Baseline:  {mean_test['Baseline']:.2f}%", log_path)
    log_message(f"  SynthOnly: {mean_test['SynthOnly']:.2f}%", log_path)
    log_message(f"  Best Mix:  {mean_test[best_mix]:.2f}%", log_path)

    plot_accuracy(mean_test['Baseline'], mean_test['SynthOnly'], mean_test[best_mix], sid, odir)

    return {
        'sid': sid,
        'bl_test': mean_test['Baseline'],
        'so_test': mean_test['SynthOnly'],
        'best_test': mean_test[best_mix],
        'strategy': best_mix,
    }

def run_p300_pipeline(data_root, subject_dirs, synth_base, conditions, output_base='results_upsampled'):
    mne.set_log_level('WARNING')
    os.makedirs(output_base, exist_ok=True)
    summary_csv = os.path.join(output_base, 'summary.csv')
    
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['subject_condition', 'bl_test', 'so_test', 'best_test', 'strategy'])

    results = []
    print(f"Searching for data in: {data_root}")
    print(f"Searching for synthetic mats in: {synth_base}")
    
    for sd in subject_dirs:
        subj_path = os.path.join(data_root, sd)
        if not os.path.isdir(subj_path):
            print(f"  SKIP: Subject folder not found -> {subj_path}")
            continue

        for cond in conditions:
            try:
                sid = f"{sd}_{cond['matrix']}_{cond['isi']}ms"
                synth_mat = os.path.join(synth_base, f'Subject_{sid}', f'synthetic_{sid}.mat')
                
                if not os.path.exists(synth_mat):
                    print(f"  SKIP: Synthetic file not found -> {synth_mat}")
                    continue

                print(f"\n>>> Processing {sid} ...")
                r = process_p300_upsample(subj_path, sd, cond, output_base, synth_mat)
                
                if r:
                    results.append(r)
                    with open(summary_csv, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            r['sid'], 
                            f"{r['bl_test']:.2f}", 
                            f"{r['so_test']:.2f}", 
                            f"{r['best_test']:.2f}", 
                            r['strategy']
                        ])
            except Exception as e:
                print(f"  ERROR processing {sd} {cond['matrix']}_{cond['isi']}ms:")
                print(traceback.format_exc())

    if results:
        print(f"\n{'='*60}\n  FINAL UPSAMPLED SUMMARY\n{'='*60}")
        print(f"{'Subject':<30} {'BL (test)':>10} {'SO (test)':>10} {'Best(test)':>10} {'Strat':>8}")
        for r in results:
            print(f"{r['sid']:<30} {r['bl_test']:>10.1f} {r['so_test']:>10.1f} {r['best_test']:>10.1f} {r['strategy']:>8}")
    else:
        print("\n[!] No results generated. Check the SKIP messages above to ensure paths are correct.")

if __name__ == '__main__':
    DATA_ROOT = 'data/p300'
    SYNTH_BASE = 'results_P300_vnew'

    TARGET_SUBJECTS = "all"
    if isinstance(TARGET_SUBJECTS, str) and TARGET_SUBJECTS.lower() == "all":
        subject_dirs = [f'H{i}' for i in range(1, 13)]
    else:
        subject_dirs = [f'H{s}' for s in TARGET_SUBJECTS]

    run_p300_pipeline(
        data_root=DATA_ROOT, 
        subject_dirs=subject_dirs, 
        synth_base=SYNTH_BASE, 
        conditions=CONDITIONS, 
        output_base='results_P300_robust'
    )