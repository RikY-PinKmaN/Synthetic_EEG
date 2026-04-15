#!/usr/bin/env python3
"""
P300 cWGAN-GP Data Augmentation Pipeline
=========================================
Replicates the MATLAB preprocessing pipeline (Step_1.m) exactly:
  - 8ch, 500Hz→80Hz, HP 0.4Hz, LP 30Hz, condition-specific notch
  - Average re-reference
  - ERP epochs (-0.1 to 0.8s, 73 samples) with artifact rejection
  - Classification feature epochs (0.15 to 0.5s, 28 samples) with 15-trial averaging
  - Stepwise feature selection + LDA (matching P300_classification.m)
  - QUIC=train, K=val, JUMP=test split

GAN architecture is IDENTICAL to MI pipeline (paradigm-agnostic).
Only the feature loss, filtering, and mix-ratio logic are P300-specific.
"""

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt, iirnotch, welch, buttord, sosfiltfilt, resample_poly
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents tkinter threading crash
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import traceback
import os, csv, time, glob

try:
    from mlxtend.feature_selection import SequentialFeatureSelector
except ImportError:
    pass

import mne

# ── CONSTANTS ────────────────────────────────────────────────────────────────
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# P300 dataset parameters (from MATLAB Step_1.m)
P300_ORIG_FS = 500          # original sampling rate
P300_TARGET_FS = 80         # resampled rate
P300_NCHAN = 8              # Fz, Cz, Pz, C3, C4, P3, P4, Oz
P300_CHAN_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']

# Epoch windows (in seconds, converted to samples at 80 Hz)
P300_ERP_TMIN = -0.1        # ERP epoch start
P300_ERP_TMAX = 0.8         # ERP epoch end
P300_FEAT_TMIN = 0.15       # classification feature start
P300_FEAT_TMAX = 0.5        # classification feature end (exclusive in MATLAB: round(0.5*80)-1)
P300_BL_TMIN = -0.1         # baseline start
P300_BL_TMAX = 0.0          # baseline end
P300_ARTIF_THRESH = 50.0    # µV peak-to-peak artifact threshold

# Derived sample counts at 80 Hz
P300_ERP_SAMPLES = len(range(round(P300_ERP_TMIN * P300_TARGET_FS),
                             round(P300_ERP_TMAX * P300_TARGET_FS) + 1))  # 73
P300_FEAT_SAMPLES = len(range(round(P300_FEAT_TMIN * P300_TARGET_FS),
                              round(P300_FEAT_TMAX * P300_TARGET_FS)))    # 28
P300_N_AVG = 15             # trials per average

# GAN parameters
LATENT_DIM = 50
EMBEDDING_DIM = 25
GAN_EPOCHS = 2000
NUM_RUNS = 5
NUM_BATCHES_PER_RUN = 100
SYNTH_PER_BATCH = 100       # synthetic target trials per batch

# Event codes (from GDF annotations)
EVT_TARGET = 33285
EVT_NONTARGET = 33286
EVT_ROW_END = 32777
EVT_COL_END = 32778

# 3x3 matrix: rows 33025-33027, cols 33028-33030
# 5x5 matrix: rows 33025-33029, cols 33030-33034

# Session structure: s=1(3x3,117ms), s=3(3x3,175ms), s=5(5x5,117ms), s=7(5x5,175ms)
# Each condition uses GDF pair (s, s+1): s=QUICK, s+1=JUMP
SESSION_IDS = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
CONDITIONS = [
    {'s': 1, 'matrix': '3x3', 'isi': 117, 'notch': [4.3, 8.6],  'n_rows': 3, 'n_cols': 3},
    {'s': 3, 'matrix': '3x3', 'isi': 175, 'notch': [5.8, 11.6], 'n_rows': 3, 'n_cols': 3},
    {'s': 5, 'matrix': '5x5', 'isi': 117, 'notch': [4.3, 8.6],  'n_rows': 5, 'n_cols': 5},
    {'s': 7, 'matrix': '5x5', 'isi': 175, 'notch': [5.8, 11.6], 'n_rows': 5, 'n_cols': 5},
]

# Words: QUICK has 5 letters, JUMP has 4
QUICK_N_LETTERS = 5
JUMP_N_LETTERS = 4

# ── LOGGING ──────────────────────────────────────────────────────────────────
_log_path = None

def log_message(msg):
    print(msg)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(str(msg) + '\n')

def set_log_path(path):
    global _log_path
    _log_path = path


# =============================================================================
# 1.  PREPROCESSING (Replicating MATLAB Step_1.m exactly)
# =============================================================================

def design_filters_at_orig_fs(orig_fs=P300_ORIG_FS):
    """Design HP/LP filters at original sampling rate, matching MATLAB exactly.

    MATLAB Step_1.m:
      HP: fdesign.highpass('Fst,Fp,Ast,Ap', 0.4, 0.5, 40, 1, 500) -> design(h,'butter')
      LP: fdesign.lowpass('Fp,Fst,Ap,Ast', 30, 40, 1, 80, 500)    -> design(H,'butter')

    Uses buttord to auto-compute order (matching MATLAB's design() behaviour),
    then SOS form for numerical stability at high orders.
    """
    nyq = orig_fs / 2.0

    # HP: stopband=0.4Hz (40dB atten), passband=0.5Hz (1dB ripple)
    N_hp, Wn_hp = buttord(0.5 / nyq, 0.4 / nyq, 1, 40)
    sos_hp = butter(N_hp, Wn_hp, btype='high', output='sos')

    # LP: passband=30Hz (1dB ripple), stopband=40Hz (80dB atten)
    N_lp, Wn_lp = buttord(30 / nyq, 40 / nyq, 1, 80)
    sos_lp = butter(N_lp, Wn_lp, btype='low', output='sos')

    return sos_hp, sos_lp


def design_notch_filters(notch_freqs, orig_fs=P300_ORIG_FS, n_cascade=3):
    """Design notch filters at stimulus-rate harmonics, matching MATLAB.

    MATLAB: fdesign.notch('N,F0,Q,Ap', 6, freq, 5, 1, 500) -> order-6 notch.
    scipy iirnotch produces order-2. Cascade 3× for order-6 equivalent.

    Returns list of (b, a) tuples — each applied 3× via filtfilt for order-6.
    """
    filters = []
    for nf in notch_freqs:
        if nf < orig_fs / 2:
            b, a = iirnotch(nf, Q=5, fs=orig_fs)
            filters.append((b, a, n_cascade))
    return filters


def load_and_preprocess_gdf(gdf_path, sos_hp, sos_lp, notch_filts, nchan=P300_NCHAN):
    """Load GDF, filter at original fs, return (data, events, orig_fs).
    Matches MATLAB: filter at 500Hz before resampling.

    Uses sosfiltfilt (SOS form) for HP/LP stability at high orders.
    Cascades notch filters 3× for order-6 equivalent.
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, eog=None, verbose=False)
    orig_fs = raw.info['sfreq']
    data = raw.get_data()[:nchan, :].copy()  # (nchan, samples)

    for ch in range(nchan):
        # HP + LP via SOS (stable at high orders)
        data[ch] = sosfiltfilt(sos_hp, data[ch])
        data[ch] = sosfiltfilt(sos_lp, data[ch])
        # Notch: cascade n_cascade times for order matching
        for b_n, a_n, n_casc in notch_filts:
            for _ in range(n_casc):
                data[ch] = filtfilt(b_n, a_n, data[ch])

    # Extract events before resampling
    evs, eid = mne.events_from_annotations(raw, verbose=False)
    id2desc = {v: k for k, v in eid.items()}
    event_codes = np.array([int(id2desc[c]) for c in evs[:, 2]])
    event_latencies = evs[:, 0]  # in samples at orig_fs

    return data, event_codes, event_latencies, orig_fs


def resample_data(data, orig_fs, target_fs=P300_TARGET_FS):
    """Resample continuous data to target_fs using polyphase FIR method.

    Matches MATLAB/EEGLAB: pop_resample uses MATLAB's resample() which is
    polyphase FIR-based. scipy.signal.resample_poly is the equivalent.
    """
    import math
    g = math.gcd(int(target_fs), int(orig_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    resampled = resample_poly(data, up, down, axis=1)
    return resampled.astype(np.float64)


def resample_latencies(latencies, orig_fs, target_fs=P300_TARGET_FS):
    """Convert sample latencies from orig_fs to target_fs."""
    return np.round(latencies * target_fs / orig_fs).astype(int)


def preprocess_condition(subj_path, cond, target_fs=P300_TARGET_FS):
    """Full preprocessing for one condition (two GDF files concatenated).

    Returns:
        EEG: (nchan, total_samples) average-referenced continuous data at target_fs, in µV
        event_codes: concatenated event code array (1-indexed equivalent to MATLAB a1, a2)
        stimes: concatenated latency array at target_fs
        a1_codes, a1_lats: session 1 (QUICK) event codes and latencies
        a2_codes, a2_lats: session 2 (JUMP) event codes and latencies
    """
    s = cond['s']
    s_idx = s - 1  # 0-based index into SESSION_IDS

    gdf1 = os.path.join(subj_path, f'{SESSION_IDS[s_idx]}.gdf')
    gdf2 = os.path.join(subj_path, f'{SESSION_IDS[s_idx + 1]}.gdf')

    if not os.path.exists(gdf1) or not os.path.exists(gdf2):
        return None

    # Design filters at original fs (SOS form for HP/LP, cascaded notch)
    sos_hp, sos_lp = design_filters_at_orig_fs()
    notch_filts = design_notch_filters(cond['notch'])

    # Load and filter at original fs
    data1, codes1, lats1, orig_fs = load_and_preprocess_gdf(gdf1, sos_hp, sos_lp, notch_filts)
    data2, codes2, lats2, _ = load_and_preprocess_gdf(gdf2, sos_hp, sos_lp, notch_filts)

    # Resample to target_fs
    data1_rs = resample_data(data1, orig_fs, target_fs)
    data2_rs = resample_data(data2, orig_fs, target_fs)

    lats1_rs = resample_latencies(lats1, orig_fs, target_fs)
    lats2_rs = resample_latencies(lats2, orig_fs, target_fs)

    # Session 1 event arrays (a1 in MATLAB)
    a1_codes = codes1
    a1_lats = lats1_rs

    # Session 2 event arrays (a2 in MATLAB) — offset latencies by session 1 length
    offset = data1_rs.shape[1]
    a2_codes = codes2
    a2_lats = lats2_rs + offset

    # Concatenate continuous data
    EEG = np.concatenate([data1_rs, data2_rs], axis=1)

    # Convert to µV (MNE loads in V)
    EEG = EEG * 1e6

    # Build combined trigger/latency arrays (matching MATLAB stimes/trigs)
    all_codes = np.concatenate([a1_codes, a2_codes])
    all_lats = np.concatenate([a1_lats, a2_lats])

    # Build target/non-target trigger array (matching MATLAB tot_sti)
    tot_sti = np.zeros(len(all_codes), dtype=int)
    tot_sti[all_codes == EVT_NONTARGET] = 1
    tot_sti[all_codes == EVT_TARGET] = 2

    # Combined stimes (all event latencies) and trigs (target/non-target only)
    stimes = np.round(all_lats).astype(int)
    trigs = tot_sti

    # Average re-reference (MATLAB: EEG = EEG - repmat(mean(EEG(ref,:),1),[numchan,1]))
    EEG = EEG - np.mean(EEG, axis=0, keepdims=True)

    return {
        'EEG': EEG,
        'stimes': stimes,
        'trigs': trigs,
        'a1_codes': a1_codes,
        'a1_lats': a1_lats,
        'a2_codes': a2_codes,
        'a2_lats': a2_lats,
        'all_codes': all_codes,
        'all_lats': all_lats,
    }


# =============================================================================
# 2.  EPOCHING (ERP + Classification Feature Epochs)
# =============================================================================

def epoch_erp_single_trials(EEG, stimes, trigs, fs=P300_TARGET_FS,
                            nchan=P300_NCHAN, artif_thresh=P300_ARTIF_THRESH):
    """Extract single-trial ERP epochs with artifact rejection.
    Matches MATLAB: ep = EEG(OurChans, stimes(n+1)+ts), baseline from stimes(n).

    Returns:
        target_epochs: (nchan, n_erp_samples, n_target)
        nontarget_epochs: (nchan, n_erp_samples, n_nontarget)
    """
    ts = np.arange(round(P300_ERP_TMIN * fs), round(P300_ERP_TMAX * fs) + 1)
    bl_start = round(P300_BL_TMIN * fs)
    bl_end = 0  # inclusive

    target_eps, nontarget_eps = [], []

    # Non-target epochs
    non_tar_idx = np.where(trigs == 1)[0]
    for n in non_tar_idx:
        if n + 1 >= len(stimes):
            continue
        ep_samples = stimes[n + 1] + ts
        bl_samples = np.arange(stimes[n] + bl_start, stimes[n] + bl_end + 1)
        if ep_samples.min() < 0 or ep_samples.max() >= EEG.shape[1]:
            continue
        if bl_samples.min() < 0 or bl_samples.max() >= EEG.shape[1]:
            continue
        ep = EEG[:nchan, ep_samples]
        bl_amp = np.mean(EEG[:nchan, bl_samples], axis=1, keepdims=True)
        ep = ep - bl_amp
        # Artifact rejection: per channel peak-to-peak > threshold
        ptp = ep.max(axis=1) - ep.min(axis=1)
        if np.any(ptp > artif_thresh):
            continue
        nontarget_eps.append(ep)

    # Target epochs
    tar_idx = np.where(trigs == 2)[0]
    for n in tar_idx:
        if n + 1 >= len(stimes):
            continue
        ep_samples = stimes[n + 1] + ts
        bl_samples = np.arange(stimes[n] + bl_start, stimes[n] + bl_end + 1)
        if ep_samples.min() < 0 or ep_samples.max() >= EEG.shape[1]:
            continue
        if bl_samples.min() < 0 or bl_samples.max() >= EEG.shape[1]:
            continue
        ep = EEG[:nchan, ep_samples]
        bl_amp = np.mean(EEG[:nchan, bl_samples], axis=1, keepdims=True)
        ep = ep - bl_amp
        ptp = ep.max(axis=1) - ep.min(axis=1)
        if np.any(ptp > artif_thresh):
            continue
        target_eps.append(ep)

    if not target_eps or not nontarget_eps:
        return None, None

    # Stack: (nchan, T, n_trials)
    target_epochs = np.stack(target_eps, axis=2)
    nontarget_epochs = np.stack(nontarget_eps, axis=2)
    return target_epochs, nontarget_epochs


def extract_row_col_flash_indices(event_codes, n_rows, n_cols):
    """Find indices of row/column flash events in the event code array.
    Filter out invalid flashes (preceded by 32777 for rows, followed by 32778 for cols).

    3x3: rows 33025-33027, cols 33028-33030
    5x5: rows 33025-33029, cols 33030-33034

    Returns dict: {('row', 0): [indices], ('row', 1): ..., ('col', 0): ..., ...}
    """
    rc_indices = {}

    # Row flashes
    for r in range(n_rows):
        code = 33025 + r
        raw_idx = np.where(event_codes == code)[0]
        # Filter: remove if preceding event is 32777 (row-end marker)
        valid = []
        for idx in raw_idx:
            if idx > 0 and event_codes[idx - 1] == EVT_ROW_END:
                continue  # invalid
            valid.append(idx)
        rc_indices[('row', r)] = np.array(valid)

    # Column flashes
    col_start = 33025 + n_rows  # 33028 for 3x3, 33030 for 5x5
    for c in range(n_cols):
        code = col_start + c
        raw_idx = np.where(event_codes == code)[0]
        # Filter: remove if following event is 32778 (column-end marker)
        valid = []
        for idx in raw_idx:
            if idx + 1 < len(event_codes) and event_codes[idx + 1] == EVT_COL_END:
                continue  # invalid
            valid.append(idx)
        rc_indices[('col', c)] = np.array(valid)

    return rc_indices


def build_15avg_classification_features(EEG, stimes, event_codes, all_lats,
                                         rc_indices, n_letters, n_rows, n_cols,
                                         fs=P300_TARGET_FS, nchan=P300_NCHAN):
    """Build 15-trial averaged classification features for a word.

    For each letter, each row/column has 15 repetitions.
    Group into blocks of 15, average, extract classification window.

    Matches MATLAB:
        ep = EEG(OurChans, stimes(n+1) + ts_f)
        BLamp = mean(EEG(OurChans, stimes(n)+BLint(1):stimes(n)+BLint(2)), 2)
        ep = ep - BLamp
        Then take first 15 and average.

    Args:
        EEG: (nchan, total_samples) continuous data
        stimes: all event latencies (combined both sessions)
        event_codes: session-specific event codes (a1 or a2)
        all_lats: session-specific latencies (already offset for session 2)
        rc_indices: from extract_row_col_flash_indices on session-specific codes
        n_letters: number of letters in the word
        n_rows, n_cols: matrix dimensions

    Returns:
        features: (nchan, feat_T, n_averaged_erps) — the p300_feature equivalent
        labels: (n_averaged_erps,) — 1=non-target, 2=target
    """
    ts_f = np.arange(round(P300_FEAT_TMIN * fs), round(P300_FEAT_TMAX * fs))  # 12:39 = 28 samples
    bl_start = round(P300_BL_TMIN * fs)
    bl_end = 0
    n_feat_T = len(ts_f)
    n_rc = n_rows + n_cols  # row/col flashes per letter

    all_m_erp = []
    all_labels = []

    for rc_type in ['row', 'col']:
        n_items = n_rows if rc_type == 'row' else n_cols
        for rc_idx in range(n_items):
            flash_indices = rc_indices.get((rc_type, rc_idx), np.array([]))
            if len(flash_indices) == 0:
                continue

            for letter_idx in range(n_letters):
                start = letter_idx * P300_N_AVG
                end = start + P300_N_AVG
                if end > len(flash_indices):
                    log_message(f"    WARNING: not enough flashes for {rc_type}{rc_idx} letter{letter_idx}: "
                                f"{len(flash_indices)} < {end}")
                    continue

                block = flash_indices[start:end]

                # Epoch each flash in the block
                epochs = []
                for idx_in_session_events in block:
                    # The flash index is in the session event array.
                    # We need the latency in the COMBINED stimes array.
                    # The flash event at idx_in_session_events has latency all_lats[idx_in_session_events]
                    flash_lat = all_lats[idx_in_session_events]

                    # Data epoch: use latency of NEXT event after the flash
                    # In MATLAB: EEG(OurChans, stimes(n+1) + ts_f)
                    # We need to find stimes index corresponding to this flash's latency
                    # and get stimes(that_index + 1)
                    # The stimes array contains ALL event latencies from both sessions.
                    # Find the closest match to flash_lat in stimes
                    stimes_idx = np.argmin(np.abs(stimes - flash_lat))

                    if stimes_idx + 1 >= len(stimes):
                        continue

                    data_lat = stimes[stimes_idx + 1]
                    bl_samples = np.arange(stimes[stimes_idx] + bl_start,
                                           stimes[stimes_idx] + bl_end + 1)
                    ep_samples = data_lat + ts_f

                    if ep_samples.min() < 0 or ep_samples.max() >= EEG.shape[1]:
                        continue
                    if bl_samples.min() < 0 or bl_samples.max() >= EEG.shape[1]:
                        continue

                    ep = EEG[:nchan, ep_samples]
                    bl_amp = np.mean(EEG[:nchan, bl_samples], axis=1, keepdims=True)
                    ep = ep - bl_amp
                    epochs.append(ep)

                if len(epochs) < P300_N_AVG:
                    log_message(f"    WARNING: only {len(epochs)}/{P300_N_AVG} valid epochs for "
                                f"{rc_type}{rc_idx} letter{letter_idx}")
                    if len(epochs) == 0:
                        continue

                # Average across trials (MATLAB: m_erp = mean(erp, 3))
                m_erp = np.mean(np.stack(epochs, axis=2), axis=2)  # (nchan, feat_T)
                all_m_erp.append(m_erp)

                # Label: mode of event codes at (flash_indices - 1) positions
                # MATLAB: l_erp = mode(a1(block_indices - 1))
                preceding_codes = event_codes[block - 1]
                label_code = int(stats.mode(preceding_codes, keepdims=False).mode)
                label = 2 if label_code == EVT_TARGET else 1
                all_labels.append(label)

    if not all_m_erp:
        return None, None

    features = np.stack(all_m_erp, axis=2)  # (nchan, feat_T, n_samples)
    labels = np.array(all_labels, dtype=np.float32)
    return features, labels


def build_15avg_erp_epochs(EEG, stimes, event_codes, all_lats,
                            rc_indices, n_letters, n_rows, n_cols,
                            fs=P300_TARGET_FS, nchan=P300_NCHAN):
    """Same as build_15avg_classification_features but using the full ERP window
    (-0.1 to 0.8s, 73 samples) for GAN training.

    Returns:
        erp_data: (nchan, erp_T, n_averaged_erps) — full ERP epochs
        labels: (n_averaged_erps,)
    """
    ts = np.arange(round(P300_ERP_TMIN * fs), round(P300_ERP_TMAX * fs) + 1)  # 73 samples
    bl_start = round(P300_BL_TMIN * fs)
    bl_end = 0
    artif_thresh = P300_ARTIF_THRESH

    all_erp = []
    all_labels = []

    for rc_type in ['row', 'col']:
        n_items = n_rows if rc_type == 'row' else n_cols
        for rc_idx in range(n_items):
            flash_indices = rc_indices.get((rc_type, rc_idx), np.array([]))
            if len(flash_indices) == 0:
                continue

            for letter_idx in range(n_letters):
                start = letter_idx * P300_N_AVG
                end = start + P300_N_AVG
                if end > len(flash_indices):
                    continue

                block = flash_indices[start:end]

                # Epoch each flash using ERP window
                epochs = []
                for idx_in_session_events in block:
                    flash_lat = all_lats[idx_in_session_events]
                    stimes_idx = np.argmin(np.abs(stimes - flash_lat))

                    if stimes_idx + 1 >= len(stimes):
                        continue

                    data_lat = stimes[stimes_idx + 1]
                    bl_samples = np.arange(stimes[stimes_idx] + bl_start,
                                           stimes[stimes_idx] + bl_end + 1)
                    ep_samples = data_lat + ts

                    if ep_samples.min() < 0 or ep_samples.max() >= EEG.shape[1]:
                        continue
                    if bl_samples.min() < 0 or bl_samples.max() >= EEG.shape[1]:
                        continue

                    ep = EEG[:nchan, ep_samples]
                    bl_amp = np.mean(EEG[:nchan, bl_samples], axis=1, keepdims=True)
                    ep = ep - bl_amp
                    epochs.append(ep)

                if len(epochs) == 0:
                    continue

                # Average across trials
                m_erp = np.mean(np.stack(epochs, axis=2), axis=2)
                all_erp.append(m_erp)

                # Label
                preceding_codes = event_codes[block - 1]
                label_code = int(stats.mode(preceding_codes, keepdims=False).mode)
                label = 2 if label_code == EVT_TARGET else 1
                all_labels.append(label)

    if not all_erp:
        return None, None

    erp_data = np.stack(all_erp, axis=2)
    labels = np.array(all_labels, dtype=np.float32)
    return erp_data, labels


# =============================================================================
# 3.  NORMALIZATION
# =============================================================================

def normalize_per_channel(x, clip_sigma=3.0, norm_stats=None):
    """Per-channel z-score → clip → scale to [-1,1].
    x shape: (nchan, T, trials). Stats computed per channel across time and trials."""
    if norm_stats is None:
        if x.ndim == 3:
            # Compute mean/std per channel across time (axis=1) and trials (axis=2)
            ch_mean = np.mean(x, axis=(1, 2), keepdims=True)  # (nchan, 1, 1)
            ch_std = np.std(x, axis=(1, 2), keepdims=True) + 1e-8
        else:
            ch_mean = np.mean(x, axis=1, keepdims=True)
            ch_std = np.std(x, axis=1, keepdims=True) + 1e-8
    else:
        ch_mean, ch_std = norm_stats

    z = (x - ch_mean) / ch_std
    return np.clip(z / clip_sigma, -1., 1.).astype(np.float32), (ch_mean, ch_std)


# =============================================================================
# 4.  CLASSIFICATION (matching P300_classification.m)
# =============================================================================

def train_p300_lda(features, labels, k_features=5):
    """Stepwise feature selection + shrinkage LDA.

    features: (nchan, feat_T, n_trials) — raw classification features
    labels: (n_trials,) — 1=non-target, 2=target

    Uses shrinkage LDA (Ledoit-Wolf automatic regularisation) instead of
    standard LDA. With small sample sizes (16-40 trials) and high feature
    dimensionality (224 raw features), the sample covariance is severely
    rank-deficient. Shrinkage regularises the covariance estimate, producing
    much more robust classification. This is standard practice in modern
    P300 BCI pipelines (Lotte et al., 2018).

    Returns: (clf, selected_features) or (None, None)
    """
    nchan, feat_T, n_trials = features.shape
    X = features.transpose(2, 1, 0).reshape(n_trials, -1)  # (n_trials, nchan*feat_T)
    y = labels

    if len(np.unique(y)) < 2 or n_trials < 4:
        return None, None

    k = min(k_features, X.shape[1], n_trials - 2)
    if k < 1:
        return None, None

    try:
        # Use shrinkage LDA for feature selection too
        lda_sfs = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        sfs = SequentialFeatureSelector(
            lda_sfs, k_features=k, forward=True, floating=True,
            scoring='accuracy', cv=0, n_jobs=-1)
        sfs.fit(X, y)
        selected = list(sfs.k_feature_idx_)
    except Exception:
        # Fallback: use all features
        selected = list(range(X.shape[1]))

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X[:, selected], y)
    return clf, selected


def evaluate_p300_lda(clf, selected_features, features, labels):
    """Evaluate LDA on test data.

    Returns: (accuracy, confusion_matrix, predictions)
    """
    if clf is None:
        return 0.0, np.zeros((2, 2)), np.array([])

    nchan, feat_T, n_trials = features.shape
    X = features.transpose(2, 1, 0).reshape(n_trials, -1)
    X_sel = X[:, selected_features]

    y_pred = clf.predict(X_sel)
    y_true = labels
    acc = np.mean(y_pred == y_true) * 100
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    return acc, cm, y_pred


# =============================================================================
# 5.  GAN ARCHITECTURE (Paradigm-Agnostic — IDENTICAL to MI)
# =============================================================================

def build_generator(num_channels, time_points, latent_dim, num_classes, embedding_dim):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,))
    merged = layers.Concatenate()([
        noise_input,
        layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))
    ])

    x = layers.Dense(((time_points + 7) // 8) * 64)(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape(((time_points + 7) // 8, 64))(x)

    for filters, k in [(128, 5), (64, 5)]:
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(num_channels, 7, padding="same", activation="tanh")(x)

    tp = x.shape[1]
    if tp > time_points:
        crop_l = (tp - time_points) // 2
        crop_r = (tp - time_points) - crop_l
        x = layers.Cropping1D((crop_l, crop_r))(x)
    elif tp < time_points:
        pad_l = (time_points - tp) // 2
        pad_r = (time_points - tp) - pad_l
        x = layers.ZeroPadding1D((pad_l, pad_r))(x)

    return models.Model([noise_input, label_input], layers.Permute((2, 1))(x))


def build_critic(num_channels, time_points, num_classes, embedding_dim):
    data_input = layers.Input(shape=(num_channels, time_points))
    label_input = layers.Input(shape=(1,))
    x = layers.Permute((2, 1))(data_input)

    for filters in [64, 128, 256]:
        x = layers.Conv1D(filters, 5, strides=2, padding="same")(x)
        if filters > 64:
            x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    merged = layers.Concatenate()([
        layers.Flatten()(x),
        layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))
    ])
    return models.Model([data_input, label_input], layers.Dense(1)(merged))


# =============================================================================
# 6.  LOSS FUNCTIONS (P300-Specific Feature Loss)
# =============================================================================

def frequency_domain_loss(real_data, generated_data):
    real_fft = tf.abs(tf.signal.rfft(tf.transpose(real_data, perm=[0, 2, 1])))
    gen_fft = tf.abs(tf.signal.rfft(tf.transpose(generated_data, perm=[0, 2, 1])))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))


def p300_feature_loss(real_data, fake_data, labels, erp_template,
                      window_start, window_end):
    """P300 feature loss: template matching + class contrast.

    Component 1: Template matching (target only) — match each synthetic target
                 to the grand-average ERP in the P300 window.
    Component 2: Class-conditional amplitude contrast — force generator to
                 produce different temporal patterns per class.

    data shape: (batch, channels, time). labels shape: (batch, 1).
    erp_template shape: (channels, window_T).
    """
    lab_flat = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    target_mask = tf.equal(lab_flat, 1)
    nontarget_mask = tf.equal(lab_flat, 0)

    n_t = tf.maximum(tf.reduce_sum(tf.cast(target_mask, tf.float32)), 1.0)
    n_nt = tf.maximum(tf.reduce_sum(tf.cast(nontarget_mask, tf.float32)), 1.0)

    # Component 1: ERP template matching (target class only)
    fake_tgt = tf.boolean_mask(fake_data, target_mask)
    n_tgt_int = tf.shape(fake_tgt)[0]
    template_loss = tf.cond(
        n_tgt_int > 0,
        lambda: tf.reduce_mean(tf.square(
            fake_tgt[:, :, window_start:window_end] -
            erp_template[tf.newaxis, :, :]
        )),
        lambda: tf.constant(0.0)
    )

    # Component 2: Class-conditional mean amplitude contrast
    real_tgt_amp = tf.reduce_sum(
        tf.reduce_mean(tf.boolean_mask(real_data[:, :, window_start:window_end], target_mask), axis=2),
        axis=0) / n_t
    real_nt_amp = tf.reduce_sum(
        tf.reduce_mean(tf.boolean_mask(real_data[:, :, window_start:window_end], nontarget_mask), axis=2),
        axis=0) / n_nt
    fake_tgt_amp = tf.reduce_sum(
        tf.reduce_mean(tf.boolean_mask(fake_data[:, :, window_start:window_end], target_mask), axis=2),
        axis=0) / n_t
    fake_nt_amp = tf.reduce_sum(
        tf.reduce_mean(tf.boolean_mask(fake_data[:, :, window_start:window_end], nontarget_mask), axis=2),
        axis=0) / n_nt

    real_contrast = real_tgt_amp - real_nt_amp
    fake_contrast = fake_tgt_amp - fake_nt_amp
    contrast_loss = tf.reduce_mean(tf.square(real_contrast - fake_contrast))

    return template_loss + 3.0 * contrast_loss


def gradient_penalty(critic, real_samples, fake_samples, real_labels):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    return 10.0 * tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-10) - 1.0) ** 2)


# =============================================================================
# 7.  TRAINING LOOP
# =============================================================================

def train_cgan_p300(eeg_data, eeg_labels, epochs, batch_size, num_channels,
                    output_dir, subject_id, run_number, time_points,
                    latent_dim=LATENT_DIM, n_critic_steps=5,
                    freq_loss_weight=1.0, feat_loss_weight=3.5,
                    erp_template=None, p300_win_start=None, p300_win_end=None):
    """Train cWGAN-GP for P300 averaged ERPs.

    eeg_data: (n_trials, channels, time) — 15-trial averaged ERPs
    eeg_labels: (n_trials,) — 1=non-target, 2=target
    erp_template: (channels, window_T) — grand-avg target ERP in P300 window
    """
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)[:, np.newaxis]
    gen = build_generator(num_channels, time_points, latent_dim, 2, EMBEDDING_DIM)
    crit = build_critic(num_channels, time_points, 2, EMBEDDING_DIM)
    opt_g = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
    opt_c = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)

    freq_w = tf.constant(freq_loss_weight, dtype=tf.float32)
    feat_w = tf.constant(feat_loss_weight, dtype=tf.float32)
    erp_tf = tf.constant(erp_template, dtype=tf.float32) if erp_template is not None else None
    ws = tf.constant(p300_win_start, dtype=tf.int32) if p300_win_start is not None else None
    we = tf.constant(p300_win_end, dtype=tf.int32) if p300_win_end is not None else None

    @tf.function
    def train_step(rx, ry):
        bs = tf.shape(rx)[0]
        for _ in range(n_critic_steps):
            nz = tf.random.normal([bs, latent_dim])
            with tf.GradientTape() as tape_c:
                fk = gen([nz, ry], training=True)
                d_loss = (tf.reduce_mean(crit([fk, ry], training=True)) -
                          tf.reduce_mean(crit([rx, ry], training=True)) +
                          gradient_penalty(crit, rx, fk, ry))
            opt_c.apply_gradients(zip(
                tape_c.gradient(d_loss, crit.trainable_variables),
                crit.trainable_variables))

        nz = tf.random.normal([bs, latent_dim])
        with tf.GradientTape() as tape_g:
            fk_g = gen([nz, ry], training=True)
            g_w = -tf.reduce_mean(crit([fk_g, ry], training=True))
            g_frq = frequency_domain_loss(rx, fk_g)

            g_feat = tf.constant(0.0)
            if erp_tf is not None and ws is not None:
                g_feat = p300_feature_loss(rx, fk_g, ry, erp_tf, ws, we)

            g_loss = g_w + freq_w * g_frq + feat_w * g_feat
        opt_g.apply_gradients(zip(
            tape_g.gradient(g_loss, gen.trainable_variables),
            gen.trainable_variables))
        return d_loss, g_w, g_frq, g_feat

    best_g_loss, best_weights, patience_cnt = float('inf'), None, 0
    log_message(f"  cWGAN-GP | P300 | S{subject_id} R{run_number}")
    log_message(f"    Data: {eeg_data.shape[0]} trials, {num_channels} ch, {time_points} T | "
                f"batch={batch_size} epochs={epochs}")

    loss_csv = os.path.join(output_dir, f'training_loss_R{run_number}.csv')
    with open(loss_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'critic', 'gen_w', 'gen_freq', 'gen_feat', 'gen_total', 'time_s'])

    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(eeg_data.shape[0])
        n_batches = max(1, eeg_data.shape[0] // batch_size)
        ep_d, ep_gw, ep_gf, ep_gft = 0., 0., 0., 0.

        for bi in range(n_batches):
            s, e = bi * batch_size, (bi + 1) * batch_size
            rx = tf.convert_to_tensor(eeg_data[perm[s:e]], tf.float32)
            ry = tf.convert_to_tensor(labels_0_1[perm[s:e]], tf.int32)
            d, gw, gfrq, gft = train_step(rx, ry)
            ep_d += d.numpy()
            ep_gw += (gw + freq_loss_weight * gfrq + feat_loss_weight * gft).numpy()
            ep_gf += gfrq.numpy()
            ep_gft += gft.numpy()

        ep_d /= n_batches
        g_total = ep_gw / n_batches
        ep_gf /= n_batches
        ep_gft /= n_batches
        wall_t = time.time() - t0
        ep_gw_only = g_total - freq_loss_weight * ep_gf - feat_loss_weight * ep_gft

        with open(loss_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep + 1, f'{ep_d:.4f}', f'{ep_gw_only:.4f}',
                                    f'{ep_gf:.4f}', f'{ep_gft:.4f}', f'{g_total:.4f}',
                                    f'{wall_t:.1f}'])

        if (ep + 1) % 100 == 0 or ep == 0:
            log_message(f"    Ep {ep+1:4d}/{epochs} | D={ep_d:.3f} Gw={ep_gw_only:.3f} "
                        f"Gfrq={ep_gf:.4f} Gfeat={ep_gft:.4f} | {wall_t:.0f}s")

        if ep >= epochs // 4:
            if g_total < best_g_loss - 0.1:
                best_g_loss, best_weights, patience_cnt = g_total, gen.get_weights(), 0
            else:
                patience_cnt += 1
            if patience_cnt >= 500:
                log_message(f"    Early stop at ep {ep+1}")
                break

    if best_weights:
        gen.set_weights(best_weights)
    log_message(f"    Training done. Best G={best_g_loss:.4f} | {time.time()-t0:.0f}s")
    return gen


def generate_synthetic(gen, num_samples, target_label, latent_dim=LATENT_DIM):
    """Generate synthetic averaged ERPs. target_label: 0 or 1 (0-indexed)."""
    if gen is None or num_samples == 0:
        return np.array([])
    synth = gen([tf.random.normal([num_samples, latent_dim]),
                 tf.ones((num_samples, 1), tf.int32) * target_label],
                training=False).numpy()
    return synth


# =============================================================================
# 8.  QUALITY METRICS & FILTERING
# =============================================================================

def compute_psd_correlation(real_data, synth_data, fs=P300_TARGET_FS):
    if real_data.size == 0 or synth_data.size == 0:
        return 0.0
    nperseg = min(32, real_data.shape[2])
    corrs = []
    for ch in range(real_data.shape[1]):
        r_psd = np.mean([welch(real_data[t, ch, :], fs=fs, nperseg=nperseg)[1]
                         for t in range(real_data.shape[0])], axis=0)
        s_psd = np.mean([welch(synth_data[t, ch, :], fs=fs, nperseg=nperseg)[1]
                         for t in range(synth_data.shape[0])], axis=0)
        if np.std(r_psd) > 1e-10 and np.std(s_psd) > 1e-10:
            corrs.append(max(stats.pearsonr(r_psd, s_psd)[0], 0.0))
    return float(np.mean(corrs)) if corrs else 0.0


def compute_amplitude_similarity(real_data, synth_data):
    if real_data.size == 0 or synth_data.size == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean([
        stats.wasserstein_distance(real_data[:, ch, :].ravel(), synth_data[:, ch, :].ravel())
        for ch in range(real_data.shape[1])])))


def compute_erp_correlation(real_data, synth_data, labels_real, labels_synth):
    """ERP waveform correlation for target class (label==2)."""
    if real_data.shape[0] != len(labels_real):
        return 0.0
    if synth_data.shape[0] != len(labels_synth):
        return 0.0
    rt = real_data[labels_real == 2]
    st = synth_data[labels_synth == 2]
    if rt.shape[0] < 1 or st.shape[0] < 1:
        return 0.0
    r_mean = np.mean(rt, axis=0)
    s_mean = np.mean(st, axis=0)
    corrs = []
    for ch in range(r_mean.shape[0]):
        if np.std(r_mean[ch, :]) > 1e-10 and np.std(s_mean[ch, :]) > 1e-10:
            corrs.append(max(stats.pearsonr(r_mean[ch, :], s_mean[ch, :])[0], 0.0))
    return float(np.mean(corrs)) if corrs else 0.0


def compute_fisher_separability(X, y):
    """Fisher discriminant ratio on feature matrix.
    X: (n_trials, n_features), y: (n_trials,)"""
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    m0, m1 = X[y == classes[0]].mean(axis=0), X[y == classes[1]].mean(axis=0)
    v0, v1 = X[y == classes[0]].var(axis=0), X[y == classes[1]].var(axis=0)
    denom = v0 + v1 + 1e-10
    fisher = np.mean((m0 - m1) ** 2 / denom)
    return float(fisher / (1.0 + fisher))


def filter_erp_correlation_gate(synth_gan, synth_y, erp_template,
                                 p300_win_start, p300_win_end,
                                 target_corr_thresh=0.3):
    """Stage 1 filter: reject synthetic target trials with low ERP correlation
    to the real grand-average template, and non-target trials with spurious P300.

    synth_gan: (n_trials, ch, T) — full ERP epochs
    erp_template: (ch, win_T) — real grand-avg target ERP in P300 window
    """
    keep_mask = np.ones(synth_gan.shape[0], dtype=bool)
    template_flat = erp_template.flatten()

    for i in range(synth_gan.shape[0]):
        win = synth_gan[i, :, p300_win_start:p300_win_end]
        if synth_y[i] == 2:  # target
            trial_flat = win.flatten()
            if np.std(trial_flat) < 1e-10:
                keep_mask[i] = False
                continue
            corr = np.corrcoef(trial_flat, template_flat)[0, 1]
            if corr < target_corr_thresh:
                keep_mask[i] = False
        # Non-target: no additional filtering here (feature filter handles it)

    return keep_mask


def filter_feature_distance(real_feats_X, real_feats_y,
                             synth_feats_X, synth_feats_y,
                             keep_ratio=0.6, min_trials=5):
    """Stage 2 filter: reject synthetic trials far from real feature distribution.

    real/synth_feats_X: (n_trials, n_features)
    real/synth_feats_y: (n_trials,)
    """
    keep_mask = np.zeros(len(synth_feats_y), dtype=bool)
    for cls in np.unique(real_feats_y):
        r_idx = real_feats_y == cls
        s_idx = synth_feats_y == cls
        if np.sum(s_idx) == 0:
            continue
        r_mean = np.mean(real_feats_X[r_idx], axis=0)
        r_std = np.std(real_feats_X[r_idx], axis=0) + 1e-8
        scores = np.mean(np.abs(synth_feats_X[s_idx] - r_mean) / r_std, axis=1)
        n_keep = min(max(int(np.sum(s_idx) * keep_ratio), min_trials), np.sum(s_idx))
        best_idx = np.argsort(scores)[:n_keep]
        keep_mask[np.where(s_idx)[0][best_idx]] = True
    return keep_mask


# =============================================================================
# 9.  PLOTTING
# =============================================================================

def plot_accuracy(bl, so, best, sid, odir):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Baseline', 'Synth-Only', 'Best Aug'],
                  [bl, so, best], color=['#2166AC', '#B2182B', '#4DAF4A'])
    ax.set(ylabel='Accuracy (%)', title=f'P300 Accuracy - {sid}', ylim=(0, 110))
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f'{b.get_height():.1f}%', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'accuracy_{sid}.png'), dpi=150)
    plt.close(fig)


def plot_erp_comparison(real_gan, y_real, synth_gan, y_synth,
                        fs, sid, odir):
    """Single-panel ERP at Cz: real target, real non-target, synthetic target."""
    cz_idx = 1  # Cz: [Fz, Cz, Pz, C3, C4, P3, P4, Oz]
    fig, ax = plt.subplots(figsize=(10, 6))
    t_axis = np.arange(real_gan.shape[2]) / fs + P300_ERP_TMIN

    # Real target
    rt = real_gan[y_real == 2]
    if rt.shape[0] >= 1:
        r_mean = np.mean(rt[:, cz_idx, :], axis=0)
        r_sem = np.std(rt[:, cz_idx, :], axis=0) / max(np.sqrt(rt.shape[0]), 1)
        ax.plot(t_axis, r_mean, '#2166AC', lw=2, label=f'Real Target (n={rt.shape[0]})')
        ax.fill_between(t_axis, r_mean - r_sem, r_mean + r_sem, color='#2166AC', alpha=0.15)

    # Real non-target
    rnt = real_gan[y_real == 1]
    if rnt.shape[0] >= 1:
        rnt_mean = np.mean(rnt[:, cz_idx, :], axis=0)
        rnt_sem = np.std(rnt[:, cz_idx, :], axis=0) / max(np.sqrt(rnt.shape[0]), 1)
        ax.plot(t_axis, rnt_mean, '#4DAF4A', lw=2, label=f'Real Non-target (n={rnt.shape[0]})')
        ax.fill_between(t_axis, rnt_mean - rnt_sem, rnt_mean + rnt_sem, color='#4DAF4A', alpha=0.15)

    # Synthetic target
    st = synth_gan[y_synth == 2]
    if st.shape[0] >= 1:
        s_mean = np.mean(st[:, cz_idx, :], axis=0)
        s_sem = np.std(st[:, cz_idx, :], axis=0) / max(np.sqrt(st.shape[0]), 1)
        ax.plot(t_axis, s_mean, '#B2182B', lw=2, ls='--', label=f'Synth Target (n={st.shape[0]})')
        ax.fill_between(t_axis, s_mean - s_sem, s_mean + s_sem, color='#B2182B', alpha=0.15)

    ax.axvline(0, color='k', ls='--', lw=0.5, alpha=0.5)
    ax.axvspan(P300_FEAT_TMIN, P300_FEAT_TMAX, alpha=0.06, color='green', label='P300 window')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (normalized)')
    ax.set_title(f'ERP at Cz - {sid}')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'erp_{sid}.png'), dpi=150)
    plt.close(fig)


def plot_psd_comparison(real_gan, synth_gan, fs, sid, odir):
    fig, ax = plt.subplots(figsize=(10, 5))
    nperseg = min(32, real_gan.shape[2])
    for lbl, xd, color in [('Real', real_gan, '#2166AC'), ('Synthetic', synth_gan, '#B2182B')]:
        all_psd = []
        for t in range(xd.shape[0]):
            ch_psd = [welch(xd[t, c, :], fs=fs, nperseg=nperseg)[1] for c in range(xd.shape[1])]
            all_psd.append(np.mean(ch_psd, axis=0))
        f_axis = welch(xd[0, 0, :], fs=fs, nperseg=nperseg)[0]
        psd_mean = 10 * np.log10(np.mean(all_psd, axis=0) + 1e-15)
        ax.plot(f_axis, psd_mean, color=color, lw=2, label=f'{lbl} ({xd.shape[0]} trials)')
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB)', title=f'PSD - P300 {sid}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'psd_{sid}.png'), dpi=150)
    plt.close(fig)


def plot_training_loss(odir, run_number, sid):
    csv_path = os.path.join(odir, f'training_loss_R{run_number}.csv')
    if not os.path.exists(csv_path):
        return
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=float)
    if data.ndim < 2 or data.shape[0] < 2:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(data[:, 0], data[:, 1], 'b-', alpha=0.7, lw=0.8, label='Critic')
    ax1.set(xlabel='Epoch', ylabel='Loss', title=f'Critic - {sid} R{run_number}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(data[:, 0], data[:, 2], 'r-', alpha=0.7, lw=0.8, label='G Wasserstein')
    ax2.plot(data[:, 0], data[:, 3], 'g-', alpha=0.7, lw=0.8, label='G Freq')
    ax2.plot(data[:, 0], data[:, 4], 'm-', alpha=0.7, lw=0.8, label='G Feature')
    ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Generator - {sid} R{run_number}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'loss_R{run_number}.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# 10. MAIN ORCHESTRATION
# =============================================================================

def process_p300_condition(subj_path, subject_name, cond, output_base):
    """Process one subject × one condition.

    QUIC = train (4 letters), K = val (1 letter), JUMP = test (4 letters).
    GAN trains on QUIC ERP epochs, generates synthetic target averaged ERPs.
    Evaluation uses QUIC/K/JUMP classification features (28-sample window).
    """
    cond_label = f"{cond['matrix']}_{cond['isi']}ms"
    sid = f"{subject_name}_{cond_label}"
    odir = os.path.join(output_base, f'Subject_{sid}')
    os.makedirs(odir, exist_ok=True)
    set_log_path(os.path.join(odir, 'run_log.txt'))

    log_message(f"\n{'='*60}\n  P300 SUBJECT {sid}\n{'='*60}")

    n_rows, n_cols = cond['n_rows'], cond['n_cols']

    # ── Step 1: Preprocessing ──
    log_message("\n--- Step 1: Preprocessing ---")
    result = preprocess_condition(subj_path, cond)
    if result is None:
        log_message(f"  SKIP: GDF files not found for {sid}")
        return None

    EEG = result['EEG']
    stimes = result['stimes']
    trigs = result['trigs']

    # ── Step 1b: Extract single-trial ERP epochs (for visualization) ──
    tgt_erp, ntgt_erp = epoch_erp_single_trials(EEG, stimes, trigs)
    if tgt_erp is not None:
        log_message(f"  Single-trial ERPs: {tgt_erp.shape[2]} target, {ntgt_erp.shape[2]} non-target")

    # ── Step 2: Build 15-trial averaged features ──
    log_message("\n--- Step 2: 15-Trial Averaging + Train/Val/Test Split ---")

    # Session 1 (QUICK): extract row/col flash indices
    a1_rc = extract_row_col_flash_indices(result['a1_codes'], n_rows, n_cols)
    # Session 2 (JUMP): extract row/col flash indices
    a2_rc = extract_row_col_flash_indices(result['a2_codes'], n_rows, n_cols)

    # QUICK classification features (letters 1-5)
    quick_feat, quick_y = build_15avg_classification_features(
        EEG, stimes, result['a1_codes'], result['a1_lats'],
        a1_rc, QUICK_N_LETTERS, n_rows, n_cols)

    # JUMP classification features (letters 1-4)
    jump_feat, jump_y = build_15avg_classification_features(
        EEG, stimes, result['a2_codes'], result['a2_lats'],
        a2_rc, JUMP_N_LETTERS, n_rows, n_cols)

    if quick_feat is None or jump_feat is None:
        log_message(f"  SKIP: Failed to build averaged features for {sid}")
        return None

    # QUICK ERP epochs (full window for GAN)
    quick_erp, quick_erp_y = build_15avg_erp_epochs(
        EEG, stimes, result['a1_codes'], result['a1_lats'],
        a1_rc, QUICK_N_LETTERS, n_rows, n_cols)

    if quick_erp is None:
        log_message(f"  SKIP: Failed to build ERP epochs for {sid}")
        return None

    # Verify ERP and classification features have same sample count
    # (they should, but the longer ERP window might cause boundary mismatches)
    if quick_erp.shape[2] != quick_feat.shape[2]:
        log_message(f"  WARNING: ERP/feat count mismatch: erp={quick_erp.shape[2]} feat={quick_feat.shape[2]}")
        n_min = min(quick_erp.shape[2], quick_feat.shape[2])
        quick_erp = quick_erp[:, :, :n_min]
        quick_erp_y = quick_erp_y[:n_min]
        quick_feat = quick_feat[:, :, :n_min]
        quick_y = quick_y[:n_min]
        log_message(f"  Truncated to {n_min} samples")

    # Split QUICK into QUIC (train, letters 0-3) and K (val, letter 4)
    n_rc = n_rows + n_cols  # row/col flashes per letter
    quic_n = 4 * n_rc   # first 4 letters
    k_n = 1 * n_rc      # last letter

    if quick_feat.shape[2] < quic_n + k_n:
        log_message(f"  SKIP: Not enough QUICK samples ({quick_feat.shape[2]} < {quic_n + k_n})")
        return None

    train_feat = quick_feat[:, :, :quic_n]
    train_y = quick_y[:quic_n]
    val_feat = quick_feat[:, :, quic_n:quic_n + k_n]
    val_y = quick_y[quic_n:quic_n + k_n]
    test_feat = jump_feat
    test_y = jump_y

    # ERP epochs: same split for GAN training
    train_erp = quick_erp[:, :, :quic_n].copy()
    train_erp_y = quick_erp_y[:quic_n].copy()

    n_tgt_train = np.sum(train_y == 2)
    n_ntgt_train = np.sum(train_y == 1)
    n_tgt_val = np.sum(val_y == 2)
    n_ntgt_val = np.sum(val_y == 1)
    n_tgt_test = np.sum(test_y == 2)
    n_ntgt_test = np.sum(test_y == 1)

    log_message(f"  QUIC features: {train_feat.shape} | {n_tgt_train} target, {n_ntgt_train} non-target")
    log_message(f"  K features:    {val_feat.shape} | {n_tgt_val} target, {n_ntgt_val} non-target")
    log_message(f"  JUMP features: {test_feat.shape} | {n_tgt_test} target, {n_ntgt_test} non-target")
    log_message(f"  Train ERP (GAN): {train_erp.shape}")

    # ── Step 3: Baseline (Downsampled Balanced, QUIC→JUMP) ──
    log_message("\n--- Step 3: Baseline (Downsampled Balanced, QUIC→JUMP) ---")

    # Use full QUICK (train+val) for baseline training, test on JUMP
    quic_full_feat = quick_feat
    quic_full_y = quick_y

    n_tgt_quic = int(np.sum(quic_full_y == 2))
    n_ntgt_quic = int(np.sum(quic_full_y == 1))
    log_message(f"  Full QUIC: {n_tgt_quic} target, {n_ntgt_quic} non-target")

    # Random undersampling: downsample non-target to match target count
    # Average over N_BL_DRAWS random draws for stability
    N_BL_DRAWS = 10
    tgt_idx_quic = np.where(quic_full_y == 2)[0]
    ntgt_idx_quic = np.where(quic_full_y == 1)[0]

    bl_accs = []
    bl_cms = []
    for draw in range(N_BL_DRAWS):
        # Randomly select n_tgt non-target samples
        rng = np.random.RandomState(SEED_VALUE + draw)
        ntgt_subsample = rng.choice(ntgt_idx_quic, size=n_tgt_quic, replace=False)
        bal_idx = np.sort(np.concatenate([tgt_idx_quic, ntgt_subsample]))

        bal_feat = quic_full_feat[:, :, bal_idx]
        bal_y = quic_full_y[bal_idx]

        clf_draw, sel_draw = train_p300_lda(bal_feat, bal_y, k_features=5)
        acc_draw, cm_draw, _ = evaluate_p300_lda(clf_draw, sel_draw, test_feat, test_y)
        bl_accs.append(acc_draw)
        bl_cms.append(cm_draw)

    bl_acc = np.mean(bl_accs)
    bl_std = np.std(bl_accs)
    bl_cm = np.mean(bl_cms, axis=0).astype(int)
    log_message(f"  Baseline ({N_BL_DRAWS} draws): {bl_acc:.2f}% ± {bl_std:.2f}%")
    log_message(f"  Per-draw: {[f'{a:.1f}' for a in bl_accs]}")
    log_message(f"  Mean confusion matrix:\n{bl_cm}")

    # ── Step 4: Normalize & prepare GAN data ──
    log_message("\n--- Step 4: GAN Training ---")

    # Normalize ERP epochs for GAN (per-channel z-score)
    train_erp_norm, norm_stats = normalize_per_channel(train_erp)
    ch_mean, ch_std = norm_stats

    # GAN input: (n_trials, ch, T) — from (nchan, erp_T, n_trials)
    x_gan = train_erp_norm.transpose(2, 0, 1)  # (trials, ch, T)
    y_gan = train_erp_y.copy()  # explicit copy — prevent view aliasing issues
    erp_T = x_gan.shape[2]  # 73 samples

    assert x_gan.shape[0] == len(y_gan), \
        f"Shape mismatch: x_gan={x_gan.shape[0]} trials, y_gan={len(y_gan)} labels"
    assert x_gan.shape[1] == P300_NCHAN, \
        f"Channel mismatch: x_gan has {x_gan.shape[1]} ch, expected {P300_NCHAN}"

    log_message(f"  GAN input: {x_gan.shape} | y_gan: {y_gan.shape} "
                f"({np.sum(y_gan==2):.0f} tgt, {np.sum(y_gan==1):.0f} nt)")

    # Integer counts for consistent use throughout
    n_tgt_int = int(n_tgt_train)         # QUIC train target count (8 for 3x3)
    n_ntgt_int = int(n_ntgt_train)       # QUIC train non-target count (16 for 3x3)
    n_quic_tgt = int(np.sum(quick_y == 2))  # QUICK full target count (10 for 3x3)
    log_message(f"  Class counts: n_tgt_train={n_tgt_int}, n_ntgt_train={n_ntgt_int}, "
                f"n_quick_tgt={n_quic_tgt}")

    # P300 window indices in the ERP epoch (for feature loss)
    # The ERP epoch starts at -0.1s. The P300 window is 0.15-0.5s.
    erp_ts = np.arange(round(P300_ERP_TMIN * P300_TARGET_FS),
                       round(P300_ERP_TMAX * P300_TARGET_FS) + 1)
    p300_win_start = np.searchsorted(erp_ts, round(P300_FEAT_TMIN * P300_TARGET_FS))
    p300_win_end = np.searchsorted(erp_ts, round(P300_FEAT_TMAX * P300_TARGET_FS))

    # ERP template: grand-avg of normalized target training ERPs in P300 window
    tgt_mask_gan = y_gan == 2
    tgt_erp_norm = x_gan[tgt_mask_gan]  # (n_target, ch, T)
    erp_template = np.mean(tgt_erp_norm[:, :, p300_win_start:p300_win_end], axis=0)  # (ch, win_T)
    log_message(f"  ERP template: {erp_template.shape} from {np.sum(tgt_mask_gan)} target trials")
    log_message(f"  P300 window in ERP: [{p300_win_start}:{p300_win_end}] "
                f"({P300_FEAT_TMIN}-{P300_FEAT_TMAX}s)")

    # ── Feature selection on training set (computed once, reused) ──
    clf_train, sel_train = train_p300_lda(train_feat, train_y, k_features=5)
    if sel_train is None:
        log_message("  WARNING: Feature selection failed, using all features")
        sel_train = list(range(train_feat.shape[0] * train_feat.shape[1]))
    log_message(f"  Selected features: {len(sel_train)} of {train_feat.shape[0] * train_feat.shape[1]}")

    # Precompute real training features for filtering
    n_ch_f, n_T_f, n_tr_f = train_feat.shape
    real_train_X = train_feat.transpose(2, 1, 0).reshape(n_tr_f, -1)[:, sel_train]

    # ── GAN runs ──
    log_message(f"\n  {NUM_RUNS} GAN runs × {NUM_BATCHES_PER_RUN} batches")
    all_batches = []

    for run_idx in range(NUM_RUNS):
        gen = train_cgan_p300(
            x_gan, y_gan,
            epochs=GAN_EPOCHS,
            batch_size=min(16, x_gan.shape[0]),
            num_channels=P300_NCHAN,
            output_dir=odir,
            subject_id=sid,
            run_number=run_idx + 1,
            time_points=erp_T,
            freq_loss_weight=1.0,
            feat_loss_weight=3.5,
            erp_template=erp_template,
            p300_win_start=p300_win_start,
            p300_win_end=p300_win_end
        )
        if gen is None:
            continue

        plot_training_loss(odir, run_idx + 1, sid)

        log_message(f"  Generating & filtering batches (Run {run_idx+1})...")
        for bi in range(NUM_BATCHES_PER_RUN):
            # Generate TARGET only — non-target is always real data
            sc_tgt = generate_synthetic(gen, SYNTH_PER_BATCH, 1)   # target
            if sc_tgt.size == 0:
                continue

            sy_tgt = np.ones(sc_tgt.shape[0]) * 2   # label=2 (target)

            # ERP correlation gate (target only) — lenient threshold
            # With only 8 real target trials the template is noisy, so 0.1 catches
            # only truly garbage trials while preserving natural variability
            keep1 = filter_erp_correlation_gate(
                sc_tgt, sy_tgt, erp_template, p300_win_start, p300_win_end,
                target_corr_thresh=0.1)
            sc_tgt = sc_tgt[keep1]
            sy_tgt = sy_tgt[keep1]
            # Require at least n_tgt_int (8) surviving trials
            if sc_tgt.shape[0] < n_tgt_int:
                continue

            # Keep exactly n_tgt_int synthetic target trials
            f_sg = sc_tgt[:n_tgt_int]
            f_sy = sy_tgt[:n_tgt_int]

            # Quality metrics — compare synthetic target vs real target
            _y_real = y_gan[:x_gan.shape[0]] if len(y_gan) > x_gan.shape[0] else y_gan
            real_tgt_data = x_gan[_y_real == 2]
            psd_c = compute_psd_correlation(real_tgt_data, f_sg, fs=P300_TARGET_FS) if real_tgt_data.shape[0] > 0 else 0.0
            amp_s = compute_amplitude_similarity(real_tgt_data, f_sg) if real_tgt_data.shape[0] > 0 else 0.0
            erp_c = compute_erp_correlation(x_gan, f_sg, _y_real, f_sy)

            synth_feat_win = f_sg[:, :, p300_win_start:p300_win_end]
            f_feat_flat = synth_feat_win.reshape(f_sg.shape[0], -1)
            fisher = compute_fisher_separability(
                np.vstack([real_train_X, f_feat_flat[:, :real_train_X.shape[1]]
                           if f_feat_flat.shape[1] >= real_train_X.shape[1]
                           else f_feat_flat]),
                np.concatenate([train_y, f_sy])
            ) if f_feat_flat.shape[0] > 0 else 0.0

            combined = 0.30 * psd_c + 0.30 * amp_s + 0.25 * erp_c + 0.15 * fisher

            log_message(f"    R{run_idx+1} B{bi+1}: filtered={sc_tgt.shape[0]}/{SYNTH_PER_BATCH} "
                        f"kept={f_sg.shape[0]} "
                        f"psd={psd_c:.3f} amp={amp_s:.3f} erp={erp_c:.3f} fisher={fisher:.3f} "
                        f"score={combined:.3f}")

            all_batches.append({
                'run': run_idx + 1, 'batch': bi + 1,
                'combined_score': combined,
                'synth_tgt_gan': f_sg,       # filtered target (trials, ch, T)
                'synth_tgt_y': f_sy,
            })

    if not all_batches:
        log_message("  CRITICAL: No valid batches produced")
        return None

    # ── Step 5: Strategy Selection ──
    log_message(f"\n--- Step 5: Strategy Selection ({len(all_batches)} batches) ---")
    all_batches.sort(key=lambda b: b['combined_score'], reverse=True)
    top_batches = all_batches[:10]

    # Strategy: progressively add synthetic targets while including more real
    # non-target to maintain balance. NO synthetic non-target anywhere.
    # Max synthetic = N_tgt (match real target count, NOT deficit).
    #
    # For 3x3: n_tgt=8, n_ntgt=16
    #   Baseline:  8 real tgt + 8 real ntgt (downsampled)
    #   SynthOnly: 8 synth tgt + 8 real ntgt (downsampled)
    #   Mix+2:     8 real tgt + 2 synth tgt = 10 tgt, 10 real ntgt
    #   Mix+4:     8 real tgt + 4 synth tgt = 12 tgt, 12 real ntgt
    #   Mix+6:     8 real tgt + 6 synth tgt = 14 tgt, 14 real ntgt
    #   Mix+8:     8 real tgt + 8 synth tgt = 16 tgt, 16 real ntgt ← MAX

    max_synth = n_tgt_int  # cap: never add more synthetic than real target count

    # Build mix configs: (name, n_synth_tgt, n_real_ntgt)
    mix_steps = list(range(2, max_synth + 1, 2))  # 2, 4, 6, 8
    if max_synth not in mix_steps:
        mix_steps.append(max_synth)

    mix_configs = []
    for n_syn in mix_steps:
        total_tgt = n_tgt_int + n_syn
        n_real_ntgt = min(total_tgt, n_ntgt_int)  # match, capped at available
        name = f'Mix+{n_syn}'
        mix_configs.append((name, n_syn, n_real_ntgt))

    log_message(f"  Mix configs: {[(c[0], f'{n_tgt_int}+{c[1]}t vs {c[2]}nt') for c in mix_configs]}")

    # Pre-select real non-target indices for each draw
    ntgt_idx_train = np.where(train_y == 1)[0]
    tgt_idx_train = np.where(train_y == 2)[0]

    all_strats = []
    for b in top_batches:
        sg_tgt = b['synth_tgt_gan']
        sy_tgt_b = b['synth_tgt_y']
        n_avail_tgt = sg_tgt.shape[0]

        # SynthOnly: N_tgt synth target + N_tgt real non-target (downsampled)
        na_so = min(n_tgt_int, n_avail_tgt)
        rng_so = np.random.RandomState(SEED_VALUE)
        ntgt_sub_so = rng_so.choice(ntgt_idx_train, size=na_so, replace=False)
        so_synth_cls = sg_tgt[:na_so, :, p300_win_start:p300_win_end].transpose(1, 2, 0)
        so_real_ntgt_cls = train_feat[:, :, ntgt_sub_so]
        so_feat = np.concatenate([so_synth_cls, so_real_ntgt_cls], axis=2)
        so_y = np.concatenate([sy_tgt_b[:na_so], train_y[ntgt_sub_so]])

        if len(np.unique(so_y)) >= 2:
            clf_so, sel_so = train_p300_lda(so_feat, so_y, k_features=5)
            va_so, _, _ = evaluate_p300_lda(clf_so, sel_so, val_feat, val_y) if clf_so else (0, None, None)
            train_pred_so = clf_so.predict(
                so_feat.transpose(2, 1, 0).reshape(so_feat.shape[2], -1)[:, sel_so]) if clf_so else np.array([])
            train_acc_so = np.mean(train_pred_so == so_y) * 100 if len(train_pred_so) > 0 else 0
            quality_so = b['combined_score']
            score_so = 0.4 * (va_so / 100.0) + 0.3 * (train_acc_so / 100.0) + 0.3 * quality_so
            all_strats.append({
                'name': 'SynthOnly', 'val_acc': va_so, 'train_acc': train_acc_so,
                'strategy_score': score_so, 'batch': b, 'n_synth': na_so,
                'synth_only': True
            })

        # Mix strategies: add n_syn synthetic targets + n_real_ntgt real non-target
        for name, n_syn, n_real_ntgt in mix_configs:
            na_t = min(n_syn, n_avail_tgt)
            if na_t == 0:
                continue

            # Real target (all) + synthetic target
            synth_tgt_cls = sg_tgt[:na_t, :, p300_win_start:p300_win_end].transpose(1, 2, 0)

            # Real non-target: randomly select n_real_ntgt from available
            rng_mix = np.random.RandomState(SEED_VALUE + n_syn)
            na_nt = min(n_real_ntgt, len(ntgt_idx_train))
            ntgt_sub = rng_mix.choice(ntgt_idx_train, size=na_nt, replace=False)

            eval_feat = np.concatenate([
                train_feat[:, :, tgt_idx_train],   # all real target
                synth_tgt_cls,                      # synthetic target
                train_feat[:, :, ntgt_sub],         # subsampled real non-target
            ], axis=2)
            eval_y = np.concatenate([
                train_y[tgt_idx_train],
                sy_tgt_b[:na_t],
                train_y[ntgt_sub],
            ])

            if len(np.unique(eval_y)) < 2:
                continue

            clf_s, sel_s = train_p300_lda(eval_feat, eval_y, k_features=5)
            va, _, _ = evaluate_p300_lda(clf_s, sel_s, val_feat, val_y) if clf_s else (0, None, None)

            if clf_s is not None:
                train_pred = clf_s.predict(
                    eval_feat.transpose(2, 1, 0).reshape(eval_feat.shape[2], -1)[:, sel_s])
                train_acc = np.mean(train_pred == eval_y) * 100
            else:
                train_acc = 0.0

            quality_score = b['combined_score']
            strategy_score = 0.4 * (va / 100.0) + 0.3 * (train_acc / 100.0) + 0.3 * quality_score

            all_strats.append({
                'name': name, 'val_acc': va, 'train_acc': train_acc,
                'strategy_score': strategy_score, 'batch': b,
                'n_synth': na_t, 'n_real_ntgt': na_nt, 'synth_only': False
            })

    if not all_strats:
        log_message("  CRITICAL: No valid strategies")
        return None

    # Sort and select — exclude SynthOnly from "best"
    aug_strats = [s for s in all_strats if not s['synth_only']]
    all_strats_sorted = sorted(all_strats, key=lambda s: s['strategy_score'], reverse=True)

    for s in all_strats_sorted[:12]:
        n_nt_str = f" ntgt={s.get('n_real_ntgt','?')}" if not s['synth_only'] else ""
        log_message(f"  {s['name']:12s} val={s['val_acc']:.1f}% train={s['train_acc']:.1f}% "
                    f"synth_tgt={s['n_synth']}{n_nt_str} "
                    f"composite={s['strategy_score']:.3f} (R{s['batch']['run']}B{s['batch']['batch']})")

    if aug_strats:
        aug_strats.sort(key=lambda s: s['strategy_score'], reverse=True)
        best = aug_strats[0]
    else:
        best = all_strats_sorted[0]
    log_message(f"\n  SELECTED: {best['name']} composite={best['strategy_score']:.3f}")

    # ── Step 6: Final Test (→ JUMP) ──
    log_message("\n--- Step 6: Final Test (→ JUMP) ---")

    # SynthOnly test: 8 synth target + 8 real non-target (downsampled)
    so_strats = [s for s in all_strats if s['synth_only']]
    so_acc = 0.0
    if so_strats:
        best_so = max(so_strats, key=lambda s: s['strategy_score'])
        b_so = best_so['batch']
        na_so = min(n_tgt_int, b_so['synth_tgt_gan'].shape[0])
        rng_so_test = np.random.RandomState(SEED_VALUE + 999)
        ntgt_idx_quic = np.where(quick_y == 1)[0]
        ntgt_sub_so_test = rng_so_test.choice(ntgt_idx_quic, size=na_so, replace=False)
        so_feat_test = np.concatenate([
            b_so['synth_tgt_gan'][:na_so, :, p300_win_start:p300_win_end].transpose(1, 2, 0),
            quick_feat[:, :, ntgt_sub_so_test],
        ], axis=2)
        so_y_test = np.concatenate([b_so['synth_tgt_y'][:na_so], quick_y[ntgt_sub_so_test]])
        clf_so_test, sel_so_test = train_p300_lda(so_feat_test, so_y_test, k_features=5)
        if clf_so_test:
            so_acc, _, _ = evaluate_p300_lda(clf_so_test, sel_so_test, test_feat, test_y)

    # Best augmentation strategy: replicate the SELECTED mix ratio for final test
    # Final test uses FULL QUIC (train+val) as real data pool (not just train)
    b_best = best['batch']
    sg_best_tgt = b_best['synth_tgt_gan']  # (N_tgt, ch, T) — full batch
    sy_best_tgt = b_best['synth_tgt_y']

    n_synth_best = best['n_synth']  # e.g. 2 for Mix+2, 8 for Mix+8

    # Real pools from FULL QUIC (train + val = all 5 letters of QUICK)
    quic_tgt_idx = np.where(quick_y == 2)[0]   # 10 for 3x3
    quic_ntgt_idx = np.where(quick_y == 1)[0]  # 20 for 3x3

    # Total target = all real QUIC target + n_synth_best synthetic
    total_tgt_final = len(quic_tgt_idx) + n_synth_best
    # Non-target: match total target count, capped at available
    na_ntgt_final = min(total_tgt_final, len(quic_ntgt_idx))
    rng_final = np.random.RandomState(SEED_VALUE + 777)
    ntgt_sub_final = rng_final.choice(quic_ntgt_idx, size=na_ntgt_final, replace=False)

    final_feat = np.concatenate([
        quick_feat[:, :, quic_tgt_idx],          # ALL real QUIC target (train+val)
        sg_best_tgt[:n_synth_best, :, p300_win_start:p300_win_end].transpose(1, 2, 0),
        quick_feat[:, :, ntgt_sub_final],         # real non-target (balanced)
    ], axis=2)
    final_y = np.concatenate([
        quick_y[quic_tgt_idx],
        sy_best_tgt[:n_synth_best],
        quick_y[ntgt_sub_final],
    ])

    clf_final, sel_final = train_p300_lda(final_feat, final_y, k_features=5)
    best_acc, best_cm, _ = evaluate_p300_lda(clf_final, sel_final, test_feat, test_y) \
        if clf_final else (0, np.zeros((2, 2)), np.array([]))

    log_message(f"\n  === SUMMARY {sid} ===")
    log_message(f"  Baseline={bl_acc:.2f}%  SynthOnly={so_acc:.2f}%  Best({best['name']})={best_acc:.2f}%")
    log_message(f"  Final train set: {int(np.sum(final_y==2))} tgt + {int(np.sum(final_y==1))} ntgt")
    log_message(f"  Confusion Matrix:\n{best_cm}")

    # ── Plots ──
    plot_accuracy(bl_acc, so_acc, best_acc, sid, odir)

    # ERP: single panel with 3 lines at Cz
    #   Real Target (all N_tgt, blue)
    #   Real Non-target (all N_ntgt, green)
    #   Synthetic Target (exactly N_tgt, red dashed) — with gain correction for display
    _y_real_plot = y_gan[:x_gan.shape[0]]
    sg_plot = sg_best_tgt[:n_tgt_int].copy()  # exactly N_tgt synthetic
    real_tgt_data = x_gan[_y_real_plot == 2]
    if real_tgt_data.shape[0] >= 2 and sg_plot.shape[0] >= 1:
        for ch in range(sg_plot.shape[1]):
            r_std = np.std(real_tgt_data[:, ch, :])
            s_std = np.std(sg_plot[:, ch, :]) + 1e-8
            sg_plot[:, ch, :] *= (r_std / s_std)

    plot_erp_comparison(x_gan, _y_real_plot, sg_plot,
                        np.ones(sg_plot.shape[0]) * 2,
                        fs=P300_TARGET_FS, sid=sid, odir=odir)
    plot_psd_comparison(real_tgt_data, sg_plot,
                        fs=P300_TARGET_FS, sid=sid, odir=odir)

    # ── Save (include real data for visualization script) ──
    scipy.io.savemat(os.path.join(odir, f'synthetic_{sid}.mat'), {
        'synthetic_tgt_x': sg_best_tgt[:n_tgt_int],
        'synthetic_tgt_y': sy_best_tgt[:n_tgt_int],
        'real_x': x_gan,                              # all real training data (trials, ch, T)
        'real_y': y_gan[:x_gan.shape[0]],             # real labels
        'norm_ch_mean': ch_mean,
        'norm_ch_std': ch_std,
        'baseline_acc': bl_acc,
        'synth_only_acc': so_acc,
        'best_aug_acc': best_acc,
        'best_strategy': best['name'],
    })
    log_message(f"  Saved: synthetic_{sid}.mat "
                f"({sg_best_tgt[:n_tgt_int].shape[0]} synth tgt + "
                f"{x_gan.shape[0]} real trials)")

    return {
        'sid': sid, 'baseline': bl_acc,
        'synth_only': so_acc, 'best': best_acc,
        'strategy': best['name']
    }


# =============================================================================
# 11. ENTRY POINT
# =============================================================================

def run_p300_pipeline(data_root, subject_dirs, conditions=None, output_base='results_P300_v3'):
    """Run the full P300 pipeline across subjects and conditions.

    Args:
        data_root: path containing H1/, H2/, ..., H12/ folders
        subject_dirs: list of subject folder names, e.g. ['H1', 'H2', ...]
        conditions: list of condition dicts to process (default: all 4)
        output_base: output directory
    """
    mne.set_log_level('WARNING')

    if conditions is None:
        conditions = CONDITIONS

    os.makedirs(output_base, exist_ok=True)
    summary_csv = os.path.join(output_base, 'summary_results.csv')
    with open(summary_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['subject_condition', 'baseline_acc', 'synth_only_acc',
                                 'best_aug_acc', 'best_strategy'])

    results = []
    for sd in subject_dirs:
        subj_path = os.path.join(data_root, sd)
        if not os.path.isdir(subj_path):
            print(f"  SKIP: {subj_path} not found")
            continue

        for cond in conditions:
            try:
                r = process_p300_condition(subj_path, sd, cond, output_base)
                if r:
                    results.append(r)
                    with open(summary_csv, 'a', newline='') as f:
                        csv.writer(f).writerow([r['sid'], f"{r['baseline']:.2f}",
                                                 f"{r['synth_only']:.2f}", f"{r['best']:.2f}",
                                                 r['strategy']])
            except Exception as e:
                print(f"  ERROR {sd} {cond['matrix']}_{cond['isi']}ms: {traceback.format_exc()}")

    if results:
        print(f"\n{'='*60}\n  P300 FINAL SUMMARY\n{'='*60}")
        for r in results:
            print(f"  {r['sid']:30s} BL={r['baseline']:.2f}%  SO={r['synth_only']:.2f}%  "
                  f"Best={r['best']:.2f}% [{r['strategy']}]")
        bl_mean = np.mean([r['baseline'] for r in results])
        best_mean = np.mean([r['best'] for r in results])
        print(f"  Mean: BL={bl_mean:.2f}%  Best={best_mean:.2f}%")

    return results


# =============================================================================
# 12. __main__
# =============================================================================

if __name__ == '__main__':
    DATA_ROOT = 'data/p300'

    # ── Subject selection ──
    # List of 1-based subject numbers, or "all" for all 12
    # Examples: [1], [1, 5, 12], "all"
    TARGET_SUBJECTS = [1]

    if isinstance(TARGET_SUBJECTS, str) and TARGET_SUBJECTS.lower() == "all":
        subject_dirs = [f'H{i}' for i in range(1, 13)]
    else:
        subject_dirs = [f'H{s}' for s in TARGET_SUBJECTS]

    # ── Pair/condition selection ──
    # Each "pair" is a condition using two GDF sessions:
    #   Pair 0 = H01+H02 = 3×3 matrix, 117ms ISI
    #   Pair 1 = H03+H04 = 3×3 matrix, 175ms ISI
    #   Pair 2 = H05+H06 = 5×5 matrix, 117ms ISI
    #   Pair 3 = H07+H08 = 5×5 matrix, 175ms ISI
    # None = all 4 pairs, or specify list of indices
    # Examples: None, [0], [0, 2], [1, 3]
    TARGET_PAIRS = None

    if TARGET_PAIRS is not None:
        conds = [CONDITIONS[i] for i in TARGET_PAIRS]
    else:
        conds = None

    run_p300_pipeline(
        data_root=DATA_ROOT,
        subject_dirs=subject_dirs,
        conditions=conds,
        output_base='results_P300_v3'
    )