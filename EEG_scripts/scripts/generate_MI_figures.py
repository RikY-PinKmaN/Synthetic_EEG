#!/usr/bin/env python3
"""
EEG cWGAN-GP — Combined Visualization v5
==========================================
Multi-dataset support:
  - results_MI_BCI4       → BCI Competition IV 2a (9 subjects)
  - results_MI_DATA1..5   → Custom MI dataset split into 5 parts

Usage:
  python generate_mi_bci2a_figures.py                        # All datasets, all subjects
  python generate_mi_bci2a_figures.py --datasets BCI4        # Only BCI4
  python generate_mi_bci2a_figures.py --datasets DATA1 DATA3 # Only DATA1 + DATA3
  python generate_mi_bci2a_figures.py --datasets BCI4 -s 1 3 7
  python generate_mi_bci2a_figures.py -o my_output_dir

Changes from v4:
  - Multi-dataset: BCI4 + DATA1-DATA5 with separate and combined views
  - 50 trials per class for real data (up from 20)
  - Balanced: synthetic subsampled to match real trial count (50/class) at load time
    so ALL plots (topoplots, grand averages, PSD, TFR, distribution) use balanced data
  - New data distribution plot: amplitude histogram, per-channel std, PSD overlay
  - Gain correction applied on full synthetic before subsampling for stable stats
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import welch, ellip, ellipord, filtfilt
from scipy.stats import pearsonr, wilcoxon, ttest_rel
import mne

mne.set_log_level('ERROR')

# ── Configuration ─────────────────────────────────────────────────────────────
TRIALS_PER_CLASS = 50  # Number of trials per class for both real and synthetic

# ── Per-dataset channel/signal configurations ────────────────────────────────

# BCI Competition IV 2a
BCI4_CONFIG = {
    'fs': 250,
    'ch_names': ['FC1', 'FCz', 'FC2', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP1', 'CP2'],
    'n_channels': 12,
    'selected_ch_0based': [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16],
    'c3_idx': 4,   # index of C3 in ch_names
    'c4_idx': 8,   # index of C4 in ch_names
    'filter_type': 'elliptic',
    'filter_low': 8,
    'filter_high': 35,
    'epoch_start': 125,   # sample index for epoch start
    'epoch_end': 625,     # sample index for epoch end
    # Class labels: 1 = Left Hand, 2 = Right Hand
    'class1_label': 'Left Hand',
    'class2_label': 'Right Hand',
    'labels_swapped': False,
}

# Cho2017 (DATA1-DATA5) — BioSemi 64ch, 13 selected channels
# Pipeline: Butterworth 8-30Hz → trim 0.5s both ends → channel select → z-score/3 clip
# NOTE: Cho2017 class labels are swapped: 1 = Right Hand, 2 = Left Hand
CHO_CONFIG = {
    'fs': 512,
    'ch_names': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP6', 'CP4'],
    'n_channels': 13,
    'selected_ch_0based': [13, 12, 11, 47, 48, 49, 50, 16, 17, 18, 55, 53, 54],  # 0-based from 64ch
    'c3_idx': 1,   # index of C3 in ch_names
    'c4_idx': 5,   # index of C4 in ch_names
    'filter_type': 'butterworth',
    'filter_low': 8,
    'filter_high': 30,
    'epoch_start': 256,    # 0.5s trim at 512Hz
    'epoch_end': -256,     # trim 0.5s from end too (negative = from end)
    # Class labels: 1 = Right Hand, 2 = Left Hand (swapped vs BCI4)
    'class1_label': 'Right Hand',
    'class2_label': 'Left Hand',
    'labels_swapped': True,
}

# Dataset definitions
DATASET_CONFIGS = {
    'BCI4': {
        'results_dir': 'results_MI_BCI4',
        'data_mat': 'data1.mat',
        'mat_key': 'xsubi_all',
        'signal': BCI4_CONFIG,
    },
    'DATA1': {
        'results_dir': 'results_MI_DATA1',
        'data_mat': 'DATA1.mat',
        'mat_key': 'xsubi_all',
        'signal': CHO_CONFIG,
    },
    'DATA2': {
        'results_dir': 'results_MI_DATA2',
        'data_mat': 'DATA2.mat',
        'mat_key': 'xsubi_all',
        'signal': CHO_CONFIG,
    },
    'DATA3': {
        'results_dir': 'results_MI_DATA3',
        'data_mat': 'DATA3.mat',
        'mat_key': 'xsubi_all',
        'signal': CHO_CONFIG,
    },
    'DATA4': {
        'results_dir': 'results_MI_DATA4',
        'data_mat': 'DATA4.mat',
        'mat_key': 'xsubi_all',
        'signal': CHO_CONFIG,
    },
    'DATA5': {
        'results_dir': 'results_MI_DATA5',
        'data_mat': 'DATA5.mat',
        'mat_key': 'xsubi_all',
        'signal': CHO_CONFIG,
    },
}

ALL_DATA_KEYS = ['DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5']


# ── Data Processing Utilities ─────────────────────────────────────────────────

def _get_eeg_field(subject_struct_item, field_name):
    if not hasattr(subject_struct_item, 'dtype') or subject_struct_item.dtype.names is None: return None
    if field_name not in subject_struct_item.dtype.names: return None
    field_content = subject_struct_item[field_name]
    actual_data = field_content[0,0] if isinstance(field_content, np.ndarray) and field_content.shape == (1,1) else field_content
    if not isinstance(actual_data, np.ndarray): return None
    if field_name == 'y': return actual_data.flatten()
    if actual_data.ndim == 2:
        s, c = actual_data.shape
        actual_data = np.expand_dims(actual_data, axis=2) if s > c else np.expand_dims(actual_data.T, axis=2)
    if actual_data.ndim != 3 or 0 in actual_data.shape: return None
    return actual_data

def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    nyq = 0.5 * fs
    low_stop = max(0.1, lowcut - 1.0)
    high_stop = min(nyq - 0.1, highcut + 1.0)
    wp = np.clip([lowcut / nyq, highcut / nyq], 1e-6, 1.0 - 1e-6).tolist()
    ws = np.clip([low_stop / nyq, high_stop / nyq], 1e-6, 1.0 - 1e-6).tolist()
    n, wn = ellipord(wp, ws, rp, rs)
    b, a = ellip(n, rp, rs, wn, btype='band')
    return filtfilt(b, a, data, axis=0)

def butterworth_filter(data, lowcut=8, highcut=30, fs=512, order=4):
    from scipy.signal import butter
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_filter(data, sig_cfg):
    """Apply the correct filter type for the dataset."""
    ftype = sig_cfg.get('filter_type', 'elliptic')
    if ftype == 'butterworth':
        return butterworth_filter(data, lowcut=sig_cfg['filter_low'],
                                  highcut=sig_cfg['filter_high'], fs=sig_cfg['fs'])
    else:
        return elliptical_filter(data, lowcut=sig_cfg['filter_low'],
                                 highcut=sig_cfg['filter_high'], fs=sig_cfg['fs'])

def get_mne_info(sig_cfg):
    """Create MNE info from signal config."""
    ch_names = sig_cfg['ch_names']
    info = mne.create_info(ch_names=ch_names, sfreq=sig_cfg['fs'],
                           ch_types=['eeg'] * len(ch_names))
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    return info

def extract_band_power(data, fs=250, band=(8, 30)):
    freqs, psd = welch(data, fs=fs, axis=2, nperseg=min(128, data.shape[2]))
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(np.sum(psd[:, :, idx_band], axis=2), axis=0)


# ── Label Helper ──────────────────────────────────────────────────────────────

def make_label(data_dict, dataset_name=''):
    sids = sorted(data_dict.keys())
    prefix = f"{dataset_name} — " if dataset_name else ""
    n_subj = len(sids)
    if n_subj > 1 and sids == list(range(1, n_subj + 1)):
        return f"{prefix}All {n_subj} Subjects", f"{dataset_name}_All{n_subj}" if dataset_name else f"All{n_subj}"
    if n_subj == 1:
        return f"{prefix}Subject {sids[0]}", f"{dataset_name}_S{sids[0]}" if dataset_name else f"S{sids[0]}"
    tag_parts = "_".join(f"S{s}" for s in sids)
    label_parts = ", ".join(f"S{s}" for s in sids)
    tag = f"{dataset_name}_{tag_parts}" if dataset_name else tag_parts
    return f"{prefix}{label_parts}", tag


def _get_sig_cfg(data_dict):
    """Extract signal config from data_dict (stored per subject)."""
    first_sid = next(iter(data_dict))
    return data_dict[first_sid].get('sig_cfg', BCI4_CONFIG)


# ── Data Loading ──────────────────────────────────────────────────────────────

def _discover_subject_ids(results_dir):
    """Scan results_dir for Subject_N folders and return sorted list of IDs."""
    ids = []
    if not os.path.isdir(results_dir):
        return ids
    for entry in os.listdir(results_dir):
        if entry.startswith('Subject_') and os.path.isdir(os.path.join(results_dir, entry)):
            try:
                sid = int(entry.split('_')[1])
                ids.append(sid)
            except (ValueError, IndexError):
                pass
    return sorted(ids)


def load_data(subject_ids=None, results_dir='results_MI_BCI4', data_mat='data1.mat',
              mat_key='xsubi_all', sig_cfg=None):
    """Load real + synthetic data for one dataset."""
    if sig_cfg is None:
        sig_cfg = BCI4_CONFIG

    # Auto-detect available subjects from directory
    available_ids = _discover_subject_ids(results_dir)
    if not available_ids:
        print(f"  No Subject_* folders found in '{results_dir}'")
        return None

    if subject_ids is None:
        subject_ids = available_ids
    else:
        subject_ids = [s for s in subject_ids if s in available_ids]

    n_subjects_in_mat = None
    n_ch = sig_cfg['n_channels']
    fs = sig_cfg['fs']
    selected_ch = sig_cfg['selected_ch_0based']
    ep_start = sig_cfg['epoch_start']
    ep_end = sig_cfg['epoch_end']
    f_low = sig_cfg['filter_low']
    f_high = sig_cfg['filter_high']

    print(f"Loading subjects: {subject_ids} from '{results_dir}' "
          f"(found {len(available_ids)} Subject_* folders, {n_ch}ch @ {fs}Hz)...")
    try:
        d1 = scipy.io.loadmat(data_mat)
        xsubi_all = d1[mat_key]
        n_subjects_in_mat = xsubi_all.shape[1]
    except Exception as e:
        print(f"  Could not load {data_mat}: {e}")
        return None

    data_dict = {}
    for sid in subject_ids:
        if sid < 1 or (n_subjects_in_mat is not None and sid > n_subjects_in_mat):
            print(f"  Skipping S{sid}: index out of range for {data_mat} "
                  f"(has {n_subjects_in_mat} subjects)")
            continue
        mat_path = os.path.join(results_dir, f'Subject_{sid}', f'synthetic_S{sid}.mat')
        if not os.path.exists(mat_path):
            print(f"  Skipping S{sid}: {mat_path} not found")
            continue

        synth_data = scipy.io.loadmat(mat_path)
        # Support both old (synthetic_x/y) and new (synthetic_filtered_x/y) field names
        if 'synthetic_filtered_x' in synth_data:
            x_synth = synth_data['synthetic_filtered_x'].copy()
            y_synth = synth_data['synthetic_filtered_y'].flatten()
        elif 'synthetic_x' in synth_data:
            x_synth = synth_data['synthetic_x'].copy()
            y_synth = synth_data['synthetic_y'].flatten()
        else:
            print(f"  Skipping S{sid}: no synthetic_x or synthetic_filtered_x in {mat_path}")
            continue
        ch_mean = synth_data['norm_ch_mean']
        ch_std = synth_data['norm_ch_std']

        raw_struct = xsubi_all[0, sid - 1]
        x_real_raw = _get_eeg_field(raw_struct, 'x')
        y_real_raw = _get_eeg_field(raw_struct, 'y')
        if x_real_raw is None or y_real_raw is None:
            print(f"  Skipping S{sid}: could not extract fields")
            continue

        c1_idx, c2_idx = np.where(y_real_raw == 1)[0], np.where(y_real_raw == 2)[0]
        n_c1 = min(TRIALS_PER_CLASS, len(c1_idx))
        n_c2 = min(TRIALS_PER_CLASS, len(c2_idx))
        x_real_selected = np.concatenate((x_real_raw[:, :, c1_idx[:n_c1]],
                                          x_real_raw[:, :, c2_idx[:n_c2]]), axis=2)
        y_real = np.concatenate((np.ones(n_c1), np.ones(n_c2) + 1))

        filtered_real = apply_filter(x_real_selected, sig_cfg)
        # Handle epoch slicing (epoch_end can be negative for trim-from-end)
        if ep_end < 0:
            filtered_real = filtered_real[ep_start:ep_end, :, :]
        else:
            filtered_real = filtered_real[ep_start:ep_end, :, :]
        x_real = filtered_real[:, selected_ch, :]
        x_real = np.transpose(x_real, (2, 1, 0))

        ch_mean_b = np.reshape(ch_mean, (1, n_ch, 1))
        ch_std_b = np.reshape(ch_std, (1, n_ch, 1))
        z = (x_real - ch_mean_b) / ch_std_b
        x_real_norm = np.clip(z / 3.0, -1., 1.).astype(np.float32)

        # Gain correction before balancing (uses full synthetic for stable stats)
        for ch in range(n_ch):
            std_real_ch = np.std(x_real_norm[:, ch, :])
            std_synth_ch = np.std(x_synth[:, ch, :])
            if std_synth_ch > 1e-8:
                x_synth[:, ch, :] *= (std_real_ch / std_synth_ch)

        # Balance synthetic to same trial count per class as real
        rng = np.random.RandomState(sid)
        synth_balanced_idx = []
        for cls, n_real_cls in [(1, n_c1), (2, n_c2)]:
            cls_idx = np.where(y_synth == cls)[0]
            if len(cls_idx) >= n_real_cls:
                chosen = rng.choice(cls_idx, size=n_real_cls, replace=False)
            else:
                chosen = cls_idx
            synth_balanced_idx.append(chosen)
        synth_balanced_idx = np.concatenate(synth_balanced_idx)
        x_synth_bal = x_synth[synth_balanced_idx]
        y_synth_bal = y_synth[synth_balanced_idx]

        # Remap labels if dataset has swapped class assignments
        # so Class 1 = Left Hand, Class 2 = Right Hand universally
        if sig_cfg.get('labels_swapped', False):
            y_real = 3.0 - y_real          # 1→2, 2→1
            y_synth_bal = 3.0 - y_synth_bal

        data_dict[sid] = {
            'real_x': x_real_norm.astype(np.float32),
            'real_y': y_real,
            'synth_x': x_synth_bal.astype(np.float32),
            'synth_y': y_synth_bal,
            'sig_cfg': sig_cfg,
        }
        print(f"  S{sid}: real={x_real_norm.shape}, synth={x_synth_bal.shape} "
              f"(balanced {n_c1}+{n_c2} per class)")
    return data_dict


def load_all_datasets(dataset_names, subject_ids=None):
    """Load data for multiple dataset names. Returns dict of {ds_name: data_dict}."""
    all_data = {}
    for ds_name in dataset_names:
        if ds_name not in DATASET_CONFIGS:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping.")
            continue
        cfg = DATASET_CONFIGS[ds_name]
        data = load_data(
            subject_ids=subject_ids,
            results_dir=cfg['results_dir'],
            data_mat=cfg['data_mat'],
            mat_key=cfg['mat_key'],
            sig_cfg=cfg['signal'],
        )
        if data and len(data) > 0:
            all_data[ds_name] = data
        else:
            print(f"  No valid data for {ds_name}.")
    return all_data


def combine_data_dicts(list_of_data_dicts):
    """Combine multiple data_dicts into one by re-keying subjects sequentially.

    Each data_dict maps subject_id -> {real_x, real_y, synth_x, synth_y}.
    We re-key them as 1, 2, 3, ... across all dicts so the plotting
    functions treat them as one pool of subjects.
    """
    combined = {}
    counter = 1
    for dd in list_of_data_dicts:
        for sid in sorted(dd.keys()):
            combined[counter] = dd[sid]
            counter += 1
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOPOPLOTS — LI + Z-scored
# ══════════════════════════════════════════════════════════════════════════════

def plot_topoplots(data_dict, output_dir, dataset_name=''):
    label, tag = make_label(data_dict, dataset_name)
    print(f"Generating Topoplots ({label})...")
    sig_cfg = _get_sig_cfg(data_dict)
    info = get_mne_info(sig_cfg)
    fs = sig_cfg['fs']

    all_real_bp1, all_real_bp2 = [], []
    all_synth_bp1, all_synth_bp2 = [], []

    for sid, d in data_dict.items():
        rx, ry = d['real_x'], d['real_y']
        sx, sy = d['synth_x'], d['synth_y']
        all_real_bp1.append(extract_band_power(rx[ry == 1], fs=fs))
        all_real_bp2.append(extract_band_power(rx[ry == 2], fs=fs))
        all_synth_bp1.append(extract_band_power(sx[sy == 1], fs=fs))
        all_synth_bp2.append(extract_band_power(sx[sy == 2], fs=fs))

    bp_r1 = np.mean(all_real_bp1, axis=0)
    bp_r2 = np.mean(all_real_bp2, axis=0)
    bp_s1 = np.mean(all_synth_bp1, axis=0)
    bp_s2 = np.mean(all_synth_bp2, axis=0)

    # ─── Laterality Index ────────────────────────────────────────────────────
    li_real = (bp_r1 - bp_r2) / (bp_r1 + bp_r2 + 1e-12)
    li_synth = (bp_s1 - bp_s2) / (bp_s1 + bp_s2 + 1e-12)
    vmax_li = max(np.max(np.abs(li_real)), np.max(np.abs(li_synth)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    kwargs_li = dict(show=False, cmap='RdBu_r', vlim=(-vmax_li, vmax_li), extrapolate='local')
    im1, _ = mne.viz.plot_topomap(li_real, info, axes=axes[0], **kwargs_li)
    im2, _ = mne.viz.plot_topomap(li_synth, info, axes=axes[1], **kwargs_li)
    axes[0].set_title("Real")
    axes[1].set_title("Synthetic")
    fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04,
                 label="LI: (Left Hand − Right Hand) / (Left Hand + Right Hand)")
    fig.suptitle(f"Laterality Index — mu/beta Band Power (8-30 Hz)\n{label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topoplot_LI_{tag}.png'), dpi=150)
    plt.close(fig)
    print(f"  LI Real: [{li_real.min():.4f}, {li_real.max():.4f}]  Synth: [{li_synth.min():.4f}, {li_synth.max():.4f}]")

    # ─── Z-scored band power (2x2) ──────────────────────────────────────────
    def zscore_channels(bp):
        return (bp - np.mean(bp)) / (np.std(bp) + 1e-8)

    z_r1, z_r2 = zscore_channels(bp_r1), zscore_channels(bp_r2)
    z_s1, z_s2 = zscore_channels(bp_s1), zscore_channels(bp_s2)
    vmax_z = max(np.max(np.abs(z_r1)), np.max(np.abs(z_r2)),
                 np.max(np.abs(z_s1)), np.max(np.abs(z_s2)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    kwargs_z = dict(show=False, cmap='RdBu_r', vlim=(-vmax_z, vmax_z), extrapolate='local')
    mne.viz.plot_topomap(z_r1, info, axes=axes[0, 0], **kwargs_z)
    im_r2, _ = mne.viz.plot_topomap(z_r2, info, axes=axes[0, 1], **kwargs_z)
    mne.viz.plot_topomap(z_s1, info, axes=axes[1, 0], **kwargs_z)
    im_s2, _ = mne.viz.plot_topomap(z_s2, info, axes=axes[1, 1], **kwargs_z)
    axes[0, 0].set_title("Real — Left Hand (Class 1)")
    axes[0, 1].set_title("Real — Right Hand (Class 2)")
    axes[1, 0].set_title("Synth — Left Hand (Class 1)")
    axes[1, 1].set_title("Synth — Right Hand (Class 2)")
    fig.colorbar(im_r2, ax=axes[0, :].ravel().tolist(), fraction=0.03, pad=0.04, label="Z-scored Power")
    fig.colorbar(im_s2, ax=axes[1, :].ravel().tolist(), fraction=0.03, pad=0.04, label="Z-scored Power")
    fig.suptitle(f"Z-scored mu/beta Band Power (8-30 Hz) — {label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topoplot_Zscore_{tag}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAND AVERAGE — C3 + C4 dual channel
# ══════════════════════════════════════════════════════════════════════════════

def plot_grand_averages(data_dict, output_dir, dataset_name=''):
    label, tag = make_label(data_dict, dataset_name)
    print(f"Generating Grand Average (C3 + C4, {label})...")
    sig_cfg = _get_sig_cfg(data_dict)
    C3_IDX = sig_cfg['c3_idx']
    C4_IDX = sig_cfg['c4_idx']
    n_samples = next(iter(data_dict.values()))['real_x'].shape[2]
    t = np.linspace(0, n_samples / sig_cfg['fs'], n_samples)

    ga = {key: [] for key in ['r1_c3', 'r2_c3', 's1_c3', 's2_c3',
                               'r1_c4', 'r2_c4', 's1_c4', 's2_c4']}
    for sid, d in data_dict.items():
        rx, ry = d['real_x'], d['real_y']
        sx, sy = d['synth_x'], d['synth_y']
        ga['r1_c3'].append(np.mean(rx[ry == 1, C3_IDX, :], axis=0))
        ga['r2_c3'].append(np.mean(rx[ry == 2, C3_IDX, :], axis=0))
        ga['s1_c3'].append(np.mean(sx[sy == 1, C3_IDX, :], axis=0))
        ga['s2_c3'].append(np.mean(sx[sy == 2, C3_IDX, :], axis=0))
        ga['r1_c4'].append(np.mean(rx[ry == 1, C4_IDX, :], axis=0))
        ga['r2_c4'].append(np.mean(rx[ry == 2, C4_IDX, :], axis=0))
        ga['s1_c4'].append(np.mean(sx[sy == 1, C4_IDX, :], axis=0))
        ga['s2_c4'].append(np.mean(sx[sy == 2, C4_IDX, :], axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)

    # Top row: C3 — contralateral to Right Hand
    # (0,0) Left Hand at C3: Real vs Synth
    axes[0, 0].plot(t, np.mean(ga['r1_c3'], axis=0), label='Real', color='#1F77B4', lw=1.8)
    axes[0, 0].plot(t, np.mean(ga['s1_c3'], axis=0), label='Synthetic', color='#D62728', lw=1.8, ls='--')
    axes[0, 0].set_title("C3 — Left Hand (ipsilateral)")
    axes[0, 0].legend(fontsize=9)

    # (0,1) Right Hand at C3: Real vs Synth
    axes[0, 1].plot(t, np.mean(ga['r2_c3'], axis=0), label='Real', color='#1F77B4', lw=1.8)
    axes[0, 1].plot(t, np.mean(ga['s2_c3'], axis=0), label='Synthetic', color='#D62728', lw=1.8, ls='--')
    axes[0, 1].set_title("C3 — Right Hand (contralateral)")
    axes[0, 1].legend(fontsize=9)

    # Bottom row: C4 — contralateral to Left Hand
    # (1,0) Left Hand at C4: Real vs Synth
    axes[1, 0].plot(t, np.mean(ga['r1_c4'], axis=0), label='Real', color='#1F77B4', lw=1.8)
    axes[1, 0].plot(t, np.mean(ga['s1_c4'], axis=0), label='Synthetic', color='#D62728', lw=1.8, ls='--')
    axes[1, 0].set_title("C4 — Left Hand (contralateral)")
    axes[1, 0].legend(fontsize=9)

    # (1,1) Right Hand at C4: Real vs Synth
    axes[1, 1].plot(t, np.mean(ga['r2_c4'], axis=0), label='Real', color='#1F77B4', lw=1.8)
    axes[1, 1].plot(t, np.mean(ga['s2_c4'], axis=0), label='Synthetic', color='#D62728', lw=1.8, ls='--')
    axes[1, 1].set_title("C4 — Right Hand (ipsilateral)")
    axes[1, 1].legend(fontsize=9)

    for ax in axes.ravel():
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Grand Average — Real vs Synthetic — C3 & C4 — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grandaverage_{tag}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PSD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_psd(data_dict, output_dir, dataset_name=''):
    label, tag = make_label(data_dict, dataset_name)
    print(f"Generating PSD ({label})...")
    sig_cfg = _get_sig_cfg(data_dict)
    FS = sig_cfg['fs']

    all_psd_real, all_psd_synth = [], []
    freqs_out = None
    for sid, d in data_dict.items():
        f_r, psd_r = welch(d['real_x'], fs=FS, axis=2, nperseg=min(128, d['real_x'].shape[2]))
        f_s, psd_s = welch(d['synth_x'], fs=FS, axis=2, nperseg=min(128, d['synth_x'].shape[2]))
        all_psd_real.append(np.mean(np.mean(psd_r, axis=0), axis=0))
        all_psd_synth.append(np.mean(np.mean(psd_s, axis=0), axis=0))
        freqs_out = f_r

    mean_r, mean_s = np.mean(all_psd_real, axis=0), np.mean(all_psd_synth, axis=0)
    std_r, std_s = np.std(all_psd_real, axis=0), np.std(all_psd_synth, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs_out, mean_r, 'b-', lw=2, label='Real')
    ax.fill_between(freqs_out, mean_r - std_r, mean_r + std_r, color='blue', alpha=0.15)
    ax.semilogy(freqs_out, mean_s, 'r-', lw=2, label='Synthetic')
    ax.fill_between(freqs_out, mean_s - std_s, mean_s + std_s, color='red', alpha=0.15)
    ax.set_xlim([2, 60])
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density (log)", fontsize=12)
    ax.set_title(f"PSD Comparison — {label} (Mean ± Std across subjects)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    mask = (freqs_out >= 8) & (freqs_out <= 30)
    r, p = pearsonr(mean_r[mask], mean_s[mask])
    ax.text(0.98, 0.95, f'r = {r:.4f} (8-30 Hz)', transform=ax.transAxes,
            ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'psd_{tag}.png'), dpi=150)
    plt.close(fig)
    print(f"  PSD correlation (8-30 Hz): r={r:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME-FREQUENCY — Log-ratio between classes, C3 + C4
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_frequency(data_dict, output_dir, dataset_name=''):
    label, tag = make_label(data_dict, dataset_name)
    print(f"Generating TFR (C3 + C4, log-ratio, {label})...")
    sig_cfg = _get_sig_cfg(data_dict)
    info = get_mne_info(sig_cfg)
    C3_IDX = sig_cfg['c3_idx']
    C4_IDX = sig_cfg['c4_idx']
    freqs = np.arange(8, 31, 1)
    n_cycles = freqs / 2.0

    def compute_tfr(x):
        epochs = mne.EpochsArray(x, info, verbose=False)
        tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                            return_itc=False, average=True, verbose=False)
        return tfr.data

    tfr_keys = ['r1_c3', 'r2_c3', 's1_c3', 's2_c3',
                'r1_c4', 'r2_c4', 's1_c4', 's2_c4']
    tfr_all = {k: [] for k in tfr_keys}

    for sid, d in data_dict.items():
        rx, ry = d['real_x'], d['real_y']
        sx, sy = d['synth_x'], d['synth_y']

        tfr_r1 = compute_tfr(rx[ry == 1])
        tfr_r2 = compute_tfr(rx[ry == 2])
        tfr_s1 = compute_tfr(sx[sy == 1])
        tfr_s2 = compute_tfr(sx[sy == 2])

        tfr_all['r1_c3'].append(tfr_r1[C3_IDX])
        tfr_all['r2_c3'].append(tfr_r2[C3_IDX])
        tfr_all['s1_c3'].append(tfr_s1[C3_IDX])
        tfr_all['s2_c3'].append(tfr_s2[C3_IDX])
        tfr_all['r1_c4'].append(tfr_r1[C4_IDX])
        tfr_all['r2_c4'].append(tfr_r2[C4_IDX])
        tfr_all['s1_c4'].append(tfr_s1[C4_IDX])
        tfr_all['s2_c4'].append(tfr_s2[C4_IDX])

    def log_ratio(num, den):
        return np.log10((np.mean(num, axis=0) + 1e-12) / (np.mean(den, axis=0) + 1e-12))

    lr_real_c3 = log_ratio(tfr_all['r2_c3'], tfr_all['r1_c3'])
    lr_synth_c3 = log_ratio(tfr_all['s2_c3'], tfr_all['s1_c3'])
    lr_real_c4 = log_ratio(tfr_all['r1_c4'], tfr_all['r2_c4'])
    lr_synth_c4 = log_ratio(tfr_all['s1_c4'], tfr_all['s2_c4'])

    vmax = max(np.max(np.abs(lr_real_c3)), np.max(np.abs(lr_synth_c3)),
               np.max(np.abs(lr_real_c4)), np.max(np.abs(lr_synth_c4)))
    extent = [0, 2, freqs[0], freqs[-1]]
    kwargs_tfr = dict(aspect='auto', origin='lower', cmap='RdBu_r',
                      extent=extent, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    im = axes[0, 0].imshow(lr_real_c3, **kwargs_tfr)
    axes[0, 0].set_title("Real — C3: log₁₀(Right / Left)")
    axes[0, 1].imshow(lr_synth_c3, **kwargs_tfr)
    axes[0, 1].set_title("Synth — C3: log₁₀(Right / Left)")

    axes[1, 0].imshow(lr_real_c4, **kwargs_tfr)
    axes[1, 0].set_title("Real — C4: log₁₀(Left / Right)")
    axes[1, 1].imshow(lr_synth_c4, **kwargs_tfr)
    axes[1, 1].set_title("Synth — C4: log₁₀(Left / Right)")

    for ax in axes.ravel():
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04,
                 label="log₁₀ power ratio (blue = contralateral ERD)")
    fig.suptitle(f"Time-Frequency — Inter-Class Log-Ratio — {label}\n"
                 f"(Blue = contralateral ERD, no baseline correction needed)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tfr_{tag}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA DISTRIBUTION — Balanced (same N for real & synthetic)
# ══════════════════════════════════════════════════════════════════════════════

def plot_data_distribution(data_dict, output_dir, dataset_name=''):
    """Plot amplitude and PSD distributions.

    Data is already balanced at load time (TRIALS_PER_CLASS per class for
    both real and synthetic), so no additional subsampling is needed here.
    """
    label, tag = make_label(data_dict, dataset_name)
    print(f"Generating Data Distribution ({label})...")
    sig_cfg = _get_sig_cfg(data_dict)
    FS = sig_cfg['fs']
    CH_NAMES = sig_cfg['ch_names']
    n_ch = sig_cfg['n_channels']

    all_real_vals = []
    all_synth_vals = []
    all_psd_real = []
    all_psd_synth = []
    freqs_out = None

    for sid, d in data_dict.items():
        rx, ry = d['real_x'], d['real_y']
        sx, sy = d['synth_x'], d['synth_y']

        # Amplitude distributions (flatten across channels & time)
        all_real_vals.append(rx.flatten())
        all_synth_vals.append(sx.flatten())

        # PSD
        f_r, psd_r = welch(rx, fs=FS, axis=2, nperseg=min(128, rx.shape[2]))
        f_s, psd_s = welch(sx, fs=FS, axis=2, nperseg=min(128, sx.shape[2]))
        all_psd_real.append(np.mean(np.mean(psd_r, axis=0), axis=0))
        all_psd_synth.append(np.mean(np.mean(psd_s, axis=0), axis=0))
        freqs_out = f_r

    real_vals = np.concatenate(all_real_vals)
    synth_vals = np.concatenate(all_synth_vals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Amplitude histogram
    bins = np.linspace(-1, 1, 100)
    axes[0].hist(real_vals, bins=bins, alpha=0.6, density=True, label='Real', color='#1F77B4')
    axes[0].hist(synth_vals, bins=bins, alpha=0.6, density=True, label='Synthetic', color='#D62728')
    axes[0].set_xlabel("Normalized Amplitude", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Amplitude Distribution", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    n_real_total = sum(d['real_x'].shape[0] for d in data_dict.values())
    n_synth_total = sum(d['synth_x'].shape[0] for d in data_dict.values())
    axes[0].text(0.02, 0.95,
                 f'Real: {n_real_total} trials\nSynth: {n_synth_total} trials\n'
                 f'({TRIALS_PER_CLASS}/class balanced)',
                 transform=axes[0].transAxes, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # (b) Per-channel std comparison
    ch_std_real, ch_std_synth = [], []
    for sid, d in data_dict.items():
        rx = d['real_x']
        sx = d['synth_x']
        ch_std_real.append([np.std(rx[:, ch, :]) for ch in range(n_ch)])
        ch_std_synth.append([np.std(sx[:, ch, :]) for ch in range(n_ch)])
    mean_std_r = np.mean(ch_std_real, axis=0)
    mean_std_s = np.mean(ch_std_synth, axis=0)
    x_pos = np.arange(n_ch)
    w = 0.35
    axes[1].bar(x_pos - w/2, mean_std_r, w, label='Real', color='#1F77B4', alpha=0.8)
    axes[1].bar(x_pos + w/2, mean_std_s, w, label='Synthetic', color='#D62728', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(CH_NAMES, rotation=45, ha='right', fontsize=9)
    axes[1].set_xlabel("Channel", fontsize=11)
    axes[1].set_ylabel("Std Dev", fontsize=11)
    axes[1].set_title("Per-Channel Std Dev", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')

    # (c) PSD overlay
    mean_psd_r = np.mean(all_psd_real, axis=0)
    mean_psd_s = np.mean(all_psd_synth, axis=0)
    std_psd_r = np.std(all_psd_real, axis=0)
    std_psd_s = np.std(all_psd_synth, axis=0)
    axes[2].semilogy(freqs_out, mean_psd_r, 'b-', lw=2, label='Real')
    axes[2].fill_between(freqs_out, mean_psd_r - std_psd_r, mean_psd_r + std_psd_r, color='blue', alpha=0.15)
    axes[2].semilogy(freqs_out, mean_psd_s, 'r-', lw=2, label='Synthetic')
    axes[2].fill_between(freqs_out, mean_psd_s - std_psd_s, mean_psd_s + std_psd_s, color='red', alpha=0.15)
    axes[2].set_xlim([2, 60])
    axes[2].set_xlabel("Frequency (Hz)", fontsize=11)
    axes[2].set_ylabel("PSD (log)", fontsize=11)
    axes[2].set_title("PSD Comparison", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    mask = (freqs_out >= 8) & (freqs_out <= 30)
    r, p = pearsonr(mean_psd_r[mask], mean_psd_s[mask])
    axes[2].text(0.98, 0.95, f'r = {r:.4f} (8-30 Hz)', transform=axes[2].transAxes,
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.suptitle(f"Data Distribution — Real vs Synthetic ({TRIALS_PER_CLASS}/class) — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_{tag}.png'), dpi=150)
    plt.close(fig)
    print(f"  Distribution saved ({n_real_total} real, {n_synth_total} synth — balanced at {TRIALS_PER_CLASS}/class)")


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_accuracies(results_dir):
    """Load per-subject CSP accuracies from a results directory.

    Returns list of dicts: [{sid, baseline, synth_only, best_aug, strategy, ratio}, ...]
    """
    available_ids = _discover_subject_ids(results_dir)
    records = []
    for sid in available_ids:
        mat_path = os.path.join(results_dir, f'Subject_{sid}', f'synthetic_S{sid}.mat')
        if not os.path.exists(mat_path):
            continue
        try:
            m = scipy.io.loadmat(mat_path)
        except Exception:
            continue

        def _scalar(key):
            v = m.get(key, None)
            if v is None:
                return np.nan
            v = np.asarray(v).flatten()
            return float(v[0]) if len(v) > 0 else np.nan

        def _string(key):
            v = m.get(key, '')
            if isinstance(v, np.ndarray):
                v = v.flatten()
                v = str(v[0]) if len(v) > 0 else ''
            return str(v)

        records.append({
            'sid': sid,
            'baseline': _scalar('csp_baseline_acc') * 100,   # → %
            'synth_only': _scalar('csp_synth_only_acc') * 100,
            'best_aug': _scalar('csp_best_aug_acc') * 100,
            'strategy': _string('csp_best_strategy'),
            'ratio': _scalar('csp_best_ratio'),
        })
    return records


def load_all_accuracies(dataset_names):
    """Load accuracies for multiple datasets.
    Returns dict: {ds_name: [records]}
    """
    all_acc = {}
    for ds_name in dataset_names:
        if ds_name not in DATASET_CONFIGS:
            continue
        cfg = DATASET_CONFIGS[ds_name]
        recs = load_accuracies(cfg['results_dir'])
        if recs:
            all_acc[ds_name] = recs
            print(f"  Loaded {len(recs)} accuracy records from {ds_name}")
        else:
            print(f"  No accuracy records found for {ds_name}")
    return all_acc


# ══════════════════════════════════════════════════════════════════════════════
# 6. ACCURACY SCATTER PLOTS — per dataset
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_scatter(records, output_dir, dataset_name):
    """Scatter: (a) Baseline vs Synth-Only, (b) Baseline vs Best-Augmented."""
    print(f"Generating Accuracy Scatter ({dataset_name})...")
    os.makedirs(output_dir, exist_ok=True)

    bl = np.array([r['baseline'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['best_aug'] for r in records])
    sids = [r['sid'] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── (a) Baseline vs Synth-Only ──
    ax = axes[0]
    lo = min(bl.min(), so.min()) - 3
    hi = max(bl.max(), so.max()) + 3
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=1, label='y = x')
    ax.scatter(bl, so, s=70, c='#2CA02C', edgecolors='k', linewidths=0.5, zorder=5)
    for i, sid in enumerate(sids):
        ax.annotate(f'S{sid}', (bl[i], so[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Baseline Accuracy (%)", fontsize=11)
    ax.set_ylabel("Synthetic-Only Accuracy (%)", fontsize=11)
    ax.set_title("Baseline vs Synthetic-Only", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ── (b) Baseline vs Best-Augmented ──
    ax = axes[1]
    lo = min(bl.min(), ba.min()) - 3
    hi = max(bl.max(), ba.max()) + 3
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=1, label='y = x')
    ax.scatter(bl, ba, s=70, c='#D62728', edgecolors='k', linewidths=0.5, zorder=5)
    for i, sid in enumerate(sids):
        ax.annotate(f'S{sid}', (bl[i], ba[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Baseline Accuracy (%)", fontsize=11)
    ax.set_ylabel("Best Augmented Accuracy (%)", fontsize=11)
    ax.set_title("Baseline vs Best Augmented", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    n = len(records)
    above_bl = np.sum(ba > bl)
    fig.suptitle(f"Accuracy Scatter — {dataset_name} ({n} subjects, "
                 f"{above_bl}/{n} improved with augmentation)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_scatter_{dataset_name}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ACCURACY BOX PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _boxplot_three(ax, bl, so, ba, title):
    """Helper: Baseline vs Synth-Only vs Best-Aug box plot on one axis."""
    data = [bl, so, ba]
    labels = ['Baseline', 'Synth-Only', 'Best Aug']
    colors = ['#1F77B4', '#2CA02C', '#D62728']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, d in enumerate(data):
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(d))
        ax.scatter(np.full_like(d, i + 1) + jitter, d, s=25, c='k', alpha=0.5, zorder=5)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    # Annotate means
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(i + 1, ax.get_ylim()[1] * 0.98, f'μ={mean_val:.1f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def plot_accuracy_boxplots(records, output_dir, dataset_name):
    """Per-dataset box plot: Baseline vs Synth-Only vs Best-Aug."""
    print(f"Generating Accuracy Box Plot ({dataset_name})...")
    os.makedirs(output_dir, exist_ok=True)

    bl = np.array([r['baseline'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['best_aug'] for r in records])

    fig, ax = plt.subplots(figsize=(7, 6))
    _boxplot_three(ax, bl, so, ba, f"{dataset_name} — All Subjects (n={len(records)})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_{dataset_name}.png'), dpi=150)
    plt.close(fig)


def plot_accuracy_boxplots_combined(all_acc, output_dir):
    """Combined box plots across BCI4 + DATA with high/low performer split.

    High performers: baseline >= 60%, Low performers: baseline < 60%.
    """
    print("Generating Combined Accuracy Box Plots (High/Low performers)...")
    os.makedirs(output_dir, exist_ok=True)

    # Flatten all records and tag by dataset group
    bci4_recs = all_acc.get('BCI4', [])
    data_recs = []
    for k in ALL_DATA_KEYS:
        data_recs.extend(all_acc.get(k, []))

    all_recs = bci4_recs + data_recs

    if not all_recs:
        print("  No accuracy records to combine.")
        return

    bl_all = np.array([r['baseline'] for r in all_recs])
    so_all = np.array([r['synth_only'] for r in all_recs])
    ba_all = np.array([r['best_aug'] for r in all_recs])

    high_mask = bl_all >= 60
    low_mask = bl_all < 60
    n_high, n_low = int(np.sum(high_mask)), int(np.sum(low_mask))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) All subjects combined
    _boxplot_three(axes[0], bl_all, so_all, ba_all,
                   f"All Subjects Combined (n={len(all_recs)})")

    # (b) High performers
    if n_high >= 2:
        _boxplot_three(axes[1], bl_all[high_mask], so_all[high_mask], ba_all[high_mask],
                       f"High Performers (baseline ≥ 60%, n={n_high})")
    else:
        axes[1].text(0.5, 0.5, f'Not enough high performers\n(n={n_high})',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("High Performers (baseline ≥ 60%)")

    # (c) Low performers
    if n_low >= 2:
        _boxplot_three(axes[2], bl_all[low_mask], so_all[low_mask], ba_all[low_mask],
                       f"Low Performers (baseline < 60%, n={n_low})")
    else:
        axes[2].text(0.5, 0.5, f'Not enough low performers\n(n={n_low})',
                     ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title("Low Performers (baseline < 60%)")

    ds_str = "BCI4" if bci4_recs else ""
    if data_recs:
        ds_str += (" + " if ds_str else "") + "DATA1-5"
    fig.suptitle(f"Accuracy Comparison — {ds_str} Combined", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_combined_highlow.png'), dpi=150)
    plt.close(fig)
    print(f"  Combined box plots: {len(all_recs)} total, {n_high} high, {n_low} low")


# ══════════════════════════════════════════════════════════════════════════════
# 8. STATISTICAL ANALYSIS — per dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(records, output_dir, dataset_name):
    """Paired statistical tests for one dataset.

    Tests:
      - Paired t-test + Wilcoxon signed-rank: Baseline vs Synth-Only
      - Paired t-test + Wilcoxon signed-rank: Baseline vs Best-Aug
      - Paired t-test + Wilcoxon signed-rank: Synth-Only vs Best-Aug
    Saves text report + summary bar chart.
    """
    print(f"Running Statistical Analysis ({dataset_name})...")
    os.makedirs(output_dir, exist_ok=True)

    bl = np.array([r['baseline'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['best_aug'] for r in records])
    n = len(records)

    lines = []
    lines.append("=" * 70)
    lines.append(f"Statistical Analysis — {dataset_name} (n={n})")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Descriptive Statistics (%):")
    lines.append(f"  Baseline:      mean={np.mean(bl):.2f}  std={np.std(bl):.2f}  "
                 f"median={np.median(bl):.2f}  min={np.min(bl):.2f}  max={np.max(bl):.2f}")
    lines.append(f"  Synth-Only:    mean={np.mean(so):.2f}  std={np.std(so):.2f}  "
                 f"median={np.median(so):.2f}  min={np.min(so):.2f}  max={np.max(so):.2f}")
    lines.append(f"  Best-Aug:      mean={np.mean(ba):.2f}  std={np.std(ba):.2f}  "
                 f"median={np.median(ba):.2f}  min={np.min(ba):.2f}  max={np.max(ba):.2f}")
    lines.append("")

    # Per-subject table
    lines.append("Per-Subject Accuracies (%):")
    lines.append(f"  {'Subject':<10} {'Baseline':>10} {'Synth-Only':>12} {'Best-Aug':>10} "
                 f"{'Δ(Aug-BL)':>10} {'Strategy':<15} {'Ratio':<8}")
    lines.append("  " + "-" * 75)
    for r in records:
        delta = r['best_aug'] - r['baseline']
        lines.append(f"  S{r['sid']:<9} {r['baseline']:>10.2f} {r['synth_only']:>12.2f} "
                     f"{r['best_aug']:>10.2f} {delta:>+10.2f} {r['strategy']:<15} "
                     f"{r['ratio']:<8.2f}")
    lines.append("")

    # Pairwise tests
    comparisons = [
        ("Baseline vs Synth-Only", bl, so),
        ("Baseline vs Best-Aug", bl, ba),
        ("Synth-Only vs Best-Aug", so, ba),
    ]

    test_results = []
    lines.append("Paired Statistical Tests:")
    lines.append("-" * 70)
    for name, a, b in comparisons:
        diff = b - a
        mean_diff = np.mean(diff)

        # Paired t-test
        if n >= 2 and np.std(diff) > 1e-10:
            t_stat, t_p = ttest_rel(a, b)
        else:
            t_stat, t_p = np.nan, np.nan

        # Wilcoxon signed-rank (needs n >= 6 and non-zero differences)
        nonzero_diff = diff[np.abs(diff) > 1e-10]
        if len(nonzero_diff) >= 6:
            try:
                w_stat, w_p = wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan

        # Effect size (Cohen's d for paired samples)
        if np.std(diff) > 1e-10:
            cohens_d = mean_diff / np.std(diff)
        else:
            cohens_d = np.nan

        sig_t = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "n.s."
        sig_w = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else "n.s."

        lines.append(f"\n  {name}:")
        lines.append(f"    Mean difference: {mean_diff:+.2f}%")
        lines.append(f"    Paired t-test:   t={t_stat:.4f}, p={t_p:.6f}  {sig_t}")
        lines.append(f"    Wilcoxon:        W={w_stat:.1f}, p={w_p:.6f}  {sig_w}" if not np.isnan(w_stat)
                     else f"    Wilcoxon:        n/a (insufficient non-zero differences, need ≥6)")
        lines.append(f"    Cohen's d:       {cohens_d:.4f}" if not np.isnan(cohens_d)
                     else f"    Cohen's d:       n/a")

        test_results.append({
            'name': name, 'mean_diff': mean_diff,
            't_stat': t_stat, 't_p': t_p, 'sig_t': sig_t,
            'w_stat': w_stat, 'w_p': w_p, 'sig_w': sig_w,
            'cohens_d': cohens_d,
        })

    lines.append("")
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant")
    lines.append("=" * 70)

    # Save text report
    report_path = os.path.join(output_dir, f'stats_report_{dataset_name}.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {report_path}")

    # Print to console
    for line in lines:
        print(f"  {line}")

    # ── Summary figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Mean accuracy bars with error bars
    means = [np.mean(bl), np.mean(so), np.mean(ba)]
    stds = [np.std(bl), np.std(so), np.std(ba)]
    colors = ['#1F77B4', '#2CA02C', '#D62728']
    labels_bar = ['Baseline', 'Synth-Only', 'Best-Aug']
    x_pos = np.arange(3)

    bars = axes[0].bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_bar, fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title(f"Mean Accuracy ± Std — {dataset_name} (n={n})", fontsize=12)
    axes[0].grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 1, f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (b) Paired difference plot (Baseline → Best-Aug)
    diff_ba = ba - bl
    sorted_idx = np.argsort(diff_ba)
    diff_sorted = diff_ba[sorted_idx]
    sids_sorted = [records[i]['sid'] for i in sorted_idx]
    colors_diff = ['#D62728' if d < 0 else '#2CA02C' for d in diff_sorted]
    y_pos = np.arange(len(diff_sorted))
    axes[1].barh(y_pos, diff_sorted, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f'S{s}' for s in sids_sorted], fontsize=9)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Δ Accuracy (Best-Aug − Baseline) (%)", fontsize=11)
    axes[1].set_title("Per-Subject Improvement", fontsize=12)
    axes[1].grid(alpha=0.3, axis='x')

    # Add significance annotation
    for tr in test_results:
        if tr['name'] == "Baseline vs Best-Aug":
            sig_label = f"p={tr['t_p']:.4f} ({tr['sig_t']})" if not np.isnan(tr['t_p']) else "n/a"
            axes[1].text(0.98, 0.02, f'Paired t: {sig_label}\nCohen\'s d: {tr["cohens_d"]:.3f}',
                         transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            break

    fig.suptitle(f"Statistical Summary — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stats_summary_{dataset_name}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER — runs all plots for a single data_dict
# ══════════════════════════════════════════════════════════════════════════════

def run_all_plots(data_dict, output_dir, dataset_name=''):
    os.makedirs(output_dir, exist_ok=True)
    plot_topoplots(data_dict, output_dir, dataset_name)
    plot_grand_averages(data_dict, output_dir, dataset_name)
    plot_psd(data_dict, output_dir, dataset_name)
    plot_time_frequency(data_dict, output_dir, dataset_name)
    plot_data_distribution(data_dict, output_dir, dataset_name)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='EEG cWGAN-GP Visualization v5 — Multi-Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                                    # All datasets, all subjects
  %(prog)s --datasets BCI4                    # Only BCI4
  %(prog)s --datasets DATA1 DATA3            # Only DATA1 + DATA3
  %(prog)s --datasets BCI4 -s 1 3 7          # BCI4, selected subjects
  %(prog)s -o my_output_dir                   # Custom output root""")

    parser.add_argument('-d', '--datasets', type=str, nargs='+', default=None, metavar='DS',
                        help='Dataset names: BCI4, DATA1, DATA2, DATA3, DATA4, DATA5. '
                             'Default: all available')
    parser.add_argument('-s', '--subjects', type=int, nargs='+', default=None, metavar='N',
                        help='Subject IDs. Default: all per dataset')
    parser.add_argument('-o', '--output-dir', type=str, default='visualization_results_v5',
                        help='Output root directory (default: visualization_results_v5)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine which datasets to run
    if args.datasets is None:
        # Default: all that exist
        available = []
        for ds_name, cfg in DATASET_CONFIGS.items():
            if os.path.isdir(cfg['results_dir']):
                available.append(ds_name)
        if not available:
            print("ERROR: No results directories found. Expected one or more of:")
            for ds_name, cfg in DATASET_CONFIGS.items():
                print(f"  {cfg['results_dir']}/")
            sys.exit(1)
        dataset_names = available
    else:
        dataset_names = [d.upper() for d in args.datasets]

    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    print("=" * 70)
    print(f"EEG cWGAN-GP Visualization v5 — Multi-Dataset")
    print(f"  Datasets:    {', '.join(dataset_names)}")
    print(f"  Subjects:    {args.subjects if args.subjects else 'All'}")
    print(f"  Output root: {output_root}")
    print("=" * 70)

    # Load all requested datasets
    all_ds_data = load_all_datasets(dataset_names, subject_ids=args.subjects)

    if not all_ds_data:
        print("ERROR: No valid data loaded from any dataset.")
        sys.exit(1)

    # ── 1. BCI4 plots (if loaded) ─────────────────────────────────────────────
    if 'BCI4' in all_ds_data:
        ds_out = os.path.join(output_root, 'BCI4')
        print(f"\n{'─' * 60}")
        print(f"Processing BCI4 ({len(all_ds_data['BCI4'])} subjects)")
        print(f"{'─' * 60}")
        run_all_plots(all_ds_data['BCI4'], ds_out, dataset_name='BCI4')

    # ── 2. Combined DATA1-DATA5 plots ──────────────────────────────────────
    loaded_data_splits = {k: v for k, v in all_ds_data.items() if k in ALL_DATA_KEYS}
    if loaded_data_splits:
        combined_name = "DATA_Combined"
        combined_dict = combine_data_dicts([loaded_data_splits[k] for k in sorted(loaded_data_splits.keys())])
        combined_out = os.path.join(output_root, combined_name)
        n_total = len(combined_dict)
        split_str = "+".join(sorted(loaded_data_splits.keys()))
        print(f"\n{'─' * 60}")
        print(f"Processing {combined_name} ({split_str}, {n_total} subjects total)")
        print(f"{'─' * 60}")
        run_all_plots(combined_dict, combined_out, dataset_name=combined_name)

    # ── 3. Accuracy analysis ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("ACCURACY ANALYSIS")
    print(f"{'═' * 60}")
    all_acc = load_all_accuracies(dataset_names)

    if all_acc:
        # BCI4 accuracy plots (scatter + box + stats)
        if 'BCI4' in all_acc:
            ds_out = os.path.join(output_root, 'BCI4')
            os.makedirs(ds_out, exist_ok=True)
            plot_accuracy_scatter(all_acc['BCI4'], ds_out, 'BCI4')
            plot_accuracy_boxplots(all_acc['BCI4'], ds_out, 'BCI4')
            run_statistical_analysis(all_acc['BCI4'], ds_out, 'BCI4')

        # DATA_Combined accuracy plots (pool all DATA1-5 subjects)
        data_recs_combined = []
        for k in ALL_DATA_KEYS:
            data_recs_combined.extend(all_acc.get(k, []))
        if data_recs_combined:
            ds_out = os.path.join(output_root, 'DATA_Combined')
            os.makedirs(ds_out, exist_ok=True)
            plot_accuracy_scatter(data_recs_combined, ds_out, 'DATA_Combined')
            plot_accuracy_boxplots(data_recs_combined, ds_out, 'DATA_Combined')
            run_statistical_analysis(data_recs_combined, ds_out, 'DATA_Combined')

        # Combined high/low performer box plots (BCI4 + DATA1-5 together)
        if 'BCI4' in all_acc and data_recs_combined:
            combined_acc_out = os.path.join(output_root, 'combined_accuracy')
            plot_accuracy_boxplots_combined(all_acc, combined_acc_out)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"All plots saved under '{output_root}/'")
    print(f"  Neurophysiology plots:")
    if 'BCI4' in all_ds_data:
        print(f"    {output_root}/BCI4/")
    if loaded_data_splits:
        print(f"    {output_root}/DATA_Combined/")
    if all_acc:
        print(f"  Accuracy analysis:")
        for ds_name in all_acc:
            print(f"    {output_root}/{ds_name}/")
        if len(all_acc) > 1 or any(k in all_acc for k in ALL_DATA_KEYS):
            print(f"    {output_root}/combined_accuracy/")
    print(f"{'=' * 70}")