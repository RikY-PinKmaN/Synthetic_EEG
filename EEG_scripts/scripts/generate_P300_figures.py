#!/usr/bin/env python3
"""
P300 cWGAN-GP — Combined Visualization & Evaluation
=====================================================
Generates publication-quality figures from P300 pipeline results:
  1. Grand Average ERP at Cz (per condition, across subjects)
  2. Grand Average PSD (per condition)
  3. P300 Topoplots — amplitude maps (real vs synthetic)
  4. Accuracy scatter plots (Baseline vs SynthOnly, Baseline vs Best-Aug)
  5. Accuracy box plots (per condition + combined)
  6. Statistical analysis (paired t-test, Wilcoxon, Cohen's d)
  7. Data distribution (amplitude histogram, per-channel std, PSD overlay)
  8. P300 peak latency/amplitude analysis (real vs synthetic)

Usage:
  python generate_P300_figures.py                          # All subjects, all conditions
  python generate_P300_figures.py -s 1 2 3                 # Subjects H1, H2, H3
  python generate_P300_figures.py -p 0 2                   # Pairs 0 (3x3-117) and 2 (5x5-117)
  python generate_P300_figures.py -s 1 -p 0 -o my_output
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr, wilcoxon, ttest_rel, sem
import mne
import csv

mne.set_log_level('ERROR')

# ── Configuration ─────────────────────────────────────────────────────────────
P300_FS = 80
P300_NCHAN = 8
P300_CH_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']
P300_ERP_TMIN = -0.1
P300_ERP_TMAX = 0.8
P300_FEAT_TMIN = 0.15
P300_FEAT_TMAX = 0.5
CZ_IDX = 1  # Cz channel index

CONDITIONS = [
    {'idx': 0, 'label': '3x3_117ms', 'matrix': '3x3', 'isi': 117},
    {'idx': 1, 'label': '3x3_175ms', 'matrix': '3x3', 'isi': 175},
    {'idx': 2, 'label': '5x5_117ms', 'matrix': '5x5', 'isi': 117},
    {'idx': 3, 'label': '5x5_175ms', 'matrix': '5x5', 'isi': 175},
]

RESULTS_DIR = 'pr'


# ── MNE Info for topoplots ───────────────────────────────────────────────────
def get_mne_info():
    info = mne.create_info(ch_names=P300_CH_NAMES, sfreq=P300_FS, ch_types=['eeg'] * P300_NCHAN)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    return info


# ── Data Loading ──────────────────────────────────────────────────────────────

def discover_subjects_conditions(results_dir=RESULTS_DIR):
    """Scan results directory for Subject_Hx_condition folders.
    Returns dict: {(subject_num, cond_label): folder_path}"""
    found = {}
    if not os.path.isdir(results_dir):
        return found
    for entry in os.listdir(results_dir):
        if entry.startswith('Subject_H') and os.path.isdir(os.path.join(results_dir, entry)):
            # e.g. Subject_H1_3x3_117ms
            parts = entry.replace('Subject_', '').split('_', 1)
            if len(parts) == 2:
                subj_str, cond_label = parts[0], parts[1]
                try:
                    subj_num = int(subj_str[1:])  # H1 -> 1
                    found[(subj_num, cond_label)] = os.path.join(results_dir, entry)
                except ValueError:
                    pass
    return found


def load_subject_condition(folder_path, subj_name, cond_label):
    """Load real + synthetic data for one subject-condition."""
    mat_path = os.path.join(folder_path, f'synthetic_{subj_name}_{cond_label}.mat')
    if not os.path.exists(mat_path):
        return None

    m = scipy.io.loadmat(mat_path)
    synth_tgt_x = m.get('synthetic_tgt_x', None)
    synth_tgt_y = m.get('synthetic_tgt_y', None)
    real_x = m.get('real_x', None)
    real_y = m.get('real_y', None)

    if synth_tgt_x is None:
        return None

    synth_tgt_y = synth_tgt_y.flatten() if synth_tgt_y is not None else np.ones(synth_tgt_x.shape[0]) * 2
    if real_y is not None:
        real_y = real_y.flatten()

    def _scalar(key):
        v = m.get(key, None)
        if v is None: return np.nan
        return float(np.asarray(v).flatten()[0])

    def _string(key):
        v = m.get(key, '')
        if isinstance(v, np.ndarray): v = str(v.flatten()[0]) if v.size > 0 else ''
        return str(v)

    ch_mean = m.get('norm_ch_mean', None)
    ch_std = m.get('norm_ch_std', None)

    return {
        'synth_tgt_x': synth_tgt_x,   # (N_tgt, ch, T) — synthetic target trials
        'synth_tgt_y': synth_tgt_y,
        'real_x': real_x,              # (n_trials, ch, T) — all real training data
        'real_y': real_y,              # labels (1=non-target, 2=target)
        'ch_mean': ch_mean,
        'ch_std': ch_std,
        'oversample_bl_acc': _scalar('oversampled_bl_acc'),
        'synth_only_acc': _scalar('synth_only_acc'),
        'balanced_aug_acc': _scalar('real_synth_acc'),
    }


def load_all_data(subject_nums=None, cond_labels=None, results_dir=RESULTS_DIR):
    """Load all available data, optionally filtered by subjects and conditions.

    Returns:
        data: dict {cond_label: {subj_num: {...}}}
        acc_records: list of dicts for accuracy analysis
    """
    found = discover_subjects_conditions(results_dir)
    if not found:
        print(f"ERROR: No Subject_* folders found in '{results_dir}'")
        return {}, []

    data = {}
    acc_records = []

    for (subj_num, cond_label), folder_path in sorted(found.items()):
        if subject_nums is not None and subj_num not in subject_nums:
            continue
        if cond_labels is not None and cond_label not in cond_labels:
            continue

        subj_name = f'H{subj_num}'
        d = load_subject_condition(folder_path, subj_name, cond_label)
        if d is None:
            print(f"  SKIP: {subj_name}_{cond_label} — mat file not found or invalid")
            continue

        if cond_label not in data:
            data[cond_label] = {}
        data[cond_label][subj_num] = d

        acc_records.append({
            'sid': f'{subj_name}_{cond_label}',
            'subj_num': subj_num,
            'cond_label': cond_label,
            'oversample_bl': d['oversample_bl_acc'],
            'synth_only': d['synth_only_acc'],
            'balanced_aug': d['balanced_aug_acc'],
        })

        print(f"  Loaded {subj_name}_{cond_label}: synth={d['synth_tgt_x'].shape}, "
              f"OversampBL={d['oversample_bl_acc']:.1f}% BalancedAug={d['balanced_aug_acc']:.1f}%")

    return data, acc_records


def apply_gain_correction(synth_x, real_x, real_y):
    """Per-channel std matching for display only. synth_x is target class."""
    gc = synth_x.copy()
    real_tgt = real_x[real_y == 2] if real_y is not None else real_x
    if real_tgt.shape[0] < 2:
        return gc
    for ch in range(gc.shape[1]):
        r_std = np.std(real_tgt[:, ch, :])
        s_std = np.std(gc[:, ch, :]) + 1e-8
        gc[:, ch, :] *= (r_std / s_std)
    return gc


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRAND AVERAGE ERP AT Cz
# ══════════════════════════════════════════════════════════════════════════════

def plot_grand_average_erp(data, output_dir):
    """One plot per condition: real target, real non-target, synth target at Cz.
    Grand average across subjects with SEM shading. Gain correction on synthetic."""
    print("Generating Grand Average ERPs...")

    for cond_label, subj_dict in data.items():
        if len(subj_dict) == 0:
            continue

        first = next(iter(subj_dict.values()))
        T = first['synth_tgt_x'].shape[2]
        t_axis = np.arange(T) / P300_FS + P300_ERP_TMIN

        rt_means, rnt_means, st_means = [], [], []

        for sid, d in sorted(subj_dict.items()):
            st = d['synth_tgt_x']   # (N_tgt, ch, T)
            rx = d.get('real_x')    # (n_trials, ch, T)
            ry = d.get('real_y')

            if rx is None or ry is None:
                continue

            rt = rx[ry == 2]  # real target
            rnt = rx[ry == 1]  # real non-target

            # Gain correction on synthetic for display
            st_gc = st.copy()
            if rt.shape[0] >= 2 and st_gc.shape[0] >= 1:
                for ch in range(st_gc.shape[1]):
                    r_std = np.std(rt[:, ch, :])
                    s_std = np.std(st_gc[:, ch, :]) + 1e-8
                    st_gc[:, ch, :] *= (r_std / s_std)

            if rt.shape[0] >= 1:
                rt_means.append(np.mean(rt[:, CZ_IDX, :], axis=0))
            if rnt.shape[0] >= 1:
                rnt_means.append(np.mean(rnt[:, CZ_IDX, :], axis=0))
            if st_gc.shape[0] >= 1:
                st_means.append(np.mean(st_gc[:, CZ_IDX, :], axis=0))

        if not rt_means and not st_means:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        if rt_means:
            ga = np.mean(rt_means, axis=0)
            ga_sem_v = sem(rt_means, axis=0) if len(rt_means) > 1 else np.zeros_like(ga)
            ax.plot(t_axis, ga, '#2166AC', lw=2.5, label=f'Real Target (n={len(rt_means)})')
            ax.fill_between(t_axis, ga - ga_sem_v, ga + ga_sem_v, color='#2166AC', alpha=0.15)

        if rnt_means:
            ga = np.mean(rnt_means, axis=0)
            ga_sem_v = sem(rnt_means, axis=0) if len(rnt_means) > 1 else np.zeros_like(ga)
            ax.plot(t_axis, ga, '#4DAF4A', lw=2.5, label=f'Real Non-target (n={len(rnt_means)})')
            ax.fill_between(t_axis, ga - ga_sem_v, ga + ga_sem_v, color='#4DAF4A', alpha=0.15)

        if st_means:
            ga = np.mean(st_means, axis=0)
            ga_sem_v = sem(st_means, axis=0) if len(st_means) > 1 else np.zeros_like(ga)
            ax.plot(t_axis, ga, '#B2182B', lw=2.5, ls='--',
                    label=f'Synth Target (n={len(st_means)})')
            ax.fill_between(t_axis, ga - ga_sem_v, ga + ga_sem_v, color='#B2182B', alpha=0.15)

        ax.axvline(0, color='k', ls='--', lw=0.5, alpha=0.5)
        ax.axvspan(P300_FEAT_TMIN, P300_FEAT_TMAX, alpha=0.06, color='green', label='P300 window')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (normalized)', fontsize=12)
        ax.set_title(f'Grand Average ERP at Cz — {cond_label}\n'
                     f'({len(subj_dict)} subjects)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'grand_avg_erp_{cond_label}.png'), dpi=150)
        plt.close(fig)
        print(f"  {cond_label}: {len(subj_dict)} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAND AVERAGE PSD
# ══════════════════════════════════════════════════════════════════════════════

def plot_grand_average_psd(data, output_dir):
    """PSD at Cz: real target vs synthetic target, grand averaged across subjects."""
    print("Generating Grand Average PSD...")

    for cond_label, subj_dict in data.items():
        all_psd_real, all_psd_synth = [], []
        freqs_out = None

        for sid, d in sorted(subj_dict.items()):
            st = d['synth_tgt_x']
            rx = d.get('real_x')
            ry = d.get('real_y')
            nperseg = min(32, st.shape[2])

            # Synthetic target PSD
            for t in range(st.shape[0]):
                f, psd = welch(st[t, CZ_IDX, :], fs=P300_FS, nperseg=nperseg)
                all_psd_synth.append(psd)
                freqs_out = f

            # Real target PSD
            if rx is not None and ry is not None:
                rt = rx[ry == 2]
                for t in range(rt.shape[0]):
                    f, psd = welch(rt[t, CZ_IDX, :], fs=P300_FS, nperseg=nperseg)
                    all_psd_real.append(psd)

        if not all_psd_synth or freqs_out is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        if all_psd_real:
            mean_r = np.mean(all_psd_real, axis=0)
            std_r = np.std(all_psd_real, axis=0)
            ax.semilogy(freqs_out, mean_r, '#2166AC', lw=2, label='Real Target')
            ax.fill_between(freqs_out, mean_r - std_r, mean_r + std_r,
                            color='#2166AC', alpha=0.15)

        mean_s = np.mean(all_psd_synth, axis=0)
        std_s = np.std(all_psd_synth, axis=0)
        ax.semilogy(freqs_out, mean_s, '#B2182B', lw=2, label='Synth Target')
        ax.fill_between(freqs_out, mean_s - std_s, mean_s + std_s,
                        color='#B2182B', alpha=0.15)

        # Correlation annotation
        if all_psd_real:
            mask = freqs_out <= 30
            r, p = pearsonr(np.mean(all_psd_real, axis=0)[mask],
                            np.mean(all_psd_synth, axis=0)[mask])
            ax.text(0.98, 0.95, f'r = {r:.4f} (0-30 Hz)', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('PSD (log)', fontsize=12)
        ax.set_title(f'PSD at Cz — {cond_label} ({len(subj_dict)} subjects)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 40])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'grand_avg_psd_{cond_label}.png'), dpi=150)
        plt.close(fig)
        print(f"  {cond_label}: PSD saved")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPOPLOTS — P300 AMPLITUDE
# ══════════════════════════════════════════════════════════════════════════════

def plot_topoplots(data, output_dir):
    """P300 window amplitude topoplots: real target vs synthetic target per condition."""
    print("Generating Topoplots...")
    info = get_mne_info()

    erp_ts = np.arange(round(P300_ERP_TMIN * P300_FS), round(P300_ERP_TMAX * P300_FS) + 1)
    win_s = np.searchsorted(erp_ts, round(P300_FEAT_TMIN * P300_FS))
    win_e = np.searchsorted(erp_ts, round(P300_FEAT_TMAX * P300_FS))

    for cond_label, subj_dict in data.items():
        all_real_amp, all_synth_amp = [], []
        for sid, d in sorted(subj_dict.items()):
            st = d['synth_tgt_x']
            rx = d.get('real_x')
            ry = d.get('real_y')

            # Synthetic target mean amplitude in P300 window
            amp_s = np.mean(st[:, :, win_s:win_e], axis=(0, 2))
            all_synth_amp.append(amp_s)

            # Real target
            if rx is not None and ry is not None:
                rt = rx[ry == 2]
                if rt.shape[0] >= 1:
                    amp_r = np.mean(rt[:, :, win_s:win_e], axis=(0, 2))
                    all_real_amp.append(amp_r)

        if not all_synth_amp:
            continue

        ga_synth = np.mean(all_synth_amp, axis=0)

        if all_real_amp:
            ga_real = np.mean(all_real_amp, axis=0)
            vmax = max(np.max(np.abs(ga_real)), np.max(np.abs(ga_synth)))

            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            kwargs = dict(show=False, cmap='RdBu_r', vlim=(-vmax, vmax), extrapolate='local')
            mne.viz.plot_topomap(ga_real, info, axes=axes[0], **kwargs)
            im, _ = mne.viz.plot_topomap(ga_synth, info, axes=axes[1], **kwargs)
            axes[0].set_title('Real Target')
            axes[1].set_title('Synthetic Target')
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04,
                         label='Mean Amplitude (P300 window)')
            fig.suptitle(f'P300 Amplitude Topoplot — {cond_label}\n'
                         f'({len(subj_dict)} subjects)', fontsize=13)
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            im, _ = mne.viz.plot_topomap(ga_synth, info, axes=ax, show=False,
                                          cmap='RdBu_r', extrapolate='local')
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='Mean Amplitude')
            ax.set_title(f'P300 Amplitude — {cond_label} (Synth Target)', fontsize=12)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'topoplot_{cond_label}.png'), dpi=150)
        plt.close(fig)
        print(f"  {cond_label}: topoplot saved")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ACCURACY SCATTER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_scatter(records, output_dir, title_suffix=''):
    """Scatter: (a) Oversample BL vs SynthOnly, (b) Oversample BL vs Balanced-Aug."""
    if len(records) < 2:
        return
    print(f"Generating Accuracy Scatter ({title_suffix})...")

    bl = np.array([r['oversample_bl'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['balanced_aug'] for r in records])
    sids = [r['sid'] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, comp, comp_label, color in [
        (axes[0], so, 'Synth-Only', '#2CA02C'),
        (axes[1], ba, 'Balanced Aug', '#D62728'),
    ]:
        lo = min(bl.min(), comp.min()) - 3
        hi = max(bl.max(), comp.max()) + 3
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=1, label='y = x')
        ax.scatter(bl, comp, s=70, c=color, edgecolors='k', linewidths=0.5, zorder=5)
        for i, sid in enumerate(sids):
            ax.annotate(sid.split('_')[0], (bl[i], comp[i]), textcoords='offset points',
                        xytext=(5, 5), fontsize=7)
        ax.set_xlabel('Oversample BL Accuracy (%)', fontsize=11)
        ax.set_ylabel(f'{comp_label} Accuracy (%)', fontsize=11)
        ax.set_title(f'Oversample BL vs {comp_label}', fontsize=12)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    n = len(records)
    above = np.sum(ba > bl)
    fig.suptitle(f'Accuracy Scatter — {title_suffix} ({n} subject-conditions, '
                 f'{above}/{n} improved)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_scatter_{title_suffix}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ACCURACY BOX PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _boxplot_three(ax, bl, so, ba, title):
    labels = ['Oversample BL', 'Synth-Only', 'Balanced Aug']
    colors = ['#1F77B4', '#2CA02C', '#D62728']
    data_list = [bl, so, ba]
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    for i, d in enumerate(data_list):
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(d))
        ax.scatter(np.full_like(d, i + 1) + jitter, d, s=25, c='k', alpha=0.5, zorder=5)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    for i, d in enumerate(data_list):
        ax.text(i + 1, ax.get_ylim()[1] * 0.98, f'μ={np.mean(d):.1f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def plot_accuracy_boxplots(records, output_dir, title_suffix=''):
    """Box plots: per condition + combined."""
    if len(records) < 2:
        return
    print(f"Generating Accuracy Box Plots ({title_suffix})...")

    bl = np.array([r['oversample_bl'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['balanced_aug'] for r in records])

    # Combined box plot
    fig, ax = plt.subplots(figsize=(7, 6))
    _boxplot_three(ax, bl, so, ba, f'{title_suffix} — All (n={len(records)})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_{title_suffix}.png'), dpi=150)
    plt.close(fig)

    # Per-condition box plots (side by side)
    cond_labels = sorted(set(r['cond_label'] for r in records))
    if len(cond_labels) > 1:
        fig, axes = plt.subplots(1, len(cond_labels), figsize=(5 * len(cond_labels), 6))
        if len(cond_labels) == 1:
            axes = [axes]
        for ax_i, cl in enumerate(cond_labels):
            recs_cl = [r for r in records if r['cond_label'] == cl]
            bl_cl = np.array([r['oversample_bl'] for r in recs_cl])
            so_cl = np.array([r['synth_only'] for r in recs_cl])
            ba_cl = np.array([r['balanced_aug'] for r in recs_cl])
            _boxplot_three(axes[ax_i], bl_cl, so_cl, ba_cl, f'{cl} (n={len(recs_cl)})')
        fig.suptitle(f'Accuracy by Condition — {title_suffix}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_per_cond_{title_suffix}.png'), dpi=150)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(records, output_dir, title_suffix=''):
    """Paired tests + summary figure."""
    if len(records) < 3:
        return
    print(f"Running Statistical Analysis ({title_suffix})...")

    bl = np.array([r['oversample_bl'] for r in records])
    so = np.array([r['synth_only'] for r in records])
    ba = np.array([r['balanced_aug'] for r in records])
    n = len(records)

    lines = []
    lines.append('=' * 70)
    lines.append(f'Statistical Analysis — {title_suffix} (n={n})')
    lines.append('=' * 70)
    lines.append('')
    lines.append('Descriptive Statistics (%):')
    for name, arr in [('Oversample BL', bl), ('Synth-Only', so), ('Balanced Aug', ba)]:
        lines.append(f'  {name:<15} mean={np.mean(arr):.2f}  std={np.std(arr):.2f}  '
                     f'median={np.median(arr):.2f}  min={np.min(arr):.2f}  max={np.max(arr):.2f}')
    lines.append('')

    # Per-subject table
    lines.append('Per-Subject Accuracies (%):')
    lines.append(f'  {"Subject":<25} {"OversampBL":>10} {"SynthOnly":>10} {"BalancedAug":>10} {"Δ(Balanced-BL)":>12}')
    lines.append('  ' + '-' * 80)
    for r in records:
        delta = r['balanced_aug'] - r['oversample_bl']
        lines.append(f'  {r["sid"]:<25} {r["oversample_bl"]:>10.2f} {r["synth_only"]:>10.2f} '
                     f'{r["balanced_aug"]:>10.2f} {delta:>+12.2f}')
    lines.append('')

    # Pairwise tests
    comparisons = [
        ('Oversample BL vs Synth-Only', bl, so),
        ('Oversample BL vs Balanced Aug', bl, ba),
        ('Synth-Only vs Balanced Aug', so, ba),
    ]

    test_results = []
    lines.append('Paired Statistical Tests:')
    lines.append('-' * 70)
    for comp_name, a, b in comparisons:
        diff = b - a
        mean_diff = np.mean(diff)

        if n >= 2 and np.std(diff) > 1e-10:
            t_stat, t_p = ttest_rel(a, b)
        else:
            t_stat, t_p = np.nan, np.nan

        nonzero = diff[np.abs(diff) > 1e-10]
        if len(nonzero) >= 6:
            try:
                w_stat, w_p = wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan

        cohens_d = mean_diff / np.std(diff) if np.std(diff) > 1e-10 else np.nan
        sig_t = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'n.s.'
        sig_w = '***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'n.s.'

        lines.append(f'\n  {comp_name}:')
        lines.append(f'    Mean difference: {mean_diff:+.2f}%')
        lines.append(f'    Paired t-test:   t={t_stat:.4f}, p={t_p:.6f}  {sig_t}')
        lines.append(f'    Wilcoxon:        W={w_stat:.1f}, p={w_p:.6f}  {sig_w}' if not np.isnan(w_stat)
                     else f'    Wilcoxon:        n/a (need ≥6 non-zero differences)')
        lines.append(f"    Cohen's d:       {cohens_d:.4f}" if not np.isnan(cohens_d)
                     else f"    Cohen's d:       n/a")

        test_results.append({
            'name': comp_name, 'mean_diff': mean_diff,
            't_stat': t_stat, 't_p': t_p, 'sig_t': sig_t,
            'w_stat': w_stat, 'w_p': w_p, 'sig_w': sig_w,
            'cohens_d': cohens_d,
        })

    lines.append('')
    lines.append('Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant')
    lines.append('=' * 70)

    # Save text report
    report_path = os.path.join(output_dir, f'stats_report_{title_suffix}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    for line in lines:
        print(f'  {line}')

    # Summary figure: mean accuracy bars + per-subject improvement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    means = [np.mean(bl), np.mean(so), np.mean(ba)]
    stds = [np.std(bl), np.std(so), np.std(ba)]
    colors = ['#1F77B4', '#2CA02C', '#D62728']
    x_pos = np.arange(3)
    bars = axes[0].bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(['Oversample BL', 'Synth-Only', 'Balanced Aug'], fontsize=11)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title(f'Mean Accuracy ± Std (n={n})', fontsize=12)
    axes[0].grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 1, f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Per-subject improvement bar
    diff_ba = ba - bl
    sorted_idx = np.argsort(diff_ba)
    diff_sorted = diff_ba[sorted_idx]
    sids_sorted = [records[i]['sid'].split('_')[0] for i in sorted_idx]
    colors_diff = ['#D62728' if d < 0 else '#2CA02C' for d in diff_sorted]
    y_pos = np.arange(len(diff_sorted))
    axes[1].barh(y_pos, diff_sorted, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sids_sorted, fontsize=7)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Δ Accuracy (Balanced Aug − Oversample BL) (%)', fontsize=11)
    axes[1].set_title('Per-Subject Improvement', fontsize=12)
    axes[1].grid(alpha=0.3, axis='x')

    for tr in test_results:
        if tr['name'] == 'Oversample BL vs Balanced Aug':
            sig_label = f"p={tr['t_p']:.4f} ({tr['sig_t']})" if not np.isnan(tr['t_p']) else 'n/a'
            axes[1].text(0.98, 0.02, f"Paired t: {sig_label}\nCohen's d: {tr['cohens_d']:.3f}",
                         transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            break

    fig.suptitle(f'Statistical Summary — {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stats_summary_{title_suffix}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. DATA DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_data_distribution(data, output_dir, title_suffix=''):
    """Amplitude histogram + per-channel std + PSD for synthetic target data."""
    print(f"Generating Data Distribution ({title_suffix})...")

    all_vals = []
    all_psd = []
    ch_stds = []
    freqs_out = None

    for cond_label, subj_dict in data.items():
        for sid, d in subj_dict.items():
            st = d['synth_tgt_x']
            all_vals.append(st.flatten())
            ch_stds.append([np.std(st[:, ch, :]) for ch in range(P300_NCHAN)])
            nperseg = min(32, st.shape[2])
            f, psd = welch(st.reshape(-1, st.shape[2]), fs=P300_FS, nperseg=nperseg, axis=1)
            all_psd.append(np.mean(psd, axis=0))
            freqs_out = f

    if not all_vals:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Amplitude histogram
    vals = np.concatenate(all_vals)
    bins = np.linspace(-1, 1, 100)
    axes[0].hist(vals, bins=bins, alpha=0.7, density=True, color='#B2182B', label='Synth Target')
    axes[0].set_xlabel('Normalized Amplitude', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Amplitude Distribution', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # (b) Per-channel std
    mean_std = np.mean(ch_stds, axis=0)
    x_pos = np.arange(P300_NCHAN)
    axes[1].bar(x_pos, mean_std, color='#B2182B', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(P300_CH_NAMES, rotation=45, ha='right', fontsize=9)
    axes[1].set_xlabel('Channel', fontsize=11)
    axes[1].set_ylabel('Std Dev', fontsize=11)
    axes[1].set_title('Per-Channel Std Dev', fontsize=12)
    axes[1].grid(alpha=0.3, axis='y')

    # (c) PSD overlay
    mean_psd = np.mean(all_psd, axis=0)
    std_psd = np.std(all_psd, axis=0)
    axes[2].semilogy(freqs_out, mean_psd, '#B2182B', lw=2, label='Synth Target')
    axes[2].fill_between(freqs_out, mean_psd - std_psd, mean_psd + std_psd,
                         color='#B2182B', alpha=0.15)
    axes[2].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[2].set_ylabel('PSD (log)', fontsize=11)
    axes[2].set_title('PSD', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 40])

    fig.suptitle(f'Data Distribution — Synthetic Target — {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_{title_suffix}.png'), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8. P300 PEAK LATENCY/AMPLITUDE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_p300_peak_analysis(data, output_dir, title_suffix=''):
    """Measure P300 peak amplitude and latency at Cz for synthetic targets.
    Scatter across subjects per condition."""
    print(f"Generating P300 Peak Analysis ({title_suffix})...")

    erp_ts = np.arange(round(P300_ERP_TMIN * P300_FS), round(P300_ERP_TMAX * P300_FS) + 1)
    win_s = np.searchsorted(erp_ts, round(P300_FEAT_TMIN * P300_FS))
    win_e = np.searchsorted(erp_ts, round(P300_FEAT_TMAX * P300_FS))
    t_axis = np.arange(len(erp_ts)) / P300_FS + P300_ERP_TMIN

    all_peaks = []
    for cond_label, subj_dict in data.items():
        for sid, d in sorted(subj_dict.items()):
            st = d['synth_tgt_x']
            if st.shape[0] < 1:
                continue
            # Grand average at Cz
            ga = np.mean(st[:, CZ_IDX, :], axis=0)
            # Peak in P300 window
            win_data = ga[win_s:win_e]
            peak_idx = np.argmax(np.abs(win_data))  # strongest deflection
            peak_amp = win_data[peak_idx]
            peak_lat = (win_s + peak_idx) / P300_FS + P300_ERP_TMIN
            all_peaks.append({
                'sid': f'H{sid}',
                'cond': cond_label,
                'amplitude': peak_amp,
                'latency': peak_lat,
            })

    if len(all_peaks) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Peak amplitude by condition
    conds = sorted(set(p['cond'] for p in all_peaks))
    for i, cl in enumerate(conds):
        amps = [p['amplitude'] for p in all_peaks if p['cond'] == cl]
        jitter = np.random.RandomState(42 + i).uniform(-0.15, 0.15, size=len(amps))
        axes[0].scatter(np.full(len(amps), i) + jitter, amps, s=40, alpha=0.7, label=cl)
    axes[0].set_xticks(range(len(conds)))
    axes[0].set_xticklabels(conds, fontsize=9, rotation=20)
    axes[0].set_ylabel('Peak Amplitude (normalized)', fontsize=11)
    axes[0].set_title('P300 Peak Amplitude at Cz', fontsize=12)
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].legend(fontsize=8)

    # (b) Peak latency by condition
    for i, cl in enumerate(conds):
        lats = [p['latency'] * 1000 for p in all_peaks if p['cond'] == cl]
        jitter = np.random.RandomState(42 + i).uniform(-0.15, 0.15, size=len(lats))
        axes[1].scatter(np.full(len(lats), i) + jitter, lats, s=40, alpha=0.7, label=cl)
    axes[1].set_xticks(range(len(conds)))
    axes[1].set_xticklabels(conds, fontsize=9, rotation=20)
    axes[1].set_ylabel('Peak Latency (ms)', fontsize=11)
    axes[1].set_title('P300 Peak Latency at Cz', fontsize=12)
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].legend(fontsize=8)

    fig.suptitle(f'P300 Peak Analysis — Synthetic Target — {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'p300_peak_{title_suffix}.png'), dpi=150)
    plt.close(fig)

    # Save peak data to CSV
    csv_path = os.path.join(output_dir, f'p300_peaks_{title_suffix}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'condition', 'peak_amplitude', 'peak_latency_ms'])
        for p in all_peaks:
            w.writerow([p['sid'], p['cond'], f"{p['amplitude']:.4f}", f"{p['latency']*1000:.1f}"])
    print(f"  Peak data saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='P300 cWGAN-GP Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                            # All subjects, all conditions
  %(prog)s -s 1 2 3                   # Subjects H1, H2, H3
  %(prog)s -p 0 2                     # Pairs 0 (3x3-117) and 2 (5x5-117)
  %(prog)s -s 1 -p 0 -o my_output
  %(prog)s --results-dir results_P300_vnew""")

    parser.add_argument('-s', '--subjects', type=int, nargs='+', default=None,
                        help='Subject numbers (1-12). Default: all available')
    parser.add_argument('-p', '--pairs', type=int, nargs='+', default=None,
                        help='Pair/condition indices (0-3). Default: all')
    parser.add_argument('-o', '--output-dir', type=str, default='visualization_P300',
                        help='Output directory')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Pipeline results directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Map pair indices to condition labels
    cond_labels = None
    if args.pairs is not None:
        cond_labels = [CONDITIONS[i]['label'] for i in args.pairs if 0 <= i <= 3]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('P300 cWGAN-GP Visualization')
    print(f'  Subjects:    {args.subjects if args.subjects else "All"}')
    print(f'  Conditions:  {cond_labels if cond_labels else "All"}')
    print(f'  Results dir: {args.results_dir}')
    print(f'  Output dir:  {output_dir}')
    print('=' * 70)

    data, acc_records = load_all_data(
        subject_nums=args.subjects,
        cond_labels=cond_labels,
        results_dir=args.results_dir
    )

    if not data:
        print('ERROR: No data loaded.')
        sys.exit(1)

    n_subj_cond = sum(len(v) for v in data.values())
    n_conds = len(data)
    print(f'\nLoaded {n_subj_cond} subject-conditions across {n_conds} conditions')

    # ── Neurophysiology plots ──
    plot_grand_average_erp(data, output_dir)
    plot_grand_average_psd(data, output_dir)
    plot_topoplots(data, output_dir)
    plot_data_distribution(data, output_dir, title_suffix='P300')
    plot_p300_peak_analysis(data, output_dir, title_suffix='P300')

    # ── Accuracy analysis ──
    if len(acc_records) >= 2:
        print(f'\n{"═" * 60}')
        print('ACCURACY ANALYSIS')
        print(f'{"═" * 60}')

        plot_accuracy_scatter(acc_records, output_dir, 'P300_All')
        plot_accuracy_boxplots(acc_records, output_dir, 'P300')
        run_statistical_analysis(acc_records, output_dir, 'P300')

        # Per-condition analysis
        for cl in sorted(set(r['cond_label'] for r in acc_records)):
            recs_cl = [r for r in acc_records if r['cond_label'] == cl]
            if len(recs_cl) >= 3:
                cond_dir = os.path.join(output_dir, cl)
                os.makedirs(cond_dir, exist_ok=True)
                plot_accuracy_scatter(recs_cl, cond_dir, cl)
                run_statistical_analysis(recs_cl, cond_dir, cl)

    # ── Summary ──
    print(f'\n{"=" * 70}')
    print(f"All plots saved under '{output_dir}/'")
    print(f'{"=" * 70}')