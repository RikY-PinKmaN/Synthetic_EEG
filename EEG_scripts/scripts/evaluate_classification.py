"""
evaluate_classification.py — CSP-SVM for BCI IV 2a (Proper Validation Selection)

Selection procedure:
  1. For each mixing ratio: train on TRAIN + synth → evaluate on VALIDATION
  2. Select best ratio based on validation accuracy
  3. Retrain on TRAIN+VALID + synth at best ratio → evaluate on TEST (single shot)

Also reports all ratios on test for transparency, but marks which was selected via validation.

Usage:
    python evaluate_classification.py --train_mat data1.mat --test_mat data2.mat \
        --pattern "ResUlTS/Subject_*_D1T_50trials" --trials 50
"""

import numpy as np
import scipy.io
import scipy.signal as sig_mod
from scipy.linalg import eigh
from sklearn.svm import SVC
import os
import argparse
import glob
import csv
import traceback

# ============================================================
# CONFIGURATION
# ============================================================
FS = 250
LOWCUT = 8
HIGHCUT = 35
TRAIN_START = 125
TRAIN_END = 624
TEST_START = 115
TEST_END = 614
NUM_VALID_PER_CLASS = 10

SELECTED_CHANNELS_1_BASED = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17]
SELECTED_CHANNELS_0_BASED = [ch - 1 for ch in SELECTED_CHANNELS_1_BASED]

SEED = 42
MIX_RATIOS = [0.25, 0.5, 0.75, 1.0]


# ============================================================
# PREPROCESSING
# ============================================================

def elliptical_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, rp=1, rs=40):
    nyq = 0.5 * fs
    low_stop = max(0.1, lowcut - 1.0)
    high_stop = highcut + 1.0
    if high_stop >= nyq:
        high_stop = nyq - 0.1
    wp = [lowcut / nyq, highcut / nyq]
    ws = [low_stop / nyq, high_stop / nyq]
    eps = 1e-6
    wp = np.clip(wp, eps, 1.0 - eps).tolist()
    ws = np.clip(ws, eps, 1.0 - eps).tolist()
    if ws[0] >= wp[0]:
        ws[0] = wp[0] * 0.9 if wp[0] > eps * 10 else wp[0] * 0.5
        ws[0] = max(ws[0], eps)
    if ws[1] <= wp[1]:
        ws[1] = wp[1] * 1.1 if wp[1] < (1.0 - eps * 10) else (wp[1] + (1.0 - wp[1]) * 0.5)
        ws[1] = min(ws[1], 1.0 - eps)
    n, wn = sig_mod.ellipord(wp, ws, rp, rs)
    b, a = sig_mod.ellip(n, rp, rs, wn, btype='band')
    return sig_mod.filtfilt(b, a, data, axis=0)


def extract_field(struct, field):
    if not hasattr(struct, 'dtype') or struct.dtype.names is None:
        return None
    if field not in struct.dtype.names:
        return None
    content = struct[field]
    if isinstance(content, np.ndarray) and content.shape == (1, 1) and hasattr(content[0, 0], 'shape'):
        return content[0, 0]
    return content


# ============================================================
# CSP-SVM
# ============================================================

def train_cspsvm(data):
    if data is None or data['x'].shape[2] == 0:
        return None, None
    c1 = np.where(data['y'] == 1)[0]
    c2 = np.where(data['y'] == 2)[0]
    if len(c1) == 0 or len(c2) == 0:
        return None, None

    X1, X2 = data['x'][:, :, c1], data['x'][:, :, c2]
    n_ch = X1.shape[1]

    def avg_cov(X):
        cov = np.zeros((n_ch, n_ch))
        v = 0
        for t in range(X.shape[2]):
            c = np.cov(X[:, :, t], rowvar=False)
            if not np.all(np.isnan(c)):
                cov += c; v += 1
        return cov / v if v > 0 else cov

    cov1, cov2 = avg_cov(X1), avg_cov(X2)
    reg = 1e-9
    try:
        evals, evecs = eigh(cov1 + reg * np.eye(n_ch), cov1 + cov2 + 2 * reg * np.eye(n_ch))
    except np.linalg.LinAlgError:
        evals, evecs = eigh(cov1 + reg * np.eye(n_ch))

    W = evecs[:, np.argsort(evals)[::-1]].T
    sw = np.vstack((W[0:3, :], W[-3:, :])) if W.shape[0] >= 6 else (
        np.vstack((W[0:1, :], W[-1:, :])) if W.shape[0] >= 2 else W)

    n_trials, n_feat = data['x'].shape[2], sw.shape[0]
    X_feat = np.zeros((n_trials, n_feat))
    for t in range(n_trials):
        var = np.var(np.dot(data['x'][:, :, t], sw.T), axis=0) + 1e-9
        X_feat[t, :] = np.log(var / np.sum(var))

    model = SVC(kernel='linear', probability=True, C=1.0)
    X_mean, X_std = np.mean(X_feat, axis=0), np.std(X_feat, axis=0)
    X_std[X_std < 1e-8] = 1e-8
    model.fit((X_feat - X_mean) / X_std, data['y'])
    model.X_mean_, model.X_std_ = X_mean, X_std
    return model, sw


def evaluate_cspsvm(model, sw, test):
    if model is None or sw is None or test is None or test['x'].shape[2] == 0:
        return 0.0
    n_trials, n_feat = test['x'].shape[2], sw.shape[0]
    X_feat = np.zeros((n_trials, n_feat))
    for t in range(n_trials):
        var = np.var(np.dot(test['x'][:, :, t], sw.T), axis=0) + 1e-9
        X_feat[t, :] = np.log(var / np.sum(var))
    if hasattr(model, 'X_mean_'):
        X_feat = (X_feat - model.X_mean_) / model.X_std_
    return np.mean(model.predict(X_feat) == test['y']) * 100


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data(mat_path, subject_idx_0based, num_train_per_class):
    mat = scipy.io.loadmat(mat_path)
    xsubi = mat.get('xsubi_all', mat.get('txsubi_all'))
    subj = xsubi[0, subject_idx_0based]
    all_x = extract_field(subj, 'x')
    if all_x.ndim == 2:
        all_x = np.expand_dims(all_x, axis=2)
    all_y = extract_field(subj, 'y')
    all_y = all_y.flatten() if all_y is not None else np.concatenate([
        np.ones(all_x.shape[2] // 2), np.ones(all_x.shape[2] - all_x.shape[2] // 2) + 1])

    c1 = np.where(all_y == 1)[0][:num_train_per_class]
    c2 = np.where(all_y == 2)[0][:num_train_per_class]
    x = np.concatenate([all_x[:, :, c1], all_x[:, :, c2]], axis=2)
    y = np.concatenate([np.ones(len(c1)), np.ones(len(c2)) + 1])

    x = elliptical_filter(x)
    x = x[TRAIN_START:TRAIN_END + 1, SELECTED_CHANNELS_0_BASED, :]
    xmin, xmax = np.min(x), np.max(x)
    xrange = xmax - xmin
    x_norm = 2 * (x - xmin) / xrange - 1 if xrange > 1e-8 else np.zeros_like(x)
    return {'x': x_norm, 'y': y}, xmin, xmax


def load_valtest_data(mat_path, subject_idx_0based, train_min, train_max):
    mat = scipy.io.loadmat(mat_path)
    xsubi = mat.get('txsubi_all', mat.get('xsubi_all'))
    if subject_idx_0based >= xsubi.shape[1]:
        return None, None
    subj = xsubi[0, subject_idx_0based]
    all_x = extract_field(subj, 'x')
    if all_x.ndim == 2:
        all_x = np.expand_dims(all_x, axis=2)
    all_y = extract_field(subj, 'y')
    all_y = all_y.flatten() if all_y is not None else np.concatenate([
        np.ones(all_x.shape[2] // 2), np.ones(all_x.shape[2] - all_x.shape[2] // 2) + 1])

    c1 = np.where(all_y == 1)[0]
    c2 = np.where(all_y == 2)[0]
    if len(c1) <= NUM_VALID_PER_CLASS or len(c2) <= NUM_VALID_PER_CLASS:
        return None, None

    val_c1, test_c1 = c1[:NUM_VALID_PER_CLASS], c1[NUM_VALID_PER_CLASS:]
    val_c2, test_c2 = c2[:NUM_VALID_PER_CLASS], c2[NUM_VALID_PER_CLASS:]
    train_range = train_max - train_min

    def process(idx_c1, idx_c2):
        x = np.concatenate([all_x[:, :, idx_c1], all_x[:, :, idx_c2]], axis=2)
        y = np.concatenate([np.ones(len(idx_c1)), np.ones(len(idx_c2)) + 1])
        x = elliptical_filter(x)
        x = x[TEST_START:TEST_END + 1, SELECTED_CHANNELS_0_BASED, :]
        if train_range < 1e-8:
            return {'x': np.zeros_like(x), 'y': y}
        x_norm = np.clip(2 * (x - train_min) / train_range - 1, -1, 1)
        return {'x': x_norm, 'y': y}

    return process(val_c1, val_c2), process(test_c1, test_c2)


def load_synthetic_corrected(results_dir, subject_id):
    patterns = [
        os.path.join(results_dir, f'best_synthetic_batch_S{subject_id}_corrected.npz'),
        os.path.join(results_dir, f'best_synthetic_batch_S{subject_id}.npz'),
    ]
    npz_file = None
    for p in patterns:
        if os.path.exists(p):
            npz_file = p
            break
    if npz_file is None:
        matches = glob.glob(os.path.join(results_dir, '*corrected*.npz'))
        if not matches:
            matches = glob.glob(os.path.join(results_dir, '*synthetic*.npz'))
        if matches:
            npz_file = matches[0]
    if npz_file is None:
        return None

    data = np.load(npz_file)
    if 'class0_data_gan_fmt' in data.files and 'class1_data_gan_fmt' in data.files:
        c0, c1 = data['class0_data_gan_fmt'], data['class1_data_gan_fmt']
        x = np.concatenate([np.transpose(c0, (2, 1, 0)), np.transpose(c1, (2, 1, 0))], axis=2)
        y = np.concatenate([np.ones(c0.shape[0]), np.ones(c1.shape[0]) + 1])
        return {'x': x, 'y': y}
    return None


def augment_data(real_csp, synth_csp, ratio):
    n_real = real_csp['x'].shape[2]
    n_want = int(n_real * ratio)
    n_avail = synth_csp['x'].shape[2]
    if n_want > n_avail:
        reps = (n_want // n_avail) + 1
        sx = np.tile(synth_csp['x'], (1, 1, reps))[:, :, :n_want]
        sy = np.tile(synth_csp['y'], reps)[:n_want]
    else:
        sx = synth_csp['x'][:, :, :n_want]
        sy = synth_csp['y'][:n_want]
    return {
        'x': np.concatenate([real_csp['x'], sx], axis=2),
        'y': np.concatenate([real_csp['y'], sy])
    }


# ============================================================
# EVALUATION — PROPER VALIDATION-BASED SELECTION
# ============================================================

def evaluate_subject(train_mat, test_mat, results_dir, subject_id, num_train_per_class):
    np.random.seed(SEED)
    idx = subject_id - 1

    # Load data
    train_csp, tmin, tmax = load_training_data(train_mat, idx, num_train_per_class)
    valid_csp, test_csp = load_valtest_data(test_mat, idx, tmin, tmax)
    if valid_csp is None or test_csp is None:
        print(f"  S{subject_id}: SKIP — no val/test data")
        return None

    train_valid = {
        'x': np.concatenate([train_csp['x'], valid_csp['x']], axis=2),
        'y': np.concatenate([train_csp['y'], valid_csp['y']])
    }

    # Baseline: train+valid → test
    model_bl, sf_bl = train_cspsvm(train_valid)
    acc_bl = evaluate_cspsvm(model_bl, sf_bl, test_csp)

    result = {
        'subject': subject_id,
        'n_train': train_csp['x'].shape[2],
        'n_test': test_csp['x'].shape[2],
        'baseline': acc_bl,
    }

    # Load synthetic
    synth = load_synthetic_corrected(results_dir, subject_id)
    if synth is None:
        print(f"  S{subject_id}: Baseline={acc_bl:.2f}% — no synthetic data")
        return result

    # Synth only (train on synth → test)
    model_s, sf_s = train_cspsvm(synth)
    acc_s = evaluate_cspsvm(model_s, sf_s, test_csp)
    result['synth_only'] = acc_s

    # ============================================================
    # STEP 1: Select best ratio on VALIDATION
    # Train on TRAIN + synth → evaluate on VALIDATION
    # ============================================================
    best_val_ratio, best_val_acc = None, 0
    val_results = {}

    for ratio in MIX_RATIOS:
        aug_train = augment_data(train_csp, synth, ratio)  # train ONLY + synth
        model_v, sf_v = train_cspsvm(aug_train)
        acc_v = evaluate_cspsvm(model_v, sf_v, valid_csp)
        val_results[ratio] = acc_v
        if acc_v > best_val_acc:
            best_val_acc = acc_v
            best_val_ratio = ratio

    # Also check baseline on validation (no augmentation)
    model_bl_val, sf_bl_val = train_cspsvm(train_csp)
    acc_bl_val = evaluate_cspsvm(model_bl_val, sf_bl_val, valid_csp)

    result['selected_ratio'] = best_val_ratio
    result['selected_val_acc'] = best_val_acc
    result['baseline_val_acc'] = acc_bl_val

    # ============================================================
    # STEP 2: Retrain with selected ratio on TRAIN+VALID → TEST
    # This is the single, unbiased test evaluation
    # ============================================================
    aug_final = augment_data(train_valid, synth, best_val_ratio)
    model_final, sf_final = train_cspsvm(aug_final)
    acc_final = evaluate_cspsvm(model_final, sf_final, test_csp)

    result['best_aug'] = acc_final
    result['improvement'] = acc_final - acc_bl

    # ============================================================
    # OPTIONAL: Report all ratios on test for transparency
    # These are NOT used for selection — just for the full picture
    # ============================================================
    for ratio in MIX_RATIOS:
        aug_all = augment_data(train_valid, synth, ratio)
        model_t, sf_t = train_cspsvm(aug_all)
        acc_t = evaluate_cspsvm(model_t, sf_t, test_csp)
        result[f'test_r{ratio}'] = acc_t

    print(f"  S{subject_id}: Base={acc_bl:.2f}%  Synth={acc_s:.2f}%  "
          f"Selected r={best_val_ratio} (val={best_val_acc:.2f}%) → Test={acc_final:.2f}%  "
          f"Δ={acc_final - acc_bl:+.2f}%")

    return result


def print_summary(results, num_trials):
    print(f"\n\n{'=' * 105}")
    print(f"  CSP-SVM CLASSIFICATION — {num_trials} trials/class — Ratio selected on VALIDATION")
    print(f"{'=' * 105}")

    # Header
    print(f"\n  {'Subj':>4s}  {'Base':>7s}  {'Synth':>7s}  ", end='')
    for r in MIX_RATIOS:
        print(f"{'r=' + str(r):>7s}  ", end='')
    print(f"{'SelRat':>6s}  {'ValAcc':>7s}  {'TestAcc':>7s}  {'Δ':>6s}")

    sep = f"  {'----':>4s}  {'-------':>7s}  {'-------':>7s}  "
    for _ in MIX_RATIOS:
        sep += f"{'-------':>7s}  "
    sep += f"{'------':>6s}  {'-------':>7s}  {'-------':>7s}  {'------':>6s}"
    print(sep)

    baselines, bests, improvements = [], [], []

    for r in results:
        bl = r['baseline']
        so = r.get('synth_only', None)
        ba = r.get('best_aug', None)
        sr = r.get('selected_ratio', None)
        va = r.get('selected_val_acc', None)
        imp = r.get('improvement', None)

        baselines.append(bl)
        if ba is not None: bests.append(ba)
        if imp is not None: improvements.append(imp)

        so_s = f"{so:.2f}" if so is not None else "N/A"
        ba_s = f"{ba:.2f}" if ba is not None else "N/A"
        sr_s = f"{sr}" if sr is not None else ""
        va_s = f"{va:.2f}" if va is not None else ""
        imp_s = f"{imp:+.2f}" if imp is not None else "N/A"

        print(f"  S{r['subject']:>3d}  {bl:>7.2f}  {so_s:>7s}  ", end='')
        for ratio in MIX_RATIOS:
            val = r.get(f'test_r{ratio}', None)
            # Mark the selected ratio with *
            marker = "*" if ratio == sr else " "
            print(f"{val:>6.2f}{marker} " if val is not None else f"{'N/A':>7s}  ", end='')
        print(f"{sr_s:>6s}  {va_s:>7s}  {ba_s:>7s}  {imp_s:>6s}")

    print(sep)
    mean_bl = np.mean(baselines)
    mean_best = np.mean(bests) if bests else 0
    mean_imp = np.mean(improvements) if improvements else 0
    std_bl = np.std(baselines)
    std_best = np.std(bests) if bests else 0

    print(f"\n  Mean baseline: {mean_bl:.2f} ± {std_bl:.2f}%")
    print(f"  Mean best augmented (val-selected): {mean_best:.2f} ± {std_best:.2f}%")
    print(f"  Mean improvement: {mean_imp:+.2f}%")
    print(f"  Subjects improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
    print(f"\n  * = ratio selected via validation (not cherry-picked from test)")


def save_csv(results, path):
    if not results:
        return
    fixed = ['subject', 'n_train', 'n_test', 'baseline', 'synth_only',
             'baseline_val_acc', 'selected_ratio', 'selected_val_acc', 'best_aug', 'improvement']
    test_cols = [f'test_r{r}' for r in MIX_RATIOS]
    cols = fixed + test_cols

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n  CSV saved: {path}")


# ============================================================
# CLI
# ============================================================

def infer_subject_id(path):
    parts = os.path.basename(os.path.normpath(path)).split('_')
    for i, p in enumerate(parts):
        if p.lower() == 'subject' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='CSP-SVM with validation-based ratio selection')
    parser.add_argument('--train_mat', required=True)
    parser.add_argument('--test_mat', required=True)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--subject_id', type=int, default=None)
    parser.add_argument('--pattern', default=None)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--output_csv', default=None)
    args = parser.parse_args()

    num_train_per_class = args.trials - NUM_VALID_PER_CLASS
    if num_train_per_class <= 0:
        num_train_per_class = max(args.trials - 5, args.trials // 2)

    csv_path = args.output_csv or f'csp_results_{args.trials}trials.csv'

    print(f"Config: {args.trials} trials/class → {num_train_per_class} train + {NUM_VALID_PER_CLASS} valid")
    print(f"Ratio selection: VALIDATION-based (not test)")

    for f in [args.train_mat, args.test_mat]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found"); return

    all_results = []

    if args.pattern:
        for d in sorted(glob.glob(args.pattern)):
            if not os.path.isdir(d):
                continue
            sid = infer_subject_id(d)
            if not sid:
                continue
            try:
                r = evaluate_subject(args.train_mat, args.test_mat, d, sid, num_train_per_class)
                if r:
                    all_results.append(r)
            except Exception as e:
                print(f"  S{sid}: ERROR — {e}")
                traceback.print_exc()
    else:
        if not args.results_dir:
            print("Need --results_dir or --pattern"); return
        sid = args.subject_id or infer_subject_id(args.results_dir)
        if not sid:
            print("Need --subject_id"); return
        r = evaluate_subject(args.train_mat, args.test_mat, args.results_dir, sid, num_train_per_class)
        if r:
            all_results.append(r)

    if all_results:
        print_summary(all_results, args.trials)
        save_csv(all_results, csv_path)


if __name__ == '__main__':
    main()