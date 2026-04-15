#!/usr/bin/env python3
"""
Multi-Dataset MI EEG cWGAN-GP Data Augmentation Framework — v3
==============================================================
Supports:
  - BCI Competition IV Dataset 2a (data1.mat / data2.mat)
  - Cho2017 Dataset (DATA1.mat - DATA5.mat)
Classifiers:
  - CSP + SVM (traditional, optimised for small data)
"""

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import ellip, ellipord, filtfilt, welch, savgol_filter, butter
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import traceback
from sklearn.svm import SVC
from scipy.linalg import eigh
import scipy.stats as stats
import os, csv
import glob
import time

# ── CONSTANTS (BCI-IV-2a) ────────────────────────────────────────────────────
SELECTED_CHANNELS_1_BASED_BCI4 = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17]
SELECTED_CHANNELS_0_BASED_BCI4 = [idx - 1 for idx in SELECTED_CHANNELS_1_BASED_BCI4]
NUM_SELECTED_CHANNELS_BCI4 = len(SELECTED_CHANNELS_0_BASED_BCI4)

# ── CONSTANTS (Cho2017) ──────────────────────────────────────────────────────
SELECTED_CHANNELS_1_BASED_CHO = [14, 13, 12, 48, 49, 50, 51, 17, 18, 19, 56, 54, 55]
SELECTED_CHANNELS_0_BASED_CHO = [idx - 1 for idx in SELECTED_CHANNELS_1_BASED_CHO]
NUM_SELECTED_CHANNELS_CHO = len(SELECTED_CHANNELS_0_BASED_CHO)
CHO_LOWCUT = 8
CHO_HIGHCUT = 30
CHO_FS = 512
CHO_TIME_TRIM_S = 0.5

# ── COMMON CONSTANTS ─────────────────────────────────────────────────────────
NUM_TRAIN_TRIALS_PER_CLASS = 40
NUM_VALID_TRIALS_PER_CLASS = 10
NUM_RUNS_PER_SUBJECT = 3
NUM_CLASSES_CGAN = 2
LATENT_DIM_CGAN = 50
EMBEDDING_DIM_CGAN = 25
SEED_VALUE = 45
GAN_EPOCHS = 2000
NUM_BATCHES_PER_RUN = 30
NUM_SYNTH_PER_CLASS = NUM_TRAIN_TRIALS_PER_CLASS * 4  # 160 per class, 4x real
MIX_RATIOS = [0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]

np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# ── LOGGING ───────────────────────────────────────────────────────────────────
current_log_file_path = None

def log_message(message):
    print(message)
    global current_log_file_path
    if current_log_file_path:
        with open(current_log_file_path, 'a') as f:
            f.write(str(message) + "\n")

def set_log_path(path):
    global current_log_file_path
    current_log_file_path = path


# =============================================================================
# 1.  PREPROCESSING & NORMALIZATION
# =============================================================================

def normalize_per_channel(x, clip_sigma=3.0, norm_stats=None):
    """Per-channel z-score -> clip -> scale to [-1,1]."""
    if norm_stats is None:
        ch_mean = np.mean(x, axis=(0, 2), keepdims=True)
        ch_std  = np.std(x,  axis=(0, 2), keepdims=True) + 1e-8
    else:
        ch_mean, ch_std = norm_stats
    z = (x - ch_mean) / ch_std
    return np.clip(z / clip_sigma, -1., 1.).astype(np.float32), (ch_mean, ch_std)

def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    nyq = 0.5 * fs
    low_stop = max(0.1, lowcut - 1.0)
    high_stop = min(nyq - 0.1, highcut + 1.0)
    wp = np.clip([lowcut / nyq, highcut / nyq], 1e-6, 1.0 - 1e-6).tolist()
    ws = np.clip([low_stop / nyq, high_stop / nyq], 1e-6, 1.0 - 1e-6).tolist()
    n, wn = ellipord(wp, ws, rp, rs)
    b, a = ellip(n, rp, rs, wn, btype='band')
    return filtfilt(b, a, data, axis=0)

def butterworth_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def _get_eeg_data_from_field(subject_struct_item, field_name, subject_id_str):
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


# ── BCI-IV-2a preprocessing ──────────────────────────────────────────────────

def preprocess_training_data(raw_subject_struct, subject_id_str,
                             num_train_per_class=NUM_TRAIN_TRIALS_PER_CLASS,
                             selected_channels=SELECTED_CHANNELS_0_BASED_BCI4):
    try:
        all_x = _get_eeg_data_from_field(raw_subject_struct, 'x', subject_id_str)
        all_y = _get_eeg_data_from_field(raw_subject_struct, 'y', subject_id_str)
        if all_x is None or all_y is None: return (None,) * 5

        c1_idx, c2_idx = np.where(all_y == 1)[0], np.where(all_y == 2)[0]
        if len(c1_idx) < num_train_per_class or len(c2_idx) < num_train_per_class: return (None,) * 5

        x_train = np.concatenate((all_x[:, :, c1_idx[:num_train_per_class]], all_x[:, :, c2_idx[:num_train_per_class]]), axis=2)
        y_train = np.concatenate((np.ones(num_train_per_class), np.ones(num_train_per_class) + 1))

        filtered = elliptical_filter(x_train)[125:625, :, :]
        selected = filtered[:, selected_channels, :]
        normalized, (ch_mean, ch_std) = normalize_per_channel(selected)

        x_gan = np.transpose(normalized, (2, 1, 0))  # (trials, ch, T)
        csp_dict = {'x': normalized, 'y': y_train}    # (T, ch, trials)
        return x_gan, y_train, csp_dict, ch_mean, ch_std
    except Exception as e: return (None,) * 5

def preprocess_valtest_data(raw_subject_struct, subject_id_str, num_valid_per_class, 
                            train_mean, train_std, selected_channels=SELECTED_CHANNELS_0_BASED_BCI4):
    try:
        all_x = _get_eeg_data_from_field(raw_subject_struct, 'x', subject_id_str)
        all_y = _get_eeg_data_from_field(raw_subject_struct, 'y', subject_id_str)
        if all_x is None or all_y is None: return None, None

        c1_idx, c2_idx = np.where(all_y == 1)[0], np.where(all_y == 2)[0]
        if len(c1_idx) <= num_valid_per_class or len(c2_idx) <= num_valid_per_class: return None, None

        def process_subset(c1, c2):
            x_data = np.concatenate((all_x[:, :, c1], all_x[:, :, c2]), axis=2)
            y_labels = np.concatenate((np.ones(len(c1)), np.ones(len(c2)) + 1))
            filtered = elliptical_filter(x_data)[115:615, :, :]
            normalized, _ = normalize_per_channel(filtered[:, selected_channels, :], norm_stats=(train_mean, train_std))
            return {'x': normalized, 'y': y_labels}

        return process_subset(c1_idx[:num_valid_per_class], c2_idx[:num_valid_per_class]), \
               process_subset(c1_idx[num_valid_per_class:], c2_idx[num_valid_per_class:])
    except Exception: return None, None


# ── Cho2017 preprocessing ────────────────────────────────────────────────────

def preprocess_cho2017_subject(raw_subject_struct, subject_id_str, dataset_name,
                               selected_channels=SELECTED_CHANNELS_0_BASED_CHO,
                               num_train=NUM_TRAIN_TRIALS_PER_CLASS,
                               num_valid=NUM_VALID_TRIALS_PER_CLASS):
    """Load, preprocess, and split Cho2017 data into train/val/test.
    Returns: x_gan, y_train, train_csp, val_csp, test_csp, ch_mean, ch_std
    or (None,)*7 on failure."""
    try:
        all_x = _get_eeg_data_from_field(raw_subject_struct, 'x', subject_id_str)
        all_y = _get_eeg_data_from_field(raw_subject_struct, 'y', subject_id_str)
        if all_x is None or all_y is None: return (None,) * 7

        # Butterworth 8-30 Hz
        filtered = butterworth_filter(all_x, CHO_LOWCUT, CHO_HIGHCUT, CHO_FS)

        # Time trim
        start_s = int(CHO_FS * CHO_TIME_TRIM_S)
        end_s = -start_s if start_s > 0 else filtered.shape[0]
        processed = filtered[start_s:end_s, :, :]

        # Channel selection
        processed = processed[:, selected_channels, :]

        # Sequential split
        idx_c1 = np.where(all_y == 1)[0]
        idx_c2 = np.where(all_y == 2)[0]
        total_needed = num_train + num_valid
        if len(idx_c1) < total_needed or len(idx_c2) < total_needed:
            log_message(f"  ERROR S{subject_id_str}: Not enough trials for split in {dataset_name}")
            return (None,) * 7

        train_idx = np.concatenate([idx_c1[:num_train], idx_c2[:num_train]])
        valid_idx = np.concatenate([idx_c1[num_train:total_needed], idx_c2[num_train:total_needed]])
        test_idx  = np.concatenate([idx_c1[total_needed:], idx_c2[total_needed:]])

        train_x = processed[:, :, train_idx]
        valid_x = processed[:, :, valid_idx]
        test_x  = processed[:, :, test_idx]
        train_y = all_y[train_idx]
        valid_y = all_y[valid_idx]
        test_y  = all_y[test_idx]

        # Per-channel normalisation from training data
        normalized_train, (ch_mean, ch_std) = normalize_per_channel(train_x)
        normalized_valid, _ = normalize_per_channel(valid_x, norm_stats=(ch_mean, ch_std))
        normalized_test, _  = normalize_per_channel(test_x, norm_stats=(ch_mean, ch_std))

        x_gan = np.transpose(normalized_train, (2, 1, 0))  # (trials, ch, T)
        train_csp = {'x': normalized_train, 'y': train_y}
        val_csp   = {'x': normalized_valid, 'y': valid_y}
        test_csp  = {'x': normalized_test, 'y': test_y}

        log_message(f"  S{subject_id_str} ({dataset_name}): Train={x_gan.shape}, Val={val_csp['x'].shape}, Test={test_csp['x'].shape}")
        return x_gan, train_y, train_csp, val_csp, test_csp, ch_mean, ch_std
    except Exception as e:
        log_message(f"  ERROR S{subject_id_str} preprocess: {traceback.format_exc()}")
        return (None,) * 7


# =============================================================================
# 2.  FEATURE EXTRACTION & CLASSIFICATION
# =============================================================================

# ── CSP + SVM ────────────────────────────────────────────────────────────────

def csp(data):
    X1, X2 = data['x'][:, :, data['y'] == 1], data['x'][:, :, data['y'] == 2]
    n_channels = X1.shape[1]
    cov1 = np.mean([np.cov(X1[:,:,i], rowvar=False) for i in range(X1.shape[2]) if not np.all(np.isnan(X1[:,:,i]))], axis=0)
    cov2 = np.mean([np.cov(X2[:,:,i], rowvar=False) for i in range(X2.shape[2]) if not np.all(np.isnan(X2[:,:,i]))], axis=0)
    cov1_reg, cov2_reg = cov1 + 1e-9 * np.eye(n_channels), cov2 + 1e-9 * np.eye(n_channels)
    evals, evecs = eigh(cov1_reg, cov1_reg + cov2_reg)
    return evecs[:, np.argsort(evals)[::-1]].T

def compute_csp_filters(data, n_channels):
    W = csp(data)
    if n_channels > 6: return np.vstack((W[0:3, :], W[-3:, :])) if W.shape[0] >= 6 else W
    return np.vstack((W[0, :][np.newaxis,:], W[-1, :][np.newaxis,:])) if W.shape[0] >= 2 else W

def extract_csp_features(data, selectedw):
    if selectedw is None or selectedw.size == 0: return {'x': np.zeros((0, data['x'].shape[2])), 'y': data['y']}
    features = np.zeros((selectedw.shape[0], data['x'].shape[2]))
    for trial in range(data['x'].shape[2]):
        var = np.var(np.dot(data['x'][:, :, trial], selectedw.T), axis=0) + 1e-9
        features[:, trial] = np.log(var / np.sum(var))
    return {'x': features, 'y': data['y']}

def train_svm_on_features(features):
    X_train, y_train = features['x'].T, features['y']
    if X_train.shape[0] == 0 or len(np.unique(y_train)) < 2:
        log_message("    [Warning] SVM skipped: Not enough classes/trials after filtering.")
        return None
    model = SVC(kernel='linear', probability=True, C=1.0)
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-8
    model.X_mean_, model.X_std_ = X_mean, X_std
    model.fit((X_train - X_mean) / X_std, y_train)
    return model

def evaluate_svm_on_features(model, features):
    if model is None or features['x'].shape[1] == 0: return 0, np.zeros((2,2)), np.array([])
    X_test, y_true = features['x'].T, features['y']
    y_pred = model.predict((X_test - model.X_mean_) / model.X_std_)
    return np.mean(y_pred == y_true) * 100, confusion_matrix(y_true, y_pred, labels=[1, 2]), y_pred


# =============================================================================
# 3.  GAN ARCHITECTURE
# =============================================================================

def build_generator(num_channels, time_points, latent_dim, num_classes, embedding_dim):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,))
    merged = layers.Concatenate()([noise_input, layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))])

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

    if x.shape[1] > time_points: x = layers.Cropping1D(((x.shape[1]-time_points)//2, (x.shape[1]-time_points)-((x.shape[1]-time_points)//2)))(x)
    elif x.shape[1] < time_points: x = layers.ZeroPadding1D(((time_points-x.shape[1])//2, (time_points-x.shape[1])-((time_points-x.shape[1])//2)))(x)

    return models.Model([noise_input, label_input], layers.Permute((2, 1))(x))

def build_critic(num_channels, time_points, num_classes, embedding_dim):
    data_input = layers.Input(shape=(num_channels, time_points))
    label_input = layers.Input(shape=(1,))
    x = layers.Permute((2, 1))(data_input)

    for filters in [64, 128, 256]:
        x = layers.Conv1D(filters, 5, strides=2, padding="same")(x)
        if filters > 64: x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    merged = layers.Concatenate()([layers.Flatten()(x), layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))])
    return models.Model([data_input, label_input], layers.Dense(1)(merged))


# =============================================================================
# 4.  LOSS FUNCTIONS & TARGETED FILTERING
# =============================================================================

def frequency_domain_loss(real_data, generated_data):
    real_fft = tf.abs(tf.signal.rfft(tf.transpose(real_data, perm=[0, 2, 1])))
    gen_fft = tf.abs(tf.signal.rfft(tf.transpose(generated_data, perm=[0, 2, 1])))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))

def csp_feature_loss(real_data, fake_data, csp_filters, labels=None):
    """MI-specific: class-conditional CSP feature matching."""
    real_proj = tf.einsum('fc,bct->bft', csp_filters, real_data)
    fake_proj = tf.einsum('fc,bct->bft', csp_filters, fake_data)
    real_var = tf.math.reduce_variance(real_proj, axis=2) + 1e-9
    fake_var = tf.math.reduce_variance(fake_proj, axis=2) + 1e-9
    real_log = tf.math.log(real_var / tf.reduce_sum(real_var, axis=1, keepdims=True))
    fake_log = tf.math.log(fake_var / tf.reduce_sum(fake_var, axis=1, keepdims=True))
    sample_loss = tf.reduce_mean(tf.square(real_log - fake_log))
    
    if labels is None:
        return sample_loss
    
    lab_flat = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    m0, m1 = tf.equal(lab_flat, 0), tf.equal(lab_flat, 1)
    n0 = tf.maximum(tf.reduce_sum(tf.cast(m0, tf.float32)), 1.0)
    n1 = tf.maximum(tf.reduce_sum(tf.cast(m1, tf.float32)), 1.0)
    
    real_mean_c0 = tf.reduce_sum(tf.boolean_mask(real_log, m0), axis=0) / n0
    real_mean_c1 = tf.reduce_sum(tf.boolean_mask(real_log, m1), axis=0) / n1
    fake_mean_c0 = tf.reduce_sum(tf.boolean_mask(fake_log, m0), axis=0) / n0
    fake_mean_c1 = tf.reduce_sum(tf.boolean_mask(fake_log, m1), axis=0) / n1
    
    class_cond_loss = (tf.reduce_mean(tf.square(real_mean_c0 - fake_mean_c0)) +
                       tf.reduce_mean(tf.square(real_mean_c1 - fake_mean_c1)))
    
    real_chpow = tf.math.reduce_variance(real_data, axis=2) + 1e-9
    fake_chpow = tf.math.reduce_variance(fake_data, axis=2) + 1e-9
    
    real_chpow_c0 = tf.reduce_sum(tf.boolean_mask(real_chpow, m0), axis=0) / n0
    real_chpow_c1 = tf.reduce_sum(tf.boolean_mask(real_chpow, m1), axis=0) / n1
    fake_chpow_c0 = tf.reduce_sum(tf.boolean_mask(fake_chpow, m0), axis=0) / n0
    fake_chpow_c1 = tf.reduce_sum(tf.boolean_mask(fake_chpow, m1), axis=0) / n1
    
    real_contrast = (real_chpow_c0 - real_chpow_c1) / (real_chpow_c0 + real_chpow_c1)
    fake_contrast = (fake_chpow_c0 - fake_chpow_c1) / (fake_chpow_c0 + fake_chpow_c1)
    contrast_loss = tf.reduce_mean(tf.square(real_contrast - fake_contrast))
    
    return sample_loss + 5.0 * class_cond_loss + 3.0 * contrast_loss

def gradient_penalty(critic, real_samples, fake_samples, real_labels):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    gradients = gp_tape.gradient(pred, [interpolated])[0]
    return 10.0 * tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10) - 1.0) ** 2)

def filter_synthetic_trials(real_feats, synth_feats, synth_y, keep_ratio=0.6, min_trials_per_class=10):
    """Top-K rejection sampling: keep trials closest to real feature distribution."""
    keep_mask = np.zeros(synth_y.shape[0], dtype=bool)
    for cls in np.unique(real_feats['y']):
        r_idx, s_idx = real_feats['y'] == cls, synth_feats['y'] == cls
        if np.sum(s_idx) == 0: continue
        r_mean = np.mean(real_feats['x'][:, r_idx], axis=1)
        r_std  = np.std(real_feats['x'][:, r_idx], axis=1) + 1e-8
        trial_scores = np.mean(np.abs(synth_feats['x'][:, s_idx] - r_mean[:, None]) / r_std[:, None], axis=0)
        num_to_keep = min(max(int(np.sum(s_idx) * keep_ratio), min_trials_per_class), np.sum(s_idx))
        keep_mask[np.where(s_idx)[0][np.argsort(trial_scores)[:num_to_keep]]] = True
    return {'x': synth_feats['x'][:, keep_mask], 'y': synth_feats['y'][keep_mask]}, keep_mask


def balanced_subsample(data, max_total):
    """Subsample data dict to max_total trials, balanced across classes.
    Works for both CSP format {'x': (feat, trials)} and raw format {'x': (T, ch, trials)}.
    Returns new dict with at most max_total trials, equal per class."""
    y = data['y']
    classes = np.unique(y)
    n_per_class = max_total // len(classes)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        idx.extend(c_idx[:n_per_class])
    idx = np.array(sorted(idx))
    if data['x'].ndim == 2:
        # CSP feature format: (n_feat, trials)
        return {'x': data['x'][:, idx], 'y': y[idx]}
    else:
        # Raw format: (T, ch, trials)
        return {'x': data['x'][:, :, idx], 'y': y[idx]}


# =============================================================================
# 5.  TRAINING LOOPS
# =============================================================================

def train_cgan(eeg_data, eeg_labels, epochs, batch_size, num_gan_channels,
               output_dir, subject_id, run_number, time_points,
               latent_dim=100, n_critic_steps=5, freq_loss_weight=1.5,
               csp_filters=None, feat_loss_weight=5.0):

    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)[:, np.newaxis]
    gen = build_generator(num_gan_channels, time_points, latent_dim, 2, 25)
    crit = build_critic(num_gan_channels, time_points, 2, 25)
    opt_g, opt_c = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9), tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
    freq_w_tf = tf.constant(freq_loss_weight, dtype=tf.float32)
    feat_w_tf = tf.constant(feat_loss_weight, dtype=tf.float32)
    csp_tf = tf.cast(tf.convert_to_tensor(csp_filters), tf.float32) if csp_filters is not None else None

    @tf.function
    def train_step(rx, ry):
        bs = tf.shape(rx)[0]
        for _ in range(n_critic_steps):
            nz = tf.random.normal([bs, latent_dim])
            with tf.GradientTape() as tape_c:
                fk = gen([nz, ry], training=True)
                d_loss = tf.reduce_mean(crit([fk, ry], training=True)) - tf.reduce_mean(crit([rx, ry], training=True)) + gradient_penalty(crit, rx, fk, ry)
            opt_c.apply_gradients(zip(tape_c.gradient(d_loss, crit.trainable_variables), crit.trainable_variables))

        nz = tf.random.normal([bs, latent_dim])
        with tf.GradientTape() as tape_g:
            fk_g = gen([nz, ry], training=True)
            g_w = -tf.reduce_mean(crit([fk_g, ry], training=True))
            g_frq = frequency_domain_loss(rx, fk_g)
            g_feat = csp_feature_loss(rx, fk_g, csp_tf, labels=ry) if csp_tf is not None else tf.constant(0.0)
            g_loss = g_w + (freq_w_tf * g_frq) + (feat_w_tf * g_feat)
        opt_g.apply_gradients(zip(tape_g.gradient(g_loss, gen.trainable_variables), gen.trainable_variables))
        return d_loss, g_w, g_frq, g_feat

    best_g_loss, best_weights, patience_cnt = float('inf'), None, 0
    log_message(f"  cWGAN-GP | MI | S{subject_id} R{run_number}")
    log_message(f"    Data: {eeg_data.shape[0]} trials, {num_gan_channels} ch, {time_points} T | batch={batch_size} epochs={epochs}")

    loss_csv_path = os.path.join(output_dir, f'training_loss_R{run_number}.csv')
    with open(loss_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'critic_loss', 'gen_wasserstein', 'gen_freq_loss', 'gen_feat_loss', 'gen_total', 'wall_time_s'])

    t_start = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(eeg_data.shape[0])
        ep_d, ep_gw, ep_gf, ep_gft = 0.0, 0.0, 0.0, 0.0
        n_batches = max(1, eeg_data.shape[0] // batch_size)
        for bi in range(n_batches):
            s, e = bi * batch_size, (bi + 1) * batch_size
            rx, ry = tf.convert_to_tensor(eeg_data[perm[s:e]], tf.float32), tf.convert_to_tensor(labels_0_1[perm[s:e]], tf.int32)
            d, gw, gfrq, gft = train_step(rx, ry)
            ep_d += d.numpy()
            ep_gw += (gw + freq_loss_weight * gfrq + feat_loss_weight * gft).numpy()
            ep_gf += gfrq.numpy(); ep_gft += gft.numpy()

        ep_d /= n_batches
        g_total = ep_gw / n_batches
        ep_gf /= n_batches; ep_gft /= n_batches
        wall_t = time.time() - t_start
        ep_gw_only = g_total - freq_loss_weight * ep_gf - feat_loss_weight * ep_gft

        with open(loss_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep+1, f'{ep_d:.4f}', f'{ep_gw_only:.4f}', f'{ep_gf:.4f}', f'{ep_gft:.4f}', f'{g_total:.4f}', f'{wall_t:.1f}'])

        if (ep + 1) % 100 == 0 or ep == 0:
            log_message(f"    Ep {ep+1:4d}/{epochs} | D={ep_d:.3f} Gw={ep_gw_only:.3f} Gfrq={ep_gf:.4f} Gfeat={ep_gft:.4f} | {wall_t:.0f}s")

        if ep >= epochs // 4:
            if g_total < best_g_loss - 0.1: best_g_loss, best_weights, patience_cnt = g_total, gen.get_weights(), 0
            else: patience_cnt += 1
            if patience_cnt >= 500:
                log_message(f"    Early stop at ep {ep+1} (patience=500)")
                break

    if best_weights: gen.set_weights(best_weights)
    log_message(f"    Training complete. Best G loss: {best_g_loss:.4f} | Total time: {time.time()-t_start:.0f}s")
    return gen

def generate_synthetic(gen, num_samples, target_lbl, latent_dim=100):
    if gen is None or num_samples == 0: return np.array([])
    synth = gen([tf.random.normal([num_samples, latent_dim]), tf.ones((num_samples, 1), tf.int32) * target_lbl], training=False).numpy()
    for i in range(synth.shape[0]):
        for j in range(synth.shape[1]):
            synth[i, j, :] = savgol_filter(synth[i, j, :], 7, 2)
    return synth


# =============================================================================
# 6.  METRICS & PLOTTING
# =============================================================================

def compute_psd_correlation(real_data, synth_data, fs=250):
    if real_data.size == 0 or synth_data.size == 0: return 0.0
    return float(np.mean([max(stats.pearsonr(np.mean([welch(real_data[t, ch, :], fs=fs, nperseg=128)[1] for t in range(real_data.shape[0])], axis=0),
                                             np.mean([welch(synth_data[t, ch, :], fs=fs, nperseg=128)[1] for t in range(synth_data.shape[0])], axis=0))[0], 0.0)
                          for ch in range(real_data.shape[1])]))

def compute_amplitude_similarity(real_data, synth_data):
    if real_data.size == 0 or synth_data.size == 0: return 0.0
    return float(1.0 / (1.0 + np.mean([stats.wasserstein_distance(real_data[:, ch, :].ravel(), synth_data[:, ch, :].ravel()) for ch in range(real_data.shape[1])])))

def compute_band_power_correlation(real_data, synth_data, fs=250):
    if real_data.size == 0 or synth_data.size == 0: return 0.0
    r_pwrs, s_pwrs = [], []
    for blo, bhi in [(8, 13), (13, 30)]:
        for ch in range(real_data.shape[1]):
            f, r_psd = welch(real_data[:, ch, :].mean(axis=0), fs=fs, nperseg=128)
            _, s_psd = welch(synth_data[:, ch, :].mean(axis=0), fs=fs, nperseg=128)
            r_pwrs.append(np.mean(r_psd[(f >= blo) & (f <= bhi)]))
            s_pwrs.append(np.mean(s_psd[(f >= blo) & (f <= bhi)]))
    return float(max(stats.pearsonr(r_pwrs, s_pwrs)[0], 0.0)) if np.std(r_pwrs)>1e-10 else 0.0

def compute_fisher_separability(synth_feats):
    X, y = synth_feats['x'], synth_feats['y']
    classes = np.unique(y)
    if len(classes) < 2: return 0.0
    c1_mask, c2_mask = y == classes[0], y == classes[1]
    if np.sum(c1_mask) < 2 or np.sum(c2_mask) < 2: return 0.0
    m1, m2 = np.mean(X[:, c1_mask], axis=1), np.mean(X[:, c2_mask], axis=1)
    v1, v2 = np.var(X[:, c1_mask], axis=1), np.var(X[:, c2_mask], axis=1)
    denom = v1 + v2 + 1e-10
    fisher = np.mean((m1 - m2) ** 2 / denom)
    return float(fisher / (1.0 + fisher))

def plot_acc(bl_csp, so_csp, best_csp, sid, odir):
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(['Baseline', 'Synth-Only', 'Best Aug'], [bl_csp, so_csp, best_csp], color=['#2166AC', '#B2182B', '#4DAF4A'])
    ax.set(ylabel='Accuracy (%)', title=f'CSP+SVM - S{sid}', ylim=(0, 110))
    for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height()+1, f'{b.get_height():.2f}%', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f'accuracy_S{sid}.png'), dpi=150)
    plt.close(fig)

def plot_psd_comparison(x_real_gan, x_synth_gan, fs, subject_id, output_dir, paradigm='MI'):
    fig, ax = plt.subplots(figsize=(10, 5))
    nperseg = min(256, x_real_gan.shape[2])
    for lbl, xd, color in [('Real', x_real_gan, '#2166AC'), ('Synthetic', x_synth_gan, '#B2182B')]:
        all_psd = []
        for t in range(xd.shape[0]):
            ch_psd = [welch(xd[t, c, :], fs=fs, nperseg=nperseg)[1] for c in range(xd.shape[1])]
            all_psd.append(np.mean(ch_psd, axis=0))
        f_axis = welch(xd[0, 0, :], fs=fs, nperseg=nperseg)[0]
        psd_mean = 10 * np.log10(np.mean(all_psd, axis=0) + 1e-15)
        mask = f_axis <= 45
        ax.plot(f_axis[mask], psd_mean[mask], color=color, lw=2, label=f'{lbl} ({xd.shape[0]} trials)')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (dB)')
    ax.set_title(f'PSD Comparison - {paradigm} S{subject_id}')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'psd_S{subject_id}.png'), dpi=150)
    plt.close(fig)

def plot_training_loss(output_dir, run_number, subject_id):
    csv_path = os.path.join(output_dir, f'training_loss_R{run_number}.csv')
    if not os.path.exists(csv_path): return
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=float)
    if data.ndim < 2 or data.shape[0] < 2: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(data[:, 0], data[:, 1], 'b-', alpha=0.7, lw=0.8, label='Critic')
    ax1.set(xlabel='Epoch', ylabel='Loss', title=f'Critic Loss - S{subject_id} R{run_number}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(data[:, 0], data[:, 2], 'r-', alpha=0.7, lw=0.8, label='G Wasserstein')
    ax2.plot(data[:, 0], data[:, 3], 'g-', alpha=0.7, lw=0.8, label='G Freq')
    ax2.plot(data[:, 0], data[:, 4], 'm-', alpha=0.7, lw=0.8, label='G Feature')
    ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Generator Losses - S{subject_id} R{run_number}')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'training_loss_R{run_number}.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# 7.  UNIFIED MI ORCHESTRATION (works for both datasets)
# =============================================================================

def process_mi_subject(x_gan, y_labels, train_csp, val_csp, test_csp,
                       ch_mean, ch_std, num_channels, sid, output_dir,
                       fs=250, dataset_tag='BCI4'):
    """Unified MI subject processing for any dataset.
    Runs CSP+SVM classifier.
    x_gan: (trials, ch, T), train/val/test_csp: {'x': (T,ch,trials), 'y': ...}"""

    os.makedirs(output_dir, exist_ok=True)
    set_log_path(os.path.join(output_dir, 'run_log.txt'))
    log_message(f"\n{'='*60}\n  MI SUBJECT {sid} [{dataset_tag}]\n{'='*60}")

    try:
        log_message(f"  Training: {x_gan.shape[0]} trials, {x_gan.shape[1]} ch, {x_gan.shape[2]} T")
        log_message(f"  Validation: {val_csp['x'].shape[2]} trials | Test: {test_csp['x'].shape[2]} trials")

        # ── Combined train+val for final evaluation ──
        tv_csp = {'x': np.concatenate([train_csp['x'], val_csp['x']], axis=2),
                   'y': np.concatenate([train_csp['y'], val_csp['y']])}

        # ── Baseline CSP-SVM ──
        log_message("\n--- Step 2: Baseline CSP-SVM ---")
        fixed_W = compute_csp_filters(train_csp, num_channels)
        tr_feats = extract_csp_features(train_csp, fixed_W)
        va_feats = extract_csp_features(val_csp, fixed_W)
        te_feats = extract_csp_features(test_csp, fixed_W)
        tv_feats = {'x': np.concatenate([tr_feats['x'], va_feats['x']], axis=1),
                     'y': np.concatenate([tr_feats['y'], va_feats['y']])}

        bl_csp = evaluate_svm_on_features(train_svm_on_features(tv_feats), te_feats)[0]
        log_message(f"  CSP+SVM Baseline: {bl_csp:.2f}%")

        # ── GAN training + batch generation ──
        log_message(f"\n--- Step 3: {NUM_RUNS_PER_SUBJECT} GAN Run(s) x {NUM_BATCHES_PER_RUN} Batches ---")
        all_batches = []
        for run_idx in range(NUM_RUNS_PER_SUBJECT):
            gen = train_cgan(x_gan, y_labels, GAN_EPOCHS, min(64, x_gan.shape[0]), num_channels,
                             output_dir, sid, run_idx+1, x_gan.shape[2],
                             csp_filters=fixed_W, feat_loss_weight=5.0)
            if not gen: continue

            plot_training_loss(output_dir, run_idx+1, sid)

            log_message(f"  Generating & filtering batches (Run {run_idx+1})...")
            for bi in range(NUM_BATCHES_PER_RUN):
                sc0, sc1 = generate_synthetic(gen, NUM_SYNTH_PER_CLASS, 0), generate_synthetic(gen, NUM_SYNTH_PER_CLASS, 1)
                if sc0.size == 0: continue
                sg, sy = np.concatenate((sc0, sc1), axis=0), np.concatenate((np.ones(sc0.shape[0]), np.ones(sc1.shape[0])+1))

                raw_feats = extract_csp_features({'x': np.transpose(sg, (2, 1, 0)), 'y': sy}, fixed_W)
                filt_feats, keep_mask = filter_synthetic_trials(tr_feats, raw_feats, sy, keep_ratio=0.6)
                f_sg, f_sy = sg[keep_mask], sy[keep_mask]

                if f_sg.shape[0] < 10: continue

                psd_c = compute_psd_correlation(x_gan, f_sg, fs=fs)
                amp_s = compute_amplitude_similarity(x_gan, f_sg)
                bpc = compute_band_power_correlation(x_gan, f_sg, fs=fs)
                combined = 0.4*psd_c + 0.3*amp_s + 0.3*bpc

                log_message(f"    R{run_idx+1} B{bi+1}: kept={f_sg.shape[0]}/{sg.shape[0]} psd={psd_c:.3f} amp={amp_s:.3f} bpc={bpc:.3f} score={combined:.3f}")
                all_batches.append({'run': run_idx+1, 'batch': bi+1, 'combined_score': combined,
                                    'synth_feats': filt_feats, 'synth_gan': f_sg, 'synth_y': f_sy})

        if not all_batches:
            log_message("CRITICAL: No valid batches")
            return

        # ── Step 4: CSP-SVM Strategy Selection (filtered data) ──
        log_message(f"\n--- Step 4: CSP-SVM Strategy Selection ({len(all_batches)} batches, FILTERED) ---")
        all_strats = []
        n_real_tr = tr_feats['x'].shape[1]  # number of real training trials
        for b in sorted(all_batches, key=lambda x: x['combined_score'], reverse=True)[:10]:
            sf, n_synth = b['synth_feats'], b['synth_feats']['x'].shape[1]
            for r in MIX_RATIOS:
                if r == 0:
                    # Synth-only: use same count as real training data, balanced
                    eval_d = balanced_subsample(sf, n_real_tr)
                    name = 'SynthOnly'
                else:
                    na = min(int(n_real_tr * r), n_synth)
                    synth_sub = balanced_subsample(sf, na)
                    eval_d = {'x': np.concatenate([tr_feats['x'], synth_sub['x']], axis=1),
                               'y': np.concatenate([tr_feats['y'], synth_sub['y']])}
                    name = f'Aug{int(r*100)}'
                va = evaluate_svm_on_features(train_svm_on_features(eval_d), va_feats)[0]
                all_strats.append({'name': name, 'val_acc': va, 'b': b, 'r': r})

        all_strats.sort(key=lambda s: s['val_acc'], reverse=True)
        for s in all_strats[:8]:
            log_message(f"  {s['name']:12s} val={s['val_acc']:.2f}% (R{s['b']['run']}B{s['b']['batch']})")
        best_csp_strat = all_strats[0]
        log_message(f"\n  CSP SELECTED: {best_csp_strat['name']} val={best_csp_strat['val_acc']:.2f}%")


        # ── Step 5: Final CSP-SVM Evaluation (filtered) ──
        log_message("\n--- Step 5: Final CSP-SVM Evaluation ---")
        n_real_tv = tv_feats['x'].shape[1]  # train+val real trial count

        # Synth-only: balanced subsample at real count
        best_so_csp_batch = max([s for s in all_strats if s['r']==0], key=lambda x: x['val_acc'])['b']['synth_feats']
        so_model = train_svm_on_features(balanced_subsample(best_so_csp_batch, n_real_tv))
        so_csp = evaluate_svm_on_features(so_model, te_feats)[0] if so_model else 0.0

        # Best strategy
        sf = best_csp_strat['b']['synth_feats']
        if best_csp_strat['r'] == 0:
            final_d = balanced_subsample(sf, n_real_tv)
        else:
            na = min(int(n_real_tv * best_csp_strat['r']), sf['x'].shape[1])
            synth_sub = balanced_subsample(sf, na)
            final_d = {'x': np.concatenate([tv_feats['x'], synth_sub['x']], axis=1),
                        'y': np.concatenate([tv_feats['y'], synth_sub['y']])}
        best_model = train_svm_on_features(final_d)
        best_csp, cm_csp, _ = evaluate_svm_on_features(best_model, te_feats) if best_model else (0, np.zeros((2,2)), np.array([]))

        # ── Summary ──
        log_message(f"\n  === SUMMARY S{sid} [{dataset_tag}] ===")
        log_message(f"  CSP+SVM: BL={bl_csp:.2f}%  SO={so_csp:.2f}%  Best({best_csp_strat['name']})={best_csp:.2f}%")
        log_message(f"  CSP+SVM CM:\n{cm_csp}")

        # ── Plots ──
        best_sg = best_csp_strat['b']['synth_gan']  # filtered, for PSD plot
        plot_acc(bl_csp, so_csp, best_csp, sid, output_dir)
        plot_psd_comparison(x_gan, best_sg, fs=fs, subject_id=sid, output_dir=output_dir, paradigm=f'MI-{dataset_tag}')

        # ── Save ──
        scipy.io.savemat(os.path.join(output_dir, f'synthetic_S{sid}.mat'), {
            'synthetic_filtered_x': best_csp_strat['b']['synth_gan'],
            'synthetic_filtered_y': best_csp_strat['b']['synth_y'],
            'norm_ch_mean': ch_mean, 'norm_ch_std': ch_std,
            'csp_baseline_acc': bl_csp, 'csp_synth_only_acc': so_csp, 'csp_best_aug_acc': best_csp,
            'csp_best_strategy': best_csp_strat['name'], 'csp_best_ratio': best_csp_strat['r']
        })
        log_message(f"  Saved: synthetic_S{sid}.mat (filtered={best_csp_strat['b']['synth_gan'].shape[0]} trials)")

        return {'sid': sid, 'bl_csp': bl_csp, 'so_csp': so_csp, 'best_csp': best_csp,
                'best_csp_name': best_csp_strat['name']}

    except Exception as e:
        log_message(f"ERROR S{sid}: {traceback.format_exc()}")
        return None


# =============================================================================
# 8.  ENTRY POINT
# =============================================================================

def run_bci4(target_subjects):
    """Run the full pipeline on BCI Competition IV Dataset 2a."""
    log_message(f"\n\n{'#'*70}\n  STARTING BCI-IV-2a PIPELINE\n{'#'*70}")
    d1, d2 = scipy.io.loadmat('data1.mat'), scipy.io.loadmat('data2.mat')
    x1, x2 = d1['xsubi_all'], d2['txsubi_all']
    n_subjects = x1.shape[1]
    fs = 250

    if isinstance(target_subjects, str) and target_subjects.lower() == "all":
        subjects = list(range(n_subjects))
    elif isinstance(target_subjects, list):
        subjects = [s - 1 for s in target_subjects if 1 <= s <= n_subjects]
    else:
        subjects = list(range(n_subjects))

    output_base = 'results_MI_BCI4'
    os.makedirs(output_base, exist_ok=True)
    summary_csv = os.path.join(output_base, 'summary_results.csv')
    with open(summary_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['subject', 'csp_baseline', 'csp_synth_only', 'csp_best_aug', 'csp_best_strategy'])

    results = []
    for si in subjects:
        sid = si + 1
        x_gan, y_labels, train_csp, ch_mean, ch_std = preprocess_training_data(x1[0, si], str(sid))
        val_csp, test_csp = preprocess_valtest_data(x2[0, si], str(sid), NUM_VALID_TRIALS_PER_CLASS, ch_mean, ch_std)
        if x_gan is None or val_csp is None:
            log_message(f"CRITICAL: Preproc failed S{sid}")
            continue

        r = process_mi_subject(x_gan, y_labels, train_csp, val_csp, test_csp,
                               ch_mean, ch_std, NUM_SELECTED_CHANNELS_BCI4, sid,
                               f'{output_base}/Subject_{sid}', fs=fs, dataset_tag='BCI4')
        if r:
            results.append(r)
            with open(summary_csv, 'a', newline='') as f:
                csv.writer(f).writerow([sid, f'{r["bl_csp"]:.2f}', f'{r["so_csp"]:.2f}', f'{r["best_csp"]:.2f}',
                                        r['best_csp_name']])

    if results:
        log_message(f"\n{'='*60}\n  BCI4 FINAL SUMMARY\n{'='*60}")
        for r in results:
            log_message(f"  S{r['sid']:2d} CSP: BL={r['bl_csp']:.1f}% Best={r['best_csp']:.1f}%")
        log_message(f"  Mean CSP: BL={np.mean([r['bl_csp'] for r in results]):.1f}% Best={np.mean([r['best_csp'] for r in results]):.1f}%")
    return results


def run_cho2017(target_subjects):
    """Run the full pipeline on Cho2017 dataset (DATA1.mat - DATA5.mat)."""
    log_message(f"\n\n{'#'*70}\n  STARTING CHO2017 PIPELINE\n{'#'*70}")
    datasets_to_process = [f'DATA{i}.mat' for i in range(1, 6)]
    fs = CHO_FS
    all_results = []

    for dataset_name in datasets_to_process:
        log_message(f"\n\n{'='*20} DATASET: {dataset_name} {'='*20}")
        try:
            data_mat = scipy.io.loadmat(dataset_name)
            xsubi_all = data_mat['xsubi_all']
        except (FileNotFoundError, KeyError) as e:
            log_message(f"FATAL: Could not load {dataset_name}: {e}. SKIPPING.")
            continue

        n_subjects = xsubi_all.shape[1]
        dataset_prefix = dataset_name.split('.')[0]

        if isinstance(target_subjects, str) and target_subjects.lower() == "all":
            subjects = list(range(n_subjects))
        elif isinstance(target_subjects, list):
            subjects = [s - 1 for s in target_subjects if 1 <= s <= n_subjects]
        else:
            subjects = list(range(n_subjects))

        output_base = f'results_MI_{dataset_prefix}'
        os.makedirs(output_base, exist_ok=True)
        summary_csv = os.path.join(output_base, 'summary_results.csv')
        with open(summary_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['subject', 'csp_baseline', 'csp_synth_only', 'csp_best_aug', 'csp_best_strategy'])

        results = []
        for si in subjects:
            sid = si + 1
            res = preprocess_cho2017_subject(xsubi_all[0, si], str(sid), dataset_name)
            x_gan, y_labels, train_csp, val_csp, test_csp, ch_mean, ch_std = res
            if x_gan is None:
                log_message(f"CRITICAL: Preproc failed S{sid} in {dataset_name}")
                continue

            r = process_mi_subject(x_gan, y_labels, train_csp, val_csp, test_csp,
                                   ch_mean, ch_std, NUM_SELECTED_CHANNELS_CHO, sid,
                                   f'{output_base}/Subject_{sid}', fs=fs, dataset_tag=dataset_prefix)
            if r:
                results.append(r)
                with open(summary_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([sid, f'{r["bl_csp"]:.2f}', f'{r["so_csp"]:.2f}', f'{r["best_csp"]:.2f}',
                                            r['best_csp_name']])

        if results:
            log_message(f"\n{'='*60}\n  {dataset_prefix} FINAL SUMMARY\n{'='*60}")
            for r in results:
                log_message(f"  S{r['sid']:2d} CSP: BL={r['bl_csp']:.1f}% Best={r['best_csp']:.1f}%")
            log_message(f"  Mean CSP: BL={np.mean([r['bl_csp'] for r in results]):.1f}% Best={np.mean([r['best_csp'] for r in results]):.1f}%")
        all_results.extend(results)
    return all_results


if __name__ == '__main__':
    DATASET = 'BCI4'       # 'BCI4', 'CHO2017', or 'BOTH'

    # Subject selection: "all" or list of 1-based subject numbers
    TARGET_SUBJECTS = [1,5,6,7]

    if DATASET == 'BCI4':
        run_bci4(TARGET_SUBJECTS)
    elif DATASET == 'CHO2017':
        run_cho2017(TARGET_SUBJECTS)
    elif DATASET == 'BOTH':
        run_bci4(TARGET_SUBJECTS)
        run_cho2017(TARGET_SUBJECTS)
    else:
        log_message(f"Unknown DATASET: {DATASET}. Use 'BCI4', 'CHO2017', or 'BOTH'.")