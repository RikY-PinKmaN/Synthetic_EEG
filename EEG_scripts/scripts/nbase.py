#!/usr/bin/env python3
"""
Multi-Paradigm EEG cWGAN-GP Data Augmentation Framework
=========================================================
MI pipeline is verbatim from working main.py.
P300 and RS to be added in later steps.
"""

# ── IMPORTS (from main.py) ────────────────────────────────────────────────────
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import ellip, ellipord, filtfilt, welch, savgol_filter, butter, iirnotch
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import traceback
from sklearn.svm import SVC
from scipy.linalg import eigh
import scipy.stats as stats
import os, csv

# ── CONSTANTS (from main.py lines 22-35) ─────────────────────────────────────
SELECTED_CHANNELS_1_BASED = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17]
SELECTED_CHANNELS_0_BASED = [idx - 1 for idx in SELECTED_CHANNELS_1_BASED]
NUM_SELECTED_CHANNELS = len(SELECTED_CHANNELS_0_BASED)
NUM_TRAIN_TRIALS_PER_CLASS = 40
NUM_VALID_TRIALS_PER_CLASS = 10
NUM_RUNS_PER_SUBJECT = 5
NUM_CLASSES_CGAN = 2
LATENT_DIM_CGAN = 100
EMBEDDING_DIM_CGAN = 25
SEED_VALUE = 42
GAN_EPOCHS = 2000

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
# 1.  PREPROCESSING  (verbatim from main.py lines 68-258)
# =============================================================================

def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    nyq = 0.5 * fs
    low_stop = max(0.1, lowcut - 1.0)
    high_stop = highcut + 1.0
    if high_stop >= nyq: high_stop = nyq - 0.1
    wp = [lowcut / nyq, highcut / nyq]
    ws = [low_stop / nyq, high_stop / nyq]
    epsilon = 1e-6
    wp = np.clip(wp, epsilon, 1.0 - epsilon).tolist()
    ws = np.clip(ws, epsilon, 1.0 - epsilon).tolist()
    if ws[0] >= wp[0]:
        ws[0] = wp[0] * 0.9 if wp[0] > epsilon * 10 else wp[0] * 0.5
        ws[0] = max(ws[0], epsilon)
    if ws[1] <= wp[1]:
        ws[1] = wp[1] * 1.1 if wp[1] < (1.0 - epsilon * 10) else (wp[1] + (1.0-wp[1])*0.5)
        ws[1] = min(ws[1], 1.0 - epsilon)
    n, wn = ellipord(wp, ws, rp, rs)
    b, a = ellip(n, rp, rs, wn, btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def _get_eeg_data_from_field(subject_struct_item, field_name, subject_id_str, data_type_str="Data"):
    if not hasattr(subject_struct_item, 'dtype') or subject_struct_item.dtype.names is None:
        return None
    if field_name not in subject_struct_item.dtype.names:
        return None
    field_content = subject_struct_item[field_name]
    actual_data = None
    if isinstance(field_content, np.ndarray) and field_content.shape == (1,1) and \
       hasattr(field_content[0,0], 'shape'):
        actual_data = field_content[0,0]
    else:
        actual_data = field_content
    if not isinstance(actual_data, np.ndarray): return None
    if field_name == 'y': return actual_data.flatten()
    if actual_data.ndim < 2: return None
    if actual_data.ndim == 2:
        s, c = actual_data.shape
        if s > c and c < 100:
            actual_data = np.expand_dims(actual_data, axis=2)
        elif c > s and s < 100:
            actual_data = np.expand_dims(actual_data.T, axis=2)
        else:
            actual_data = np.expand_dims(actual_data, axis=2)
    if actual_data.ndim != 3: return None
    if 0 in actual_data.shape: return None
    return actual_data


def preprocess_training_data(raw_subject_struct, subject_id_str,
                             num_train_per_class=NUM_TRAIN_TRIALS_PER_CLASS,
                             selected_channels_0_based=SELECTED_CHANNELS_0_BASED,
                             lowfreq=8, highfreq=35, fs=250,
                             startSample=125, endSample=624):
    try:
        all_x = _get_eeg_data_from_field(raw_subject_struct, 'x', subject_id_str, "Train-'x'")
        if all_x is None: return (None,) * 5
        all_y = _get_eeg_data_from_field(raw_subject_struct, 'y', subject_id_str, "Train-'y'")
        n_total = all_x.shape[2]
        if all_y is None:
            if n_total < 2: return (None,) * 5
            n1 = n_total // 2
            all_y = np.concatenate((np.ones(n1), np.ones(n_total - n1) + 1))
        if len(all_y) != n_total: return (None,) * 5

        c1_idx = np.where(all_y == 1)[0]
        c2_idx = np.where(all_y == 2)[0]
        if len(c1_idx) < num_train_per_class or len(c2_idx) < num_train_per_class:
            log_message(f"  ERROR S{subject_id_str}: Not enough trials")
            return (None,) * 5

        tr_c1 = c1_idx[:num_train_per_class]
        tr_c2 = c2_idx[:num_train_per_class]
        x_train = np.concatenate((all_x[:, :, tr_c1], all_x[:, :, tr_c2]), axis=2)
        y_train = np.concatenate((np.ones(num_train_per_class), np.ones(num_train_per_class) + 1))

        filtered = elliptical_filter(x_train, lowcut=lowfreq, highcut=highfreq, fs=fs)
        windowed = filtered[startSample:endSample + 1, :, :]
        selected = windowed[:, selected_channels_0_based, :]
        if selected.size == 0: return (None,) * 5

        g_min = np.min(selected)
        g_max = np.max(selected)
        g_range = g_max - g_min
        if g_range < 1e-8:
            normalized = np.zeros_like(selected)
        else:
            normalized = 2 * (selected - g_min) / g_range - 1

        x_gan = np.transpose(normalized, (2, 1, 0))  # (trials, ch, T)
        csp_dict = {'x': normalized, 'y': y_train}    # (T, ch, trials)
        log_message(f"  S{subject_id_str} (Train): GAN {x_gan.shape}, CSP {csp_dict['x'].shape}")
        return x_gan, y_train, csp_dict, g_min, g_max
    except Exception as e:
        log_message(f"ERROR S{subject_id_str} (Train): {e}\n{traceback.format_exc()}")
        return (None,) * 5


def preprocess_valtest_data(raw_subject_struct, subject_id_str,
                            num_valid_per_class, train_min, train_max,
                            selected_channels_0_based=SELECTED_CHANNELS_0_BASED,
                            lowfreq=8, highfreq=35, fs=250,
                            startSample=115, endSample=614):
    try:
        all_x = _get_eeg_data_from_field(raw_subject_struct, 'x', subject_id_str, "Val/Test-'x'")
        if all_x is None: return None, None
        all_y = _get_eeg_data_from_field(raw_subject_struct, 'y', subject_id_str, "Val/Test-'y'")
        n_total = all_x.shape[2]
        if all_y is None:
            if n_total < 2: return None, None
            n1 = n_total // 2
            all_y = np.concatenate((np.ones(n1), np.ones(n_total - n1) + 1))
        if len(all_y) != n_total: return None, None

        c1_idx = np.where(all_y == 1)[0]
        c2_idx = np.where(all_y == 2)[0]
        if len(c1_idx) <= num_valid_per_class or len(c2_idx) <= num_valid_per_class:
            return None, None

        val_c1 = c1_idx[:num_valid_per_class]
        tst_c1 = c1_idx[num_valid_per_class:]
        val_c2 = c2_idx[:num_valid_per_class]
        tst_c2 = c2_idx[num_valid_per_class:]

        def process_subset(x_data, y_labels):
            if x_data.size == 0: return None
            filtered = elliptical_filter(x_data, lowcut=lowfreq, highcut=highfreq, fs=fs)
            windowed = filtered[startSample:endSample + 1, :, :]
            selected = windowed[:, selected_channels_0_based, :]
            if selected.size == 0: return None
            rng = train_max - train_min
            if rng < 1e-8:
                normalized = np.zeros_like(selected)
            else:
                normalized = np.clip(2 * (selected - train_min) / rng - 1, -1, 1)
            return {'x': normalized, 'y': y_labels.squeeze()}

        x_val = np.concatenate((all_x[:, :, val_c1], all_x[:, :, val_c2]), axis=2)
        y_val = np.concatenate((np.ones(len(val_c1)), np.ones(len(val_c2)) + 1))
        x_tst = np.concatenate((all_x[:, :, tst_c1], all_x[:, :, tst_c2]), axis=2)
        y_tst = np.concatenate((np.ones(len(tst_c1)), np.ones(len(tst_c2)) + 1))

        return process_subset(x_val, y_val), process_subset(x_tst, y_tst)
    except Exception as e:
        log_message(f"ERROR S{subject_id_str} (Val/Test): {e}\n{traceback.format_exc()}")
        return None, None


# =============================================================================
# 2.  CSP-SVM  (verbatim from main.py lines 261-423)
# =============================================================================

def csp(data):
    class1_indices = np.where(data['y'] == 1)[0]
    class2_indices = np.where(data['y'] == 2)[0]
    if len(class1_indices) == 0 or len(class2_indices) == 0:
        log_message("CSP Error: One class has no trials.")
        return np.eye(data['x'].shape[1])
    X1 = data['x'][:, :, class1_indices]
    X2 = data['x'][:, :, class2_indices]
    n_channels = X1.shape[1]
    cov1 = np.zeros((n_channels, n_channels))
    valid_trials_c1 = 0
    for trial in range(X1.shape[2]):
        current_trial_data = X1[:,:,trial]
        cov_trial = np.cov(current_trial_data, rowvar=False)
        if not np.all(np.isnan(cov_trial)):
            cov1 += cov_trial
            valid_trials_c1 += 1
    if valid_trials_c1 > 0: cov1 /= valid_trials_c1

    cov2 = np.zeros((n_channels, n_channels))
    valid_trials_c2 = 0
    for trial in range(X2.shape[2]):
        current_trial_data = X2[:,:,trial]
        cov_trial = np.cov(current_trial_data, rowvar=False)
        if not np.all(np.isnan(cov_trial)):
            cov2 += cov_trial
            valid_trials_c2 += 1
    if valid_trials_c2 > 0: cov2 /= valid_trials_c2

    epsilon_reg = 1e-9
    cov1_reg = cov1 + epsilon_reg * np.eye(n_channels)
    cov2_reg = cov2 + epsilon_reg * np.eye(n_channels)
    try:
        evals, evecs = eigh(cov1_reg, cov1_reg + cov2_reg)
    except np.linalg.LinAlgError:
        evals, evecs = eigh(cov1_reg)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    W = evecs.T
    return W


def featcrossval(finaldataset, ChRanking, numchannel):
    W = csp(finaldataset)
    if numchannel > 6:
        if W.shape[0] < 6: selectedw = W
        else: selectedw = np.vstack((W[0:3, :], W[-3:, :]))
    elif W.shape[0] >= 2:
        selectedw = np.vstack((W[0, :][np.newaxis,:], W[-1, :][np.newaxis,:]))
    elif W.shape[0] == 1: selectedw = W
    else:
        return {'x': np.array([]), 'y': finaldataset['y']}, np.array([])

    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]
    if num_features == 0: return {'x': np.zeros((0, ntrial)), 'y': finaldataset['y']}, selectedw

    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}
    for trial in range(ntrial):
        projected = finaldataset['x'][:, :, trial]
        selectedZ = np.dot(projected, selectedw.T)
        variances = np.var(selectedZ, axis=0)
        variances_reg = variances + 1e-9
        sum_var = np.sum(variances_reg)
        if sum_var < 1e-9: producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_var)
    return producedfeatur, selectedw


def featcrostest(finaldataset, ChRanking, numchannel, selectedw):
    if selectedw is None or selectedw.shape[0] == 0:
        return {'x': np.zeros((0, finaldataset['x'].shape[2])), 'y': finaldataset['y']}
    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]
    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}
    for trial in range(ntrial):
        projected = finaldataset['x'][:, :, trial]
        selectedZ = np.dot(projected, selectedw.T)
        variances = np.var(selectedZ, axis=0)
        variances_reg = variances + 1e-9
        sum_var = np.sum(variances_reg)
        if sum_var < 1e-9: producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_var)
    return producedfeatur


def fitcsvm(X, Y, **kwargs):
    standardize = kwargs.get('Standardize', False)
    kernel = kwargs.get('KernelFunction', 'linear')
    kernel_map = {'linear': 'linear', 'rbf': 'rbf', 'gaussian': 'rbf', 'polynomial': 'poly'}
    model = SVC(kernel=kernel_map.get(kernel, kernel), probability=True, C=1.0)
    if X.shape[0] == 0: return None
    if standardize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std < 1e-8] = 1e-8
        X_norm = (X - X_mean) / X_std
        model.X_mean_ = X_mean
        model.X_std_ = X_std
        model.fit(X_norm, Y)
    else:
        model.fit(X, Y)
    return model


def predict_svm(model, X):
    if model is None: return np.array([])
    if X.shape[0] == 0: return np.array([])
    if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
        X_norm = (X - model.X_mean_) / model.X_std_
        return model.predict(X_norm)
    return model.predict(X)


def train_cspsvm(data):
    if data is None or 'x' not in data or data['x'].shape[2] == 0:
        return None, None
    n_ch = data['x'].shape[1]
    features_dict, spatial_filters = featcrossval(data, ChRanking=None, numchannel=n_ch)
    if features_dict['x'].size == 0: return None, spatial_filters
    X_train = features_dict['x'].T
    y_train = features_dict['y']
    if X_train.shape[0] == 0: return None, spatial_filters
    model = fitcsvm(X_train, y_train, Standardize=True, KernelFunction='linear')
    return model, spatial_filters


def evaluate_cspsvm(model, spatial_filters, data_test):
    if model is None: return 0, np.zeros((2,2)), np.array([])
    if spatial_filters is None or spatial_filters.size == 0: return 0, np.zeros((2,2)), np.array([])
    if data_test is None or 'x' not in data_test or data_test['x'].shape[2] == 0:
        return 0, np.zeros((2,2)), np.array([])
    n_ch = data_test['x'].shape[1]
    features_test = featcrostest(data_test, ChRanking=None, numchannel=n_ch, selectedw=spatial_filters)
    if features_test['x'].size == 0:
        return 0, np.zeros((2,2)), np.array([])
    X_test = features_test['x'].T
    y_true = features_test['y']
    if X_test.shape[0] == 0:
        return 0, np.zeros((2,2)), np.array([])
    y_pred = predict_svm(model, X_test)
    accuracy = 0; cm = np.zeros((2,2))
    if len(y_true) > 0 and len(y_pred) > 0 and len(y_true) == len(y_pred):
        accuracy = np.mean(y_pred == y_true) * 100
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    return accuracy, cm, y_pred


# =============================================================================
# 3.  GAN ARCHITECTURE  (verbatim from main.py lines 426-471)
# =============================================================================

def build_generator(num_channels, time_points, latent_dim=LATENT_DIM_CGAN,
                    num_classes=NUM_CLASSES_CGAN, embedding_dim=EMBEDDING_DIM_CGAN):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(num_classes, embedding_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    merged_input = layers.Concatenate()([noise_input, label_embedding])
    initial_reshape_dim1 = (time_points + 7) // 8
    initial_filters = 64
    x = layers.Dense(initial_reshape_dim1 * initial_filters)(merged_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((initial_reshape_dim1, initial_filters))(x)
    x = layers.Conv1DTranspose(128, 4, 2, "same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1DTranspose(64, 4, 2, "same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1DTranspose(num_channels, 7, 2, "same", activation="tanh")(x)
    current_time_points_gen = x.shape[1]
    if current_time_points_gen > time_points:
        crop = current_time_points_gen - time_points
        x = layers.Cropping1D((crop // 2, crop - crop // 2))(x)
    elif current_time_points_gen < time_points:
        pad = time_points - current_time_points_gen
        x = layers.ZeroPadding1D((pad // 2, pad - pad // 2))(x)
    x = layers.Permute((2, 1))(x)
    return models.Model([noise_input, label_input], x, name="generator")


def build_critic(num_channels, time_points, num_classes=NUM_CLASSES_CGAN,
                 embedding_dim=EMBEDDING_DIM_CGAN):
    data_input = layers.Input(shape=(num_channels, time_points))
    label_input = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(num_classes, embedding_dim)(label_input)
    x = layers.Permute((2, 1))(data_input)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    data_features_flat = layers.Flatten()(x)
    label_embedding_flat = layers.Flatten()(label_embedding)
    merged_features = layers.Concatenate()([data_features_flat, label_embedding_flat])
    output_score = layers.Dense(1)(merged_features)
    return models.Model([data_input, label_input], output_score, name="critic")


# =============================================================================
# 4.  LOSS FUNCTIONS  (verbatim from main.py lines 473-500)
# =============================================================================

def frequency_domain_loss(real_data, generated_data):
    real_data_t = tf.transpose(real_data, perm=[0, 2, 1])
    gen_data_t = tf.transpose(generated_data, perm=[0, 2, 1])
    real_fft = tf.abs(tf.signal.rfft(real_data_t))
    gen_fft = tf.abs(tf.signal.rfft(gen_data_t))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))


def smooth_eeg(data, window_size=5):
    smoothed_data = np.copy(data)
    if window_size % 2 == 0: window_size += 1
    if data.shape[2] <= window_size: return data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            smoothed_data[i, j, :] = savgol_filter(data[i, j, :], window_size, 2)
    return smoothed_data


def gradient_penalty(critic, real_samples, fake_samples, real_labels, lambda_gp=10):
    alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    gradients = gp_tape.gradient(pred, [interpolated])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
    gp_loss = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return lambda_gp * gp_loss


# =============================================================================
# 5.  TRAINING  (verbatim from main.py lines 502-604)
# =============================================================================

def train_cgan(eeg_data, eeg_labels, epochs, batch_size, num_gan_channels,
               output_dir, subject_id, run_number,
               time_points=500, latent_dim=LATENT_DIM_CGAN, n_critic_steps=5,
               freq_loss_weight=1.5, num_classes=NUM_CLASSES_CGAN,
               embedding_dim=EMBEDDING_DIM_CGAN):
    """Verbatim from main.py train_wgan_gp_cgan."""
    if not isinstance(eeg_labels, np.ndarray): eeg_labels = np.array(eeg_labels)
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)
    if not (np.all(labels_0_1 >= 0) and np.all(labels_0_1 < num_classes)):
        log_message(f"ERROR: Labels not 0-{num_classes-1}. Found: {np.unique(labels_0_1)}")
        return None
    labels_0_1_reshaped = labels_0_1[:, np.newaxis]

    @tf.function
    def train_step(real_eeg_batch, real_labels_batch, generator, critic,
                   gen_opt, crit_opt, lambda_gp_val=10,
                   latent_dim_step=LATENT_DIM_CGAN, n_critic=5,
                   num_cls_step=NUM_CLASSES_CGAN):
        batch_size_tf = tf.shape(real_eeg_batch)[0]
        for _ in range(n_critic):
            noise = tf.random.normal([batch_size_tf, latent_dim_step])
            random_labels = tf.random.uniform(shape=[batch_size_tf, 1],
                                              minval=0, maxval=num_cls_step, dtype=tf.int32)
            with tf.GradientTape() as tape:
                fake_eeg = generator([noise, random_labels], training=True)
                real_output = critic([real_eeg_batch, real_labels_batch], training=True)
                fake_output = critic([fake_eeg, random_labels], training=True)
                gp = gradient_penalty(critic, real_eeg_batch, fake_eeg,
                                      real_labels_batch, lambda_gp_val)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
            crit_opt.apply_gradients(zip(
                tape.gradient(d_loss, critic.trainable_variables),
                critic.trainable_variables))
        noise_g = tf.random.normal([batch_size_tf, latent_dim_step])
        random_labels_g = tf.random.uniform(shape=[batch_size_tf, 1],
                                            minval=0, maxval=num_cls_step, dtype=tf.int32)
        with tf.GradientTape() as tape_g:
            fake_eeg_g = generator([noise_g, random_labels_g], training=True)
            fake_output_g = critic([fake_eeg_g, random_labels_g], training=True)
            g_loss = -tf.reduce_mean(fake_output_g)
        gen_opt.apply_gradients(zip(
            tape_g.gradient(g_loss, generator.trainable_variables),
            generator.trainable_variables))
        return d_loss, g_loss, fake_eeg_g

    generator = build_generator(num_gan_channels, time_points, latent_dim, num_classes, embedding_dim)
    critic = build_critic(num_gan_channels, time_points, num_classes, embedding_dim)
    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    d_losses, g_losses = [], []
    if eeg_data.shape[0] == 0: return None
    if eeg_data.shape[0] < batch_size: batch_size = eeg_data.shape[0]
    num_batches = eeg_data.shape[0] // batch_size
    if num_batches == 0 and eeg_data.shape[0] > 0: num_batches = 1

    log_message(f"\n  cWGAN-GP S{subject_id} Run{run_number} | T={time_points} nch={num_gan_channels} ep={epochs}")
    log_message(f"  Gen params={generator.count_params():,}  Crit params={critic.count_params():,}")

    for epoch in range(epochs):
        perm = np.random.permutation(eeg_data.shape[0])
        data_shuffled = eeg_data[perm]
        labels_shuffled = labels_0_1_reshaped[perm]
        epoch_d, epoch_g = 0.0, 0.0
        for bi in range(num_batches):
            s, e = bi * batch_size, (bi + 1) * batch_size
            rx = tf.convert_to_tensor(data_shuffled[s:e].astype(np.float32), dtype=tf.float32)
            ry = tf.convert_to_tensor(labels_shuffled[s:e], dtype=tf.int32)
            if rx.shape[0] == 0: continue

            d_loss, g_loss_base, _ = train_step(
                rx, ry, generator, critic, gen_opt, crit_opt,
                latent_dim_step=latent_dim, n_critic=n_critic_steps,
                num_cls_step=num_classes)

            current_g_total = g_loss_base.numpy()

            # Spectral loss — SEPARATE tape, REAL labels (main.py lines 569-578)
            if freq_loss_weight > 0:
                noise_freq = tf.random.normal([tf.shape(rx)[0], latent_dim])
                labels_freq = ry  # REAL labels for class-matched comparison
                with tf.GradientTape() as freq_tape:
                    fake_freq = generator([noise_freq, labels_freq], training=True)
                    freq_loss = freq_loss_weight * frequency_domain_loss(rx, fake_freq)
                freq_grads = freq_tape.gradient(freq_loss, generator.trainable_variables)
                if not any(g is None for g in freq_grads):
                    gen_opt.apply_gradients(zip(freq_grads, generator.trainable_variables))
                current_g_total += freq_loss.numpy()

            epoch_d += d_loss.numpy()
            epoch_g += current_g_total
        d_losses.append(epoch_d / num_batches if num_batches > 0 else 0)
        g_losses.append(epoch_g / num_batches if num_batches > 0 else 0)
        if epoch % 100 == 0 or epoch == epochs - 1:
            log_message(f"    E{epoch:04d}: D={d_losses[-1]:.4f} G={g_losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_losses, label="Critic"); ax.plot(g_losses, label="Gen (W+Freq)")
    ax.set(title=f'Losses S{subject_id} Run{run_number}', xlabel='Epoch', ylabel='Loss')
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'losses_S{subject_id}_run{run_number}.png'))
    plt.close(fig)
    return generator


def generate_synthetic(generator, num_samples, target_label_0_1,
                       latent_dim=LATENT_DIM_CGAN, smooth=True, window_size=5):
    """Verbatim from main.py generate_synthetic_data_cgan."""
    if generator is None or num_samples == 0: return np.array([])
    noise = tf.random.normal([num_samples, latent_dim])
    labels = tf.ones((num_samples, 1), dtype=tf.int32) * target_label_0_1
    synthetic = generator([noise, labels], training=False).numpy()
    if smooth and synthetic.size > 0:
        synthetic = smooth_eeg(synthetic, window_size)
    return synthetic


# =============================================================================
# 6.  PLOTTING (minimal)
# =============================================================================

def plot_acc(baseline, synth_only, best_aug, subject_id, output_dir):
    labels = ['Baseline', 'Synth-Only', 'Best Augmented']
    accs = [baseline, synth_only, best_aug]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, accs, color=['blue', 'red', 'green'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Comparison - S{subject_id}')
    ax.set_ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1,
                f'{yval:.2f}%', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'accuracy_S{subject_id}.png'))
    plt.close(fig)


# =============================================================================
# 7.  ORCHESTRATION  (verbatim from main.py)
# =============================================================================

def process_subject(subject_idx, xsubi_train, xsubi_valtest, output_dir):
    sid = subject_idx + 1
    os.makedirs(output_dir, exist_ok=True)
    set_log_path(os.path.join(output_dir, 'run_log.txt'))
    log_message(f"\n{'='*60}\n  PROCESSING SUBJECT {sid}\n{'='*60}")

    try:
        np.random.seed(SEED_VALUE)
        tf.random.set_seed(SEED_VALUE)

        log_message("\n--- Step 1: Preprocessing ---")
        x_gan, y_labels, train_csp, g_min, g_max = preprocess_training_data(
            xsubi_train, str(sid))
        if x_gan is None:
            log_message(f"CRITICAL: Training failed S{sid}"); return None

        val_csp, test_csp = preprocess_valtest_data(
            xsubi_valtest, str(sid), NUM_VALID_TRIALS_PER_CLASS, g_min, g_max)
        if val_csp is None or test_csp is None:
            log_message(f"CRITICAL: Val/Test failed S{sid}"); return None

        log_message("\n--- Step 2: Baseline ---")
        tv_csp = {
            'x': np.concatenate((train_csp['x'], val_csp['x']), axis=2),
            'y': np.concatenate((train_csp['y'], val_csp['y']))
        }
        model_bl, sf_bl = train_cspsvm(tv_csp)
        baseline_acc, _, _ = evaluate_cspsvm(model_bl, sf_bl, test_csp)
        log_message(f"  Baseline: {baseline_acc:.2f}%")

        log_message(f"\n--- Step 3: {NUM_RUNS_PER_SUBJECT} GAN run(s) ---")
        all_batches = []
        for run_idx in range(NUM_RUNS_PER_SUBJECT):
            run_num = run_idx + 1
            gen = train_cgan(
                eeg_data=x_gan, eeg_labels=y_labels,
                epochs=GAN_EPOCHS, batch_size=min(64, x_gan.shape[0]),
                num_gan_channels=NUM_SELECTED_CHANNELS,
                output_dir=output_dir, subject_id=sid, run_number=run_num,
                time_points=x_gan.shape[2], freq_loss_weight=1.5)
            if gen is None: continue

            log_message(f"  Generating batches (Run {run_num})...")
            npc = val_csp['x'].shape[2] // 2

            for bi in range(30):
                sc0 = generate_synthetic(gen, npc, 0)
                sc1 = generate_synthetic(gen, npc, 1)
                if not (sc0.size > 0 and sc1.size > 0): continue

                sg = np.concatenate((sc0, sc1), axis=0)
                sy = np.concatenate((np.ones(sc0.shape[0]), np.ones(sc1.shape[0]) + 1))
                sd = {'x': np.transpose(sg, (2, 1, 0)), 'y': sy}

                m_s, sf_s = train_cspsvm(sd)
                ba = 0.0
                if m_s: ba, _, _ = evaluate_cspsvm(m_s, sf_s, val_csp)
                log_message(f"    R{run_num} B{bi+1}: val_acc={ba:.1f}%")

                all_batches.append({
                    'run': run_num, 'batch': bi+1, 'val_acc': ba,
                    'synth_csp': sd, 'synth_gan': sg, 'synth_y': sy})

        if not all_batches:
            log_message("CRITICAL: No batches"); return None

        log_message("\n--- Step 4: Strategy Selection ---")
        all_batches.sort(key=lambda b: b['val_acc'], reverse=True)
        top10 = all_batches[:10]

        mix_ratios = [0, 0.25, 0.50, 1.0]
        all_strats = []
        for b in top10:
            for r in mix_ratios:
                if r == 0:
                    sd = b['synth_csp']; name = 'SynthOnly'
                else:
                    name = f'Aug{int(r*100)}pct'
                    nr = train_csp['x'].shape[2]
                    na = int(nr * r)
                    sx = b['synth_csp']['x'][:, :, :na]
                    sy = b['synth_csp']['y'][:na]
                    sd = {'x': np.concatenate((train_csp['x'], sx), axis=2),
                          'y': np.concatenate((train_csp['y'], sy))}
                m, sf = train_cspsvm(sd)
                va, _, _ = evaluate_cspsvm(m, sf, val_csp) if m else (0, None, None)
                all_strats.append({'name': name, 'val_acc': va,
                                   'batch_info': b, 'mix_ratio': r})

        if not all_strats:
            log_message("CRITICAL: No strategies"); return None

        all_strats.sort(key=lambda s: s['val_acc'], reverse=True)
        for s in all_strats[:8]:
            log_message(f"  {s['name']:15s} val={s['val_acc']:.2f}%")
        best = all_strats[0]
        log_message(f"\n  SELECTED: {best['name']} val={best['val_acc']:.2f}%")

        log_message("\n--- Step 5: Final Test Evaluation ---")
        # Synth-only test accuracy
        so_strats = [s for s in all_strats if s['mix_ratio'] == 0]
        so_acc = 0.0
        if so_strats:
            best_so = max(so_strats, key=lambda s: s['val_acc'])
            m_so, sf_so = train_cspsvm(best_so['batch_info']['synth_csp'])
            if m_so: so_acc, _, _ = evaluate_cspsvm(m_so, sf_so, test_csp)
        log_message(f"  Synth-Only -> Test: {so_acc:.2f}%")

        # Best strategy: retrain on train+val + synth
        if best['mix_ratio'] == 0:
            final_data = best['batch_info']['synth_csp']
        else:
            sf = best['batch_info']['synth_csp']
            nr = tv_csp['x'].shape[2]
            na = int(nr * best['mix_ratio'])
            if na > sf['x'].shape[2]: na = sf['x'].shape[2]
            final_data = {
                'x': np.concatenate((tv_csp['x'], sf['x'][:, :, :na]), axis=2),
                'y': np.concatenate((tv_csp['y'], sf['y'][:na]))
            }
        m_f, sf_f = train_cspsvm(final_data)
        best_acc, cm, _ = evaluate_cspsvm(m_f, sf_f, test_csp) if m_f else (0, np.zeros((2,2)), None)
        log_message(f"  Best({best['name']}) -> Test: {best_acc:.2f}%\n{cm}")

        plot_acc(baseline_acc, so_acc, best_acc, sid, output_dir)

        # Save synthetic data
        best_sg = best['batch_info']['synth_gan']
        best_sy = best['batch_info']['synth_y']
        scipy.io.savemat(os.path.join(output_dir, f'synthetic_S{sid}.mat'),
                         {'synthetic_x': best_sg, 'synthetic_y': best_sy,
                          'norm_min': g_min, 'norm_max': g_max})

        log_message(f"\n  === SUMMARY S{sid} ===")
        log_message(f"  Baseline={baseline_acc:.2f}%  SynthOnly={so_acc:.2f}%  Best({best['name']})={best_acc:.2f}%")
        return {'baseline': baseline_acc, 'synth_only': so_acc, 'best_aug': best_acc,
                'strategy': best['name']}

    except Exception as e:
        log_message(f"CRITICAL ERROR S{sid}: {e}\n{traceback.format_exc()}")
        return None




# =============================================================================
# P300/RS ADDITIONS
# =============================================================================

import glob


# ── Normalization (for P300 zscore, RS minmax) ────────────────────────────────

def normalize_per_channel(x, clip_sigma=3.0, norm_stats=None):
    """Per-channel z-score -> clip -> scale to [-1,1]. x: (T, ch, trials)."""
    if norm_stats is None:
        ch_mean = np.mean(x, axis=(0, 2), keepdims=True)
        ch_std  = np.std(x,  axis=(0, 2), keepdims=True) + 1e-8
    else:
        ch_mean, ch_std = norm_stats
    z = (x - ch_mean) / ch_std
    return np.clip(z / clip_sigma, -1., 1.).astype(np.float32), (ch_mean, ch_std)


def normalize_global_minmax(x, norm_stats=None):
    """Global minmax to [-1,1]. x: (T, ch, trials). Preserves inter-channel ratios for CSP."""
    if norm_stats is None:
        g_min = float(np.min(x))
        g_max = float(np.max(x))
    else:
        g_min, g_max = norm_stats
    g_range = g_max - g_min
    if g_range < 1e-8:
        return np.zeros_like(x, dtype=np.float32), (g_min, g_max)
    normed = 2.0 * (x - g_min) / g_range - 1.0
    return np.clip(normed, -1., 1.).astype(np.float32), (g_min, g_max)


def _decimate_trials(x, factor):
    """Anti-aliased decimation along axis 0. x: (T, ch, trials)."""
    from scipy.signal import decimate as _dec
    out = np.zeros((x.shape[0] // factor, x.shape[1], x.shape[2]), dtype=x.dtype)
    for t in range(x.shape[2]):
        for c in range(x.shape[1]):
            out[:, c, t] = _dec(x[:, c, t], factor)
    return out


# ── P300 Constants & Grid Definitions ────────────────────────────────────────
P300_FS = 80             # Target Fs matching MATLAB pipeline
P300_NCHAN = 8           # Channels: Fz, Cz, Pz, C3, C4, P3, P4, Oz
P300_NREPS = 15          # Flash repetitions per row/col per letter
P300_N_XDAWN = 4         # xDAWN components (CSP analog)
P300_ARTIF_THRESH = 50.0 # µV peak-to-peak artifact threshold
P300_FEAT_TMIN = 0.15    # Feature window start (s) — MATLAB ts_f
P300_FEAT_TMAX = 0.5     # Feature window end (s)
P300_BL_TMIN = -0.1      # Baseline window start (s)
P300_BL_TMAX = 0.0       # Baseline window end (s)

# Grid config per pair: pairs 0,1 → 3×3, pairs 2,3 → 5×5
P300_GRIDS = {
    0: {'rows': [33025, 33026, 33027], 'cols': [33028, 33029, 33030],
        'nq': 5, 'nj': 4},
    1: {'rows': [33025, 33026, 33027], 'cols': [33028, 33029, 33030],
        'nq': 5, 'nj': 4},
    2: {'rows': [33025, 33026, 33027, 33028, 33029],
        'cols': [33030, 33031, 33032, 33033, 33034], 'nq': 5, 'nj': 4},
    3: {'rows': [33025, 33026, 33027, 33028, 33029],
        'cols': [33030, 33031, 33032, 33033, 33034], 'nq': 5, 'nj': 4},
}
# Notch freqs per pair (MATLAB: pairs 0,2 → 4.3/8.6 Hz; pairs 1,3 → 5.8/11.6 Hz)
P300_NOTCH = {0: [4.3, 8.6], 1: [5.8, 11.6], 2: [4.3, 8.6], 3: [5.8, 11.6]}


# ── P300 Preprocessing Functions (matching MATLAB Step_1.m) ─────────────────

def p300_load_filter_resample(filepath, pair_idx, nchan=P300_NCHAN, target_fs=P300_FS):
    """Load one GDF, filter (HP 0.5Hz + LP 30Hz + notch at orig Fs), resample to 80Hz.
    Returns: data_uV (nchan, T_resampled), event_codes (N_events,), event_lats (N_events,)."""
    import mne
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_gdf(filepath, preload=True, eog=None)
    orig_fs = raw.info['sfreq']

    # Extract events before processing
    evs, eid = mne.events_from_annotations(raw)
    id2desc = {v: k for k, v in eid.items()}

    d = raw.get_data()[:nchan, :] * 1e6  # V → µV

    # Butterworth HP + LP at original Fs (MATLAB: fdesign + butter)
    b_hp, a_hp = butter(4, 0.5 / (orig_fs / 2), btype='high')
    b_lp, a_lp = butter(4, 30.0 / (orig_fs / 2), btype='low')
    for j in range(nchan):
        d[j] = filtfilt(b_hp, a_hp, d[j])
        d[j] = filtfilt(b_lp, a_lp, d[j])

    # Notch filters at pair-specific frequencies
    for nf in P300_NOTCH.get(pair_idx, []):
        if nf < orig_fs / 2:
            b_n, a_n = iirnotch(nf, Q=5, fs=orig_fs)
            for j in range(nchan):
                d[j] = filtfilt(b_n, a_n, d[j])

    # Resample to target Fs (MATLAB: pop_resample)
    from scipy.signal import resample as _resample
    n_new = int(round(d.shape[1] * target_fs / orig_fs))
    d_r = np.zeros((nchan, n_new))
    for j in range(nchan):
        d_r[j] = _resample(d[j], n_new)

    # Convert event latencies to resampled timeline
    codes = np.array([int(id2desc.get(c, 0)) for c in evs[:, 2]])
    lats = np.round(evs[:, 0] * target_fs / orig_fs).astype(int)

    return d_r, codes, lats


def p300_extract_session_letters(codes, lats, EEG, grid, n_letters, fs=P300_FS):
    """Extract letter-level averaged features and single trials from one session.

    For each flash type (row/col) × each letter:
      - 15 single-trial epochs are extracted (150–500 ms, baseline-corrected)
      - Averaged to produce one feature vector per flash per letter
      - Labeled target (2) or non-target (1) from preceding event code

    Args:
        codes: event codes array for this session
        lats:  event latencies for this session (in combined-EEG sample indices)
        EEG:   combined continuous data (nchan, T_total) in µV
        grid:  dict with 'rows' and 'cols' flash code lists
        n_letters: number of letters in this session (5 for QUICK, 4 for JUMP)
        fs:    sampling rate (80 Hz)

    Returns:
        letters: list of n_letters dicts, each with:
            'avg_x':  (T_feat, nchan, n_flash_types) averaged features
            'avg_y':  (n_flash_types,) labels
            'single_x': (T_feat, nchan, n_total_single) all single trials
            'single_y': (n_total_single,) labels
    """
    ts_f = np.arange(round(P300_FEAT_TMIN * fs), round(P300_FEAT_TMAX * fs))
    bl_s = np.arange(round(P300_BL_TMIN * fs), round(P300_BL_TMAX * fs) + 1)
    T_feat = len(ts_f)
    nchan = EEG.shape[0]

    all_flash_codes = grid['rows'] + grid['cols']
    is_row_code = set(grid['rows'])

    # For each flash code, find valid event indices and group by letter
    flash_groups = {}
    for code in all_flash_codes:
        indices = np.where(codes == code)[0]

        # MATLAB filter: rows remove if preceded by 32777, cols if followed by 32778
        if code in is_row_code:
            valid = [i for i in indices if not (i > 0 and codes[i - 1] == 32777)]
        else:
            valid = [i for i in indices
                     if not (i + 1 < len(codes) and codes[i + 1] == 32778)]
        valid = np.array(valid, dtype=int) if valid else np.array([], dtype=int)

        # Group into letters of P300_NREPS
        letter_groups = []
        for li in range(min(n_letters, len(valid) // P300_NREPS)):
            letter_groups.append(valid[P300_NREPS * li: P300_NREPS * (li + 1)])
        flash_groups[code] = letter_groups

    # Build letter-level data
    letters = []
    for li in range(n_letters):
        avg_list, label_list = [], []
        single_x_list, single_y_list = [], []

        for code in all_flash_codes:
            if li >= len(flash_groups[code]):
                continue
            letter_indices = flash_groups[code][li]

            # Extract epochs: MATLAB uses stimes(n+1)+ts_f for epoch, stimes(n)+BLint for baseline
            trials = []
            for fi in letter_indices:
                if fi + 1 >= len(lats):
                    continue
                stim_lat = lats[fi + 1]   # next event latency (stimulus onset)
                bl_lat = lats[fi]          # current event latency (marker)

                ep_idx = stim_lat + ts_f
                bl_idx = bl_lat + bl_s

                if ep_idx.min() < 0 or ep_idx.max() >= EEG.shape[1]:
                    continue
                if bl_idx.min() < 0 or bl_idx.max() >= EEG.shape[1]:
                    continue

                ep = EEG[:, ep_idx]  # (nchan, T_feat)
                baseline = np.mean(EEG[:, bl_idx], axis=1, keepdims=True)
                ep = ep - baseline
                trials.append(ep)

            if len(trials) == 0:
                continue

            trials_arr = np.stack(trials, axis=2)       # (nchan, T_feat, n_reps)
            avg = np.mean(trials_arr, axis=2)            # (nchan, T_feat)

            # Label: mode of preceding event codes (MATLAB: mode(a1(indices-1)))
            preceding = [int(codes[fi - 1]) for fi in letter_indices if fi > 0]
            if preceding:
                label_code = int(stats.mode(preceding, keepdims=False).mode)
                label = 2 if label_code == 33285 else 1
            else:
                label = 1

            avg_list.append(avg.T)  # (T_feat, nchan) — matches CSP format axis order
            label_list.append(label)

            # Collect single trials for GAN training
            for t in range(trials_arr.shape[2]):
                single_x_list.append(trials_arr[:, :, t].T)  # (T_feat, nchan)
                single_y_list.append(label)

        if avg_list:
            letters.append({
                'avg_x': np.stack(avg_list, axis=2),                 # (T_feat, nchan, n_flash)
                'avg_y': np.array(label_list, dtype=np.float32),     # (n_flash,)
                'single_x': np.stack(single_x_list, axis=2) if single_x_list
                            else np.zeros((T_feat, nchan, 0)),       # (T_feat, nchan, N)
                'single_y': np.array(single_y_list, dtype=np.float32) if single_y_list
                            else np.array([]),                        # (N,)
            })

    return letters


# ── P300 xDAWN Spatial Filter (CSP analog) ──────────────────────────────────

def p300_xdawn(data):
    """Compute xDAWN spatial filters for P300 data.
    Maximises evoked-response SNR, analogous to CSP for oscillatory paradigms.

    data: {'x': (T, ch, N), 'y': (N,)} with labels 1=non-target, 2=target.
    Returns: W (ch, ch) matrix, rows sorted by decreasing eigenvalue.
    """
    target_idx = np.where(data['y'] == 2)[0]
    n_ch = data['x'].shape[1]

    if len(target_idx) == 0:
        return np.eye(n_ch)

    # Average target response (evoked potential template)
    target_mean = np.mean(data['x'][:, :, target_idx], axis=2)  # (T, ch)

    # Signal covariance from evoked response
    C_signal = target_mean.T @ target_mean  # (ch, ch)

    # Total data covariance (all trials)
    n_trials = data['x'].shape[2]
    C_total = np.zeros((n_ch, n_ch))
    for i in range(n_trials):
        trial = data['x'][:, :, i]  # (T, ch)
        C_total += trial.T @ trial
    C_total /= n_trials

    # Solve generalised eigenvalue problem: C_signal w = λ C_total w
    epsilon = 1e-9
    C_total_reg = C_total + epsilon * np.eye(n_ch)
    try:
        evals, evecs = eigh(C_signal, C_total_reg)
    except np.linalg.LinAlgError:
        return np.eye(n_ch)

    idx = np.argsort(evals)[::-1]
    W = evecs[:, idx].T
    return W


def p300_feat_xdawn_train(data, n_components=P300_N_XDAWN):
    """xDAWN feature extraction — analog of featcrossval for MI.
    Projects with top xDAWN filters, flattens time × components.

    Returns:
        features: {'x': (n_feat, N), 'y': (N,)}
        selectedw: (n_components, ch) spatial filter matrix
    """
    W = p300_xdawn(data)
    n_comp = min(n_components, W.shape[0])
    selectedw = W[:n_comp, :]

    T, ch, N = data['x'].shape
    n_feat = n_comp * T

    if n_feat == 0 or N == 0:
        return {'x': np.array([]), 'y': data['y']}, selectedw

    features = {'x': np.zeros((n_feat, N)), 'y': data['y']}
    for trial in range(N):
        projected = data['x'][:, :, trial] @ selectedw.T  # (T, n_comp)
        features['x'][:, trial] = projected.flatten()

    return features, selectedw


def p300_feat_xdawn_test(data, selectedw):
    """Apply learned xDAWN to test data — analog of featcrostest for MI."""
    if selectedw is None or selectedw.shape[0] == 0:
        return {'x': np.zeros((0, data['x'].shape[2])), 'y': data['y']}

    T, ch, N = data['x'].shape
    n_comp = selectedw.shape[0]
    n_feat = n_comp * T

    features = {'x': np.zeros((n_feat, N)), 'y': data['y']}
    for trial in range(N):
        projected = data['x'][:, :, trial] @ selectedw.T  # (T, n_comp)
        features['x'][:, trial] = projected.flatten()

    return features


# ── P300 xDAWN-SVM Train / Evaluate (matching CSP-SVM interface) ────────────

def train_p300_xdawnsvm(data):
    """Train xDAWN + linear SVM pipeline — analog of train_cspsvm.
    Returns: (model, spatial_filters) or (None, None)."""
    if data is None or 'x' not in data or data['x'].shape[2] == 0:
        return None, None
    features, W = p300_feat_xdawn_train(data)
    if features['x'].size == 0:
        return None, W
    X = features['x'].T  # (N, n_feat)
    y = features['y']
    if X.shape[0] == 0:
        return None, W
    model = fitcsvm(X, y, Standardize=True, KernelFunction='linear')
    return model, W


def evaluate_p300_xdawnsvm(model, spatial_filters, data_test):
    """Evaluate xDAWN + SVM — analog of evaluate_cspsvm.
    Returns: accuracy (%), confusion_matrix, predictions."""
    if model is None:
        return 0, np.zeros((2, 2)), np.array([])
    if spatial_filters is None or spatial_filters.size == 0:
        return 0, np.zeros((2, 2)), np.array([])
    if data_test is None or 'x' not in data_test or data_test['x'].shape[2] == 0:
        return 0, np.zeros((2, 2)), np.array([])

    features = p300_feat_xdawn_test(data_test, spatial_filters)
    if features['x'].size == 0:
        return 0, np.zeros((2, 2)), np.array([])

    X = features['x'].T
    y_true = features['y']
    if X.shape[0] == 0:
        return 0, np.zeros((2, 2)), np.array([])

    y_pred = predict_svm(model, X)
    accuracy = 0
    cm = np.zeros((2, 2))
    if len(y_true) > 0 and len(y_pred) > 0 and len(y_true) == len(y_pred):
        accuracy = np.mean(y_pred == y_true) * 100
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    return accuracy, cm, y_pred


# ── RS Preprocessing Functions ────────────────────────────────────────────────

def rs_find_marker_column(raw_array, marker_val, n_eeg_channels):
    n_cols = raw_array.shape[1]
    for col in range(n_cols - 1, max(n_cols - 6, n_eeg_channels), -1):
        if marker_val in raw_array[:, col]: return col
    for col in range(n_eeg_channels, n_cols):
        if marker_val in raw_array[:, col]: return col
    return -1


def rs_load_and_filter(filepath, fs=500, duration_s=150, marker_val=5,
                        n_ch=8, lowcut=2.0, highcut=40.0):
    """Load .easy file, slice from marker, bandpass filter."""
    try:
        raw = np.loadtxt(filepath)
    except Exception as e:
        log_message(f"    RS: cannot read {filepath}: {e}")
        return None
    log_message(f"    RS: loaded {os.path.basename(filepath)} shape={raw.shape}")
    if raw.shape[1] <= n_ch:
        log_message(f"    RS: only {raw.shape[1]} columns, need >{n_ch}")
        return None
    mc = rs_find_marker_column(raw, marker_val, n_ch)
    if mc == -1:
        log_message(f"    RS: no marker column with value {marker_val}")
        return None
    starts = np.where(raw[:, mc] == marker_val)[0]
    if len(starts) == 0:
        log_message(f"    RS: marker value {marker_val} not found")
        return None
    si = starts[0]
    n_sam = int(fs * duration_s)
    eeg = raw[si:si+n_sam, :n_ch] if si+n_sam <= len(raw) else raw[si:, :n_ch]
    log_message(f"    RS: marker at sample {si}, sliced {eeg.shape[0]} samples")
    if eeg.shape[0] < int(0.5 * n_sam):
        log_message(f"    RS: too short ({eeg.shape[0]} < {int(0.5*n_sam)})")
        return None
    if np.median(np.abs(eeg)) > 500: eeg = eeg / 1000.0
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, eeg.astype(np.float64), axis=0)


def rs_epoch_and_reject(continuous, label, epoch_s=2.0, stride_s=2.0, fs=500, thresh=150.0):
    """Epoch continuous RS data with global p2p artifact rejection."""
    ep_len = int(epoch_s * fs)
    stride = int(stride_s * fs)
    n_ep = (continuous.shape[0] - ep_len) // stride + 1
    kept = []
    for i in range(n_ep):
        s = i * stride
        ep = continuous[s:s+ep_len, :]
        if (ep.max() - ep.min()) < thresh: kept.append(ep)
    if not kept: return np.empty((ep_len, continuous.shape[1], 0)), np.array([])
    x = np.stack(kept, axis=0).transpose(1, 2, 0).astype(np.float32)
    y = np.full(x.shape[2], label, dtype=np.float32)
    return x, y


# ── Universal PSD Plot ────────────────────────────────────────────────────────

def plot_psd_comparison(x_real_gan, y_real, x_synth_gan, y_synth,
                         fs, subject_id, output_dir, paradigm='MI'):
    """
    Universal PSD plot: real vs synthetic, all channels averaged.
    Works for MI, P300, RS. x in GAN format (trials, ch, T).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    nperseg = min(256, x_real_gan.shape[2])
    for lbl, xd, color in [('Real', x_real_gan, '#2166AC'),
                             ('Synthetic', x_synth_gan, '#B2182B')]:
        all_psd = []
        for t in range(xd.shape[0]):
            ch_psd = []
            for c in range(xd.shape[1]):
                f, p = welch(xd[t, c, :], fs=fs, nperseg=nperseg)
                ch_psd.append(p)
            all_psd.append(np.mean(ch_psd, axis=0))
        psd_mean = 10 * np.log10(np.mean(all_psd, axis=0) + 1e-15)
        mask = f <= 45
        ax.plot(f[mask], psd_mean[mask], color=color, lw=2, label=f'{lbl} ({xd.shape[0]} trials)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title(f'PSD - {paradigm} S{subject_id}')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'psd_S{subject_id}.png'), dpi=150)
    plt.close(fig)


# ── P300 Subject Processing (matching MI orchestration) ─────────────────────

def _p300_avg_groups(sg, nreps=P300_NREPS):
    """Average groups of nreps GAN-format trials → pseudo-averaged features.
    sg: (N, ch, T) → returns (N//nreps, ch, T)."""
    n_groups = sg.shape[0] // nreps
    if n_groups == 0:
        return sg[:1]
    return np.stack([np.mean(sg[g * nreps:(g + 1) * nreps], axis=0)
                     for g in range(n_groups)], axis=0)


def _p300_letters_to_dataset(letter_list):
    """Combine letter dicts into a single CSP-format dataset."""
    xs = [l['avg_x'] for l in letter_list if l['avg_x'].shape[2] > 0]
    ys = [l['avg_y'] for l in letter_list if len(l['avg_y']) > 0]
    if not xs:
        return None
    return {'x': np.concatenate(xs, axis=2), 'y': np.concatenate(ys)}


def _p300_letters_single_trials(letter_list):
    """Combine single trials from letter dicts."""
    xs = [l['single_x'] for l in letter_list
          if isinstance(l['single_x'], np.ndarray) and l['single_x'].size > 0]
    ys = [l['single_y'] for l in letter_list
          if isinstance(l['single_y'], np.ndarray) and l['single_y'].size > 0]
    if not xs:
        return None, None
    return np.concatenate(xs, axis=2), np.concatenate(ys)


def process_p300_subjects(p300_data_root, subject_dirs, target_pairs=None,
                           output_base='results_P300'):
    """Process P300 subjects — letter-based split, 15-rep averaging, xDAWN-SVM.

    Pipeline matching MATLAB Step_1.m + MI orchestration structure:
      1. Load GDF pairs, filter (HP+LP+notch at orig Fs), resample to 80 Hz, CAR
      2. Extract 9 letters (5 QUICK + 4 JUMP) with row/col flash averaging (15 reps)
      3. Split: 4 train, 1 val, 4 test letters
      4. GAN trains on single-trial flash epochs from training letters
      5. Batch selection: generate synthetic → average 15 → xDAWN-SVM on val
      6. Strategy selection: mix ratios (0/25/50/100% real+synth)
      7. Final test with best strategy
    """
    session_ids = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    nchan = P300_NCHAN
    fs = P300_FS
    ts_f = np.arange(round(P300_FEAT_TMIN * fs), round(P300_FEAT_TMAX * fs))
    T_feat = len(ts_f)  # 28 samples at 80 Hz

    log_message(f"P300: target_fs={fs}Hz, T_feat={T_feat}, nreps={P300_NREPS}")

    results = []
    for sd in subject_dirs:
        sp = os.path.join(p300_data_root, sd)
        if not os.path.isdir(sp):
            continue

        for pi, (sa, sb) in enumerate(pairs):
            if target_pairs is not None and pi not in target_pairs:
                continue

            fa = os.path.join(sp, f'{session_ids[sa]}.gdf')
            fb = os.path.join(sp, f'{session_ids[sb]}.gdf')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue

            grid = P300_GRIDS[pi]
            sid = f'{sd}_pair{pi}'
            odir = os.path.join(output_base, f'Subject_{sid}')
            os.makedirs(odir, exist_ok=True)
            set_log_path(os.path.join(odir, 'run_log.txt'))
            log_message(f"\n{'='*60}\n  P300 {sid}\n{'='*60}")

            try:
                # ── Step 1: Load, filter, resample each session ──────────────
                d1, codes1, lats1 = p300_load_filter_resample(fa, pi, nchan, fs)
                d2, codes2, lats2 = p300_load_filter_resample(fb, pi, nchan, fs)

                # Concatenate EEG, offset session-2 latencies
                n1 = d1.shape[1]
                EEG = np.concatenate([d1, d2], axis=1)

                # Common average reference (MATLAB: EEG - mean across channels)
                EEG = EEG - np.mean(EEG, axis=0, keepdims=True)

                lats2_offset = lats2 + n1
                log_message(f"  EEG: {EEG.shape}, events: s1={len(codes1)} s2={len(codes2)}")

                # ── Step 2: Extract letters (QUICK + JUMP) ──────────────────
                letters_q = p300_extract_session_letters(
                    codes1, lats1, EEG, grid, grid['nq'], fs)
                letters_j = p300_extract_session_letters(
                    codes2, lats2_offset, EEG, grid, grid['nj'], fs)

                all_letters = letters_q + letters_j
                n_total = len(all_letters)
                log_message(f"  Letters: {n_total} (QUICK={len(letters_q)}, JUMP={len(letters_j)})")

                if n_total < 9:
                    log_message(f"  Not enough letters ({n_total}<9), skipping {sid}")
                    continue

                # ── Step 3: Split — 4 train / 1 val / 4 test ────────────────
                train_letters = all_letters[:4]
                val_letters = [all_letters[4]]
                test_letters = all_letters[5:9]

                train_avg = _p300_letters_to_dataset(train_letters)
                val_avg = _p300_letters_to_dataset(val_letters)
                test_avg = _p300_letters_to_dataset(test_letters)

                if train_avg is None or val_avg is None or test_avg is None:
                    log_message(f"  Missing averaged data for {sid}")
                    continue

                n_flash = len(grid['rows']) + len(grid['cols'])
                log_message(f"  Grid: {len(grid['rows'])}×{len(grid['cols'])} "
                            f"→ {n_flash} flash types/letter")
                log_message(f"  Train: {train_avg['x'].shape[2]} avg features "
                            f"(T={int(np.sum(train_avg['y']==2))}, "
                            f"NT={int(np.sum(train_avg['y']==1))})")
                log_message(f"  Val:   {val_avg['x'].shape[2]} avg features")
                log_message(f"  Test:  {test_avg['x'].shape[2]} avg features")

                # Single trials for GAN (training letters only)
                single_x, single_y = _p300_letters_single_trials(train_letters)
                if single_x is None:
                    log_message(f"  No single trials for GAN, skipping {sid}")
                    continue

                log_message(f"  GAN data: {single_x.shape[2]} single trials "
                            f"(T={int(np.sum(single_y==2))}, "
                            f"NT={int(np.sum(single_y==1))})")

                # ── Normalize (global min-max → [-1,1], matching MI) ─────────
                g_min = float(np.min(single_x))
                g_max = float(np.max(single_x))
                g_range = g_max - g_min
                if g_range < 1e-8:
                    log_message(f"  Zero range in data, skipping {sid}")
                    continue

                def _norm(arr):
                    return np.clip(2.0 * (arr - g_min) / g_range - 1.0,
                                   -1, 1).astype(np.float32)

                single_x_n = _norm(single_x)
                train_avg_n = {'x': _norm(train_avg['x']), 'y': train_avg['y']}
                val_avg_n = {'x': _norm(val_avg['x']), 'y': val_avg['y']}
                test_avg_n = {'x': _norm(test_avg['x']), 'y': test_avg['y']}

                # GAN format: (trials, ch, T)
                x_gan = single_x_n.transpose(2, 1, 0)
                y_gan = single_y

                # ── Step 4: Baseline (train+val → xDAWN-SVM → test) ─────────
                tv_avg = {
                    'x': np.concatenate([train_avg_n['x'], val_avg_n['x']], axis=2),
                    'y': np.concatenate([train_avg_n['y'], val_avg_n['y']])
                }
                model_bl, sf_bl = train_p300_xdawnsvm(tv_avg)
                baseline_acc, _, _ = evaluate_p300_xdawnsvm(model_bl, sf_bl, test_avg_n)
                log_message(f"  Baseline: {baseline_acc:.2f}%")

                # ── Step 5: GAN Training + Batch Selection ───────────────────
                log_message(f"\n--- Step 5: {NUM_RUNS_PER_SUBJECT} GAN run(s) ---")
                all_batches = []
                # Number of averaged features to generate per class
                n_avg_per_class = max(train_avg_n['x'].shape[2] // 2, 5)
                n_single_per_class = n_avg_per_class * P300_NREPS

                for run_idx in range(NUM_RUNS_PER_SUBJECT):
                    run_num = run_idx + 1
                    gen = train_cgan(
                        eeg_data=x_gan, eeg_labels=y_gan,
                        epochs=GAN_EPOCHS,
                        batch_size=min(64, x_gan.shape[0]),
                        num_gan_channels=nchan,
                        output_dir=odir, subject_id=sid, run_number=run_num,
                        time_points=T_feat, freq_loss_weight=1.5)
                    if gen is None:
                        continue

                    log_message(f"  Generating batches (Run {run_num})...")
                    for bi in range(30):
                        # Generate single trials, then average groups of 15
                        sc0 = generate_synthetic(gen, n_single_per_class, 0)
                        sc1 = generate_synthetic(gen, n_single_per_class, 1)
                        if not (sc0.size > 0 and sc1.size > 0):
                            continue

                        sc0_avg = _p300_avg_groups(sc0)  # (N0, ch, T)
                        sc1_avg = _p300_avg_groups(sc1)  # (N1, ch, T)

                        sg = np.concatenate((sc0_avg, sc1_avg), axis=0)
                        sy = np.concatenate((np.ones(sc0_avg.shape[0]),
                                             np.ones(sc1_avg.shape[0]) * 2))
                        # To CSP format: (T, ch, N)
                        sd = {'x': np.transpose(sg, (2, 1, 0)), 'y': sy}

                        m_s, sf_s = train_p300_xdawnsvm(sd)
                        ba = 0.0
                        if m_s:
                            ba, _, _ = evaluate_p300_xdawnsvm(m_s, sf_s, val_avg_n)
                        log_message(f"    R{run_num} B{bi+1}: val_acc={ba:.1f}%")

                        all_batches.append({
                            'run': run_num, 'batch': bi + 1, 'val_acc': ba,
                            'synth_csp': sd, 'synth_gan': sg, 'synth_y': sy})

                if not all_batches:
                    log_message("  CRITICAL: No batches generated")
                    continue

                # ── Step 6: Strategy Selection (matching MI) ─────────────────
                log_message("\n--- Step 6: Strategy Selection ---")
                all_batches.sort(key=lambda b: b['val_acc'], reverse=True)
                top10 = all_batches[:10]

                mix_ratios = [0, 0.25, 0.50, 1.0]
                all_strats = []
                for b in top10:
                    for r in mix_ratios:
                        if r == 0:
                            sd = b['synth_csp']
                            name = 'SynthOnly'
                        else:
                            name = f'Aug{int(r * 100)}pct'
                            nr = train_avg_n['x'].shape[2]
                            na = int(nr * r)
                            na = min(na, b['synth_csp']['x'].shape[2])
                            sx = b['synth_csp']['x'][:, :, :na]
                            sy_s = b['synth_csp']['y'][:na]
                            sd = {
                                'x': np.concatenate((train_avg_n['x'], sx), axis=2),
                                'y': np.concatenate((train_avg_n['y'], sy_s))
                            }
                        m, sf = train_p300_xdawnsvm(sd)
                        va = 0.0
                        if m:
                            va, _, _ = evaluate_p300_xdawnsvm(m, sf, val_avg_n)
                        all_strats.append({
                            'name': name, 'val_acc': va,
                            'batch_info': b, 'mix_ratio': r})

                if not all_strats:
                    log_message("  CRITICAL: No strategies")
                    continue

                all_strats.sort(key=lambda s: s['val_acc'], reverse=True)
                for s in all_strats[:8]:
                    log_message(f"  {s['name']:15s} val={s['val_acc']:.2f}%")
                best = all_strats[0]
                log_message(f"\n  SELECTED: {best['name']} val={best['val_acc']:.2f}%")

                # ── Step 7: Final Test Evaluation ────────────────────────────
                log_message("\n--- Step 7: Final Test ---")

                # Synth-only test accuracy
                so_strats = [s for s in all_strats if s['mix_ratio'] == 0]
                so_acc = 0.0
                if so_strats:
                    best_so = max(so_strats, key=lambda s: s['val_acc'])
                    m_so, sf_so = train_p300_xdawnsvm(
                        best_so['batch_info']['synth_csp'])
                    if m_so:
                        so_acc, _, _ = evaluate_p300_xdawnsvm(
                            m_so, sf_so, test_avg_n)
                log_message(f"  Synth-Only -> Test: {so_acc:.2f}%")

                # Best strategy: retrain on train+val + synth
                if best['mix_ratio'] == 0:
                    final_data = best['batch_info']['synth_csp']
                else:
                    sf_d = best['batch_info']['synth_csp']
                    nr = tv_avg['x'].shape[2]
                    na = int(nr * best['mix_ratio'])
                    if na > sf_d['x'].shape[2]:
                        na = sf_d['x'].shape[2]
                    final_data = {
                        'x': np.concatenate(
                            (tv_avg['x'], sf_d['x'][:, :, :na]), axis=2),
                        'y': np.concatenate(
                            (tv_avg['y'], sf_d['y'][:na]))
                    }
                m_f, sf_f = train_p300_xdawnsvm(final_data)
                best_acc, cm, _ = evaluate_p300_xdawnsvm(
                    m_f, sf_f, test_avg_n) if m_f else (0, np.zeros((2, 2)), None)
                log_message(f"  Best({best['name']}) -> Test: {best_acc:.2f}%\n{cm}")

                plot_acc(baseline_acc, so_acc, best_acc, sid, odir)

                # PSD comparison (GAN single trials vs real single trials)
                best_sg = best['batch_info']['synth_gan']
                best_sy = best['batch_info']['synth_y']
                plot_psd_comparison(x_gan, y_gan, best_sg, best_sy,
                                     fs, sid, odir, 'P300')

                # Save synthetic data + normalization stats
                scipy.io.savemat(
                    os.path.join(odir, f'synthetic_S{sid}.mat'),
                    {'synthetic_x': best_sg, 'synthetic_y': best_sy,
                     'norm_min': g_min, 'norm_max': g_max})

                log_message(f"\n  === SUMMARY {sid} ===")
                log_message(f"  Baseline={baseline_acc:.2f}%  "
                            f"SynthOnly={so_acc:.2f}%  "
                            f"Best({best['name']})={best_acc:.2f}%")
                results.append({
                    'sid': sid, 'baseline': baseline_acc,
                    'synth_only': so_acc, 'best_aug': best_acc,
                    'strategy': best['name']})

            except Exception as e:
                log_message(f"  ERROR {sid}: {e}\n{traceback.format_exc()}")

    return results


# ── RS Subject Processing ─────────────────────────────────────────────────────

def process_rs_subjects(rs_data_root, groups, session='pre',
                         num_train=50, output_base='results_RS'):
    """Process RS subjects using the same GAN + batch selection pipeline."""
    op_pat = 'rs1o' if session == 'pre' else 'rs2o'
    cl_pat = 'rs1c' if session == 'pre' else 'rs2c'
    fs_native = 500; decimate_factor = 4; gan_fs = fs_native // decimate_factor
    results = []

    for group in groups:
        gdir = os.path.join(rs_data_root, group)
        if not os.path.isdir(gdir): continue
        sub_dirs = sorted(d for d in glob.glob(os.path.join(gdir, '*')) if os.path.isdir(d))

        for si, sd in enumerate(sub_dirs):
            sid = f'{group}_S{si+1}_{session}'
            odir = os.path.join(output_base, session, f'Subject_{sid}')
            os.makedirs(odir, exist_ok=True)
            set_log_path(os.path.join(odir, 'run_log.txt'))
            log_message(f"\n{'='*60}\n  RS {sid}\n{'='*60}")

            try:
                files = glob.glob(os.path.join(sd, '*.easy'))
                log_message(f"  Found {len(files)} .easy files in {os.path.basename(sd)}")
                log_message(f"  Looking for patterns: open='{op_pat}', closed='{cl_pat}'")
                xo_list, xc_list = [], []
                for fp in files:
                    fn = os.path.basename(fp).lower()
                    if op_pat in fn: cond, label = 'open', 1
                    elif cl_pat in fn: cond, label = 'closed', 2
                    else:
                        log_message(f"    Skipped (no pattern match): {fn}")
                        continue
                    log_message(f"    Loading {fn} -> {cond}")
                    cont = rs_load_and_filter(fp, fs=fs_native)
                    if cont is None: continue
                    xe, ye = rs_epoch_and_reject(cont, label, fs=fs_native)
                    if xe.shape[2] == 0: continue
                    (xo_list if cond == 'open' else xc_list).append((xe, ye))

                if not xo_list or not xc_list:
                    log_message(f"  Missing condition for {sid}"); continue

                def _cat(lst):
                    return np.concatenate([t[0] for t in lst], axis=2), np.concatenate([t[1] for t in lst])
                xo, yo = _cat(xo_list); xc, yc = _cat(xc_list)
                x = np.concatenate([xo, xc], axis=2)
                y_all = np.concatenate([yo, yc])

                # Decimate
                x = _decimate_trials(x, decimate_factor)
                log_message(f"  Decimated to {gan_fs}Hz: x={x.shape}")

                # Split
                idx1 = np.where(y_all == 1)[0]; idx2 = np.where(y_all == 2)[0]
                np.random.shuffle(idx1); np.random.shuffle(idx2)
                nt = min(num_train, len(idx1), len(idx2))
                nv = min(10, len(idx1)-nt, len(idx2)-nt)
                tr_idx = np.concatenate([idx1[:nt], idx2[:nt]])
                va_idx = np.concatenate([idx1[nt:nt+nv], idx2[nt:nt+nv]])
                te_idx = np.concatenate([idx1[nt+nv:], idx2[nt+nv:]])

                # Normalize (minmax for RS — preserves CSP ratios)
                _, ns = normalize_global_minmax(x[:, :, tr_idx])
                x_n, _ = normalize_global_minmax(x, norm_stats=ns)

                x_gan = x_n[:, :, tr_idx].transpose(2, 1, 0)
                y_tr = y_all[tr_idx]
                train_d = {'x': x_n[:, :, tr_idx], 'y': y_all[tr_idx]}
                val_d = {'x': x_n[:, :, va_idx], 'y': y_all[va_idx]}
                test_d = {'x': x_n[:, :, te_idx], 'y': y_all[te_idx]}

                # Baseline (CSP-SVM same as MI)
                tv_d = {'x': np.concatenate([train_d['x'], val_d['x']], axis=2),
                         'y': np.concatenate([train_d['y'], val_d['y']])}
                m_bl, sf_bl = train_cspsvm(tv_d)
                bl_acc, _, _ = evaluate_cspsvm(m_bl, sf_bl, test_d)
                log_message(f"  Baseline: {bl_acc:.2f}%")

                # GAN (same architecture)
                T = x_gan.shape[2]
                n_ch = x_gan.shape[1]
                for run in range(NUM_RUNS_PER_SUBJECT):
                    gen = train_cgan(x_gan, y_tr, epochs=GAN_EPOCHS,
                                     batch_size=min(64, x_gan.shape[0]),
                                     num_gan_channels=n_ch,
                                     output_dir=odir, subject_id=sid, run_number=run+1,
                                     time_points=T, freq_loss_weight=1.5)
                    if gen is None: continue

                    npc = val_d['x'].shape[2] // 2
                    batches = []
                    for bi in range(30):
                        sc0 = generate_synthetic(gen, npc, 0)
                        sc1 = generate_synthetic(gen, npc, 1)
                        if not (sc0.size > 0 and sc1.size > 0): continue
                        sg = np.concatenate((sc0, sc1), axis=0)
                        sy = np.concatenate((np.ones(sc0.shape[0]), np.ones(sc1.shape[0]) * 2))
                        sd_csp = {'x': np.transpose(sg, (2, 1, 0)), 'y': sy}
                        m_s, sf_s = train_cspsvm(sd_csp)
                        ba = 0.0
                        if m_s: ba, _, _ = evaluate_cspsvm(m_s, sf_s, val_d)
                        batches.append({'val_acc': ba, 'synth_csp': sd_csp,
                                         'synth_gan': sg, 'synth_y': sy})

                if not batches: continue
                batches.sort(key=lambda b: b['val_acc'], reverse=True)

                # Strategy (same as MI)
                best_strat = None; best_va = -1
                for b in batches[:10]:
                    for r in [0, 0.25, 0.50, 1.0]:
                        if r == 0: sd2 = b['synth_csp']
                        else:
                            nr = train_d['x'].shape[2]; na = int(nr*r)
                            sd2 = {'x': np.concatenate([train_d['x'], b['synth_csp']['x'][:,:,:na]], axis=2),
                                   'y': np.concatenate([train_d['y'], b['synth_csp']['y'][:na]])}
                        m, sf = train_cspsvm(sd2)
                        va, _, _ = evaluate_cspsvm(m, sf, val_d) if m else (0, None, None)
                        if va > best_va: best_va = va; best_strat = {'ratio': r, 'batch': b}

                if best_strat['ratio'] == 0: final_d = best_strat['batch']['synth_csp']
                else:
                    sf = best_strat['batch']['synth_csp']
                    nr = tv_d['x'].shape[2]; na = int(nr*best_strat['ratio'])
                    final_d = {'x': np.concatenate([tv_d['x'], sf['x'][:,:,:na]], axis=2),
                               'y': np.concatenate([tv_d['y'], sf['y'][:na]])}
                m_f, sf_f = train_cspsvm(final_d)
                fa, _, _ = evaluate_cspsvm(m_f, sf_f, test_d) if m_f else (0, None, None)

                so_b = max(batches, key=lambda b: b['val_acc'])
                m_so, sf_so = train_cspsvm(so_b['synth_csp'])
                so_acc, _, _ = evaluate_cspsvm(m_so, sf_so, test_d) if m_so else (0, None, None)

                log_message(f"  SynthOnly={so_acc:.2f}% Best={fa:.2f}%")
                plot_acc(bl_acc, so_acc, fa, sid, odir)
                plot_psd_comparison(x_gan, y_tr, best_strat['batch']['synth_gan'],
                                     best_strat['batch']['synth_y'],
                                     gan_fs, sid, odir, 'RS')
                results.append({'sid': sid, 'baseline': bl_acc, 'synth_only': so_acc, 'best': fa})

            except Exception as e:
                log_message(f"  ERROR {sid}: {e}\n{traceback.format_exc()}")

    return results


# =============================================================================
# 8.  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    PARADIGM = 'P300'  # 'MI', 'P300', or 'RS'

    if PARADIGM == 'MI':
        data_train = scipy.io.loadmat('data1.mat')
        data_valtest = scipy.io.loadmat('data2.mat')
        xsubi_train = data_train['xsubi_all']
        xsubi_valtest = data_valtest['txsubi_all']
        n_subjects = xsubi_train.shape[1]

        TARGET_SUBJECTS = "all"
        if isinstance(TARGET_SUBJECTS, str) and TARGET_SUBJECTS.lower() == "all":
            subjects = list(range(n_subjects))
        elif isinstance(TARGET_SUBJECTS, int):
            subjects = [TARGET_SUBJECTS - 1]
        elif isinstance(TARGET_SUBJECTS, list):
            subjects = [s - 1 for s in TARGET_SUBJECTS if 1 <= s <= n_subjects]
        else:
            subjects = list(range(n_subjects))

        for si in subjects:
            odir = f'results_MI/Subject_{si+1}'
            process_subject(si, xsubi_train[0, si], xsubi_valtest[0, si], odir)

    elif PARADIGM == 'P300':
        process_p300_subjects(
            p300_data_root='.',
            subject_dirs=[f'H{i}' for i in range(1, 13)],
            target_pairs=[3],         # which pair(s) to process; None = all
            output_base='results_P300'
        )

    elif PARADIGM == 'RS':
        for session in ['pre', 'post']:
            log_message(f"\n{'#'*60}\n  RS SESSION: {session.upper()}\n{'#'*60}")
            process_rs_subjects(
                rs_data_root='.',
                groups=['Young', 'Elderly'],
                session=session,
                output_base='results_RS'
            )