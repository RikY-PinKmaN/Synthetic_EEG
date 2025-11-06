# --- START OF FILE MAIN.py ---

# =================================================================================================
# main.py - v6 (Looping through datasets)
#
# This script integrates the advanced GAN and evaluation framework with the data processing
# and splitting methodology from the baseline script.
#
# KEY CHANGES:
# 1. MULTIPLE DATA FILES: Loops through 'DATA1.mat' to 'DATA5.mat'.
# 2. FIXED SEQUENTIAL SPLIT:
#    - For each subject, the data is split sequentially:
#      - First 40 trials/class -> Training Set
#      - Next 20 trials/class -> Validation Set
#      - Remaining trials -> Test Set
#    - The random split variation loop has been removed.
# 3. "TOP 10" EVALUATION:
#    - GANs are trained on the Training set.
#    - All generated batches are evaluated on the Validation set.
#    - The top 10 batches are selected and then evaluated on the Test set.
# 4. ISOLATED RESULTS:
#    - Results for each dataset are stored in separate directories (e.g., DATA1_results/, DATA2_results/).
# =================================================================================================

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt, welch
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import traceback
from sklearn.svm import SVC
from scipy.linalg import eigh
import scipy.stats as stats
from scipy.fft import fft
from scipy.stats import wasserstein_distance
import os
import csv

# --- Baseline Processing & Splitting Parameters ---
LOWCUT_BASELINE = 8
HIGHCUT_BASELINE = 30
FS_BASELINE = 512
TIME_TRIM_SECONDS_BASELINE = 0.5
NUM_TRAIN_TRIALS_PER_CLASS_BASELINE = 40 # Number of trials per class for the training split
NUM_VALID_TRIALS_PER_CLASS_BASELINE = 10 # Number of trials per class for the validation split

# --- Define Selected Channels (1-based) from baseline.py ---
SELECTED_CHANNELS_1_BASED = [14, 13, 12, 48, 49, 50, 51, 17, 18, 19, 56, 54, 55]
# Convert to 0-based indices for Python
SELECTED_CHANNELS_0_BASED = [idx - 1 for idx in SELECTED_CHANNELS_1_BASED]
NUM_SELECTED_CHANNELS = len(SELECTED_CHANNELS_0_BASED)

# NUM_PREPROCESSING_VARIATIONS is no longer needed as the split is deterministic
NUM_RUNS_PER_SUBJECT = 5 # Number of GAN training runs per subject

# --- cGAN Specific Constants ---
NUM_CLASSES_CGAN = 2
LATENT_DIM_CGAN = 100
EMBEDDING_DIM_CGAN = 25
R_TRIALS = NUM_TRAIN_TRIALS_PER_CLASS_BASELINE

# --- Set Random Seed ---
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Global variable for log file path
current_log_file_path = None

def log_message(message):
    """Prints to console and appends to the current subject's log file."""
    print(message)
    global current_log_file_path
    if current_log_file_path:
        with open(current_log_file_path, 'a') as f:
            f.write(str(message) + "\n")
    else:
        print(f"WARNING: current_log_file_path not set. Message not logged: {message}")

def butterworth_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# --- Robust Data Extraction Helper ---
def _get_eeg_data_from_field_main(subject_struct_item, field_name, subject_id_str):
    if not hasattr(subject_struct_item, 'dtype') or subject_struct_item.dtype.names is None:
        log_message(f"  ERROR S{subject_id_str}: Input is not a structured array. Cannot find field '{field_name}'.")
        return None
    if field_name not in subject_struct_item.dtype.names:
        log_message(f"  INFO S{subject_id_str}: Field '{field_name}' not found.")
        return None
    field_content = subject_struct_item[field_name]
    actual_data = field_content[0,0] if isinstance(field_content, np.ndarray) and field_content.shape == (1,1) else field_content

    if not isinstance(actual_data, np.ndarray):
        log_message(f"  ERROR S{subject_id_str}: Field '{field_name}' content is not a NumPy array. Type: {type(actual_data)}")
        return None
    if field_name == 'y':
        return actual_data.flatten()
    if actual_data.ndim != 3:
        log_message(f"  ERROR S{subject_id_str}: Field 'x' data is not 3D. Shape: {actual_data.shape}")
        return None
    return actual_data

# <<< MODIFIED >>>: Function now performs a FIXED, SEQUENTIAL 3-way split.
def preprocess_and_split_subject_data(raw_subject_struct, subject_id_str, dataset_name,
                                     selected_channels_0_based,
                                     num_train_trials_per_class,
                                     num_valid_trials_per_class):
    """
    Loads, preprocesses, and splits data for a single subject into training,
    validation, and test sets using a fixed, sequential method.
    """
    try:
        log_message(f"  Preprocessing and splitting data for S{subject_id_str} from {dataset_name}...")
        all_x_data_orig_ch = _get_eeg_data_from_field_main(raw_subject_struct, 'x', subject_id_str)
        all_y_labels_orig = _get_eeg_data_from_field_main(raw_subject_struct, 'y', subject_id_str)
        if all_x_data_orig_ch is None or all_y_labels_orig is None:
            return (None,) * 8

        # 1. Butterworth Filter
        filtered_x = butterworth_filter(all_x_data_orig_ch, LOWCUT_BASELINE, HIGHCUT_BASELINE, FS_BASELINE)

        # 2. Time Trimming
        if filtered_x.shape[0] <= int(FS_BASELINE * TIME_TRIM_SECONDS_BASELINE * 2):
            log_message(f"  ERROR S{subject_id_str}: Trial length too short for trimming. Skipping.")
            return (None,) * 8
        start_sample = int(FS_BASELINE * TIME_TRIM_SECONDS_BASELINE)
        end_sample = -start_sample if start_sample > 0 else filtered_x.shape[0]
        processed_x = filtered_x[start_sample:end_sample, :, :]

        # 3. Channel Selection
        processed_x_sel_ch = processed_x[:, selected_channels_0_based, :]

        # 4. Train/Validation/Test Split (FIXED, SEQUENTIAL)
        idx_c1 = np.where(all_y_labels_orig == 1)[0]
        idx_c2 = np.where(all_y_labels_orig == 2)[0]
        
        total_needed = num_train_trials_per_class + num_valid_trials_per_class
        if len(idx_c1) < total_needed or len(idx_c2) < total_needed:
            log_message(f"  ERROR S{subject_id_str}: Not enough trials per class for train/validation split.")
            return (None,) * 8

        # <<< MODIFIED >>>: Removed np.random.shuffle to ensure sequential splitting
        # np.random.shuffle(idx_c1)
        # np.random.shuffle(idx_c2)

        train_idx = np.concatenate((idx_c1[:num_train_trials_per_class], idx_c2[:num_train_trials_per_class]))
        valid_idx = np.concatenate((idx_c1[num_train_trials_per_class:total_needed], idx_c2[num_train_trials_per_class:total_needed]))
        
        # All remaining trials are for testing
        test_idx = np.concatenate((idx_c1[total_needed:], idx_c2[total_needed:]))

        train_x_unnormalized = processed_x_sel_ch[:, :, train_idx]
        train_y = all_y_labels_orig[train_idx]
        valid_x_unnormalized = processed_x_sel_ch[:, :, valid_idx]
        valid_y = all_y_labels_orig[valid_idx]
        test_x_unnormalized = processed_x_sel_ch[:, :, test_idx]
        test_y = all_y_labels_orig[test_idx]

        # 5. Normalization (based on Training set ONLY)
        train_min = np.min(train_x_unnormalized)
        train_max = np.max(train_x_unnormalized)
        train_range = train_max - train_min
        epsilon_norm = 1e-8

        if train_range < epsilon_norm:
            normalized_train_x = np.zeros_like(train_x_unnormalized)
            normalized_valid_x = np.zeros_like(valid_x_unnormalized)
            normalized_test_x = np.zeros_like(test_x_unnormalized)
        else:
            normalized_train_x = 2 * (train_x_unnormalized - train_min) / train_range - 1
            normalized_valid_x = 2 * (valid_x_unnormalized - train_min) / train_range - 1
            normalized_test_x = 2 * (test_x_unnormalized - train_min) / train_range - 1
            normalized_valid_x = np.clip(normalized_valid_x, -1, 1)
            normalized_test_x = np.clip(normalized_test_x, -1, 1)

        # 6. Format data for return
        x_train_gan = np.transpose(normalized_train_x, (2, 1, 0))
        y_train_gan = train_y
        finaltrn_csp = {'x': normalized_train_x, 'y': train_y}
        finalval_csp = {'x': normalized_valid_x, 'y': valid_y}
        finaltest_csp = {'x': normalized_test_x, 'y': test_y}

        log_message(f"  S{subject_id_str}: Split complete. Train: {x_train_gan.shape}, Validation: {finalval_csp['x'].shape}, Test: {finaltest_csp['x'].shape}")
        return x_train_gan, y_train_gan, finaltrn_csp, finalval_csp, finaltest_csp, train_min, train_max, R_TRIALS
    except Exception as e:
        log_message(f"ERROR S{subject_id_str} in preprocess_and_split_subject_data: {e}\n{traceback.format_exc()}")
        return (None,) * 8


# --- CSP-SVM Functions (Unchanged) ---
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
            valid_trials_c1 +=1
    if valid_trials_c1 > 0: cov1 /= valid_trials_c1
    else: log_message("CSP Warning: All trials for class 1 had NaN covariance.")

    cov2 = np.zeros((n_channels, n_channels))
    valid_trials_c2 = 0
    for trial in range(X2.shape[2]):
        current_trial_data = X2[:,:,trial]
        cov_trial = np.cov(current_trial_data, rowvar=False)
        if not np.all(np.isnan(cov_trial)):
            cov2 += cov_trial
            valid_trials_c2 += 1
    if valid_trials_c2 > 0: cov2 /= valid_trials_c2
    else: log_message("CSP Warning: All trials for class 2 had NaN covariance.")

    epsilon_reg = 1e-9
    cov1_reg = cov1 + epsilon_reg * np.eye(n_channels)
    cov2_reg = cov2 + epsilon_reg * np.eye(n_channels)
    try:
        evals, evecs = eigh(cov1_reg, cov1_reg + cov2_reg)
    except np.linalg.LinAlgError:
        log_message("CSP Error: Generalized eigenvalue problem failed. Using regularized fallback.")
        evals, evecs = eigh(cov1_reg)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    W = evecs.T
    return W

def featcrossval(finaldataset, ChRanking, numchannel):
    a = finaldataset
    W = csp(a)
    if numchannel > 6:
        if W.shape[0] < 6 : selectedw = W
        else: selectedw = np.vstack((W[0:3, :], W[-3:, :]))
    elif W.shape[0] >=2 : selectedw = np.vstack((W[0, :][np.newaxis,:], W[-1, :][np.newaxis,:]))
    elif W.shape[0] == 1: selectedw = W
    else:
        log_message("featcrossval Warning: CSP generated 0 filters.")
        return {'x': np.array([]), 'y': finaldataset['y']}, np.array([])

    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]
    if num_features == 0: return {'x': np.zeros((0, ntrial)), 'y': finaldataset['y']}, selectedw

    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}
    for trial in range(ntrial):
        projected_trial_data = finaldataset['x'][:, :, trial]
        selectedZ = np.dot(projected_trial_data, selectedw.T)
        variances = np.var(selectedZ, axis=0)
        epsilon_var = 1e-9
        variances_reg = variances + epsilon_var
        sum_variances_reg = np.sum(variances_reg)
        if sum_variances_reg < epsilon_var : producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_variances_reg)
    return producedfeatur, selectedw

def featcrostest(finaldataset, ChRanking, numchannel, selectedw):
    a = finaldataset
    if selectedw is None or selectedw.shape[0] == 0:
        log_message("featcrostest Warning: No spatial filters provided.")
        return {'x': np.zeros((0, a['x'].shape[2] if a['x'].ndim == 3 and a['x'].shape[2] > 0 else 0)),
                'y': finaldataset['y']}
    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]
    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}
    for trial in range(ntrial):
        projected_trial_data = finaldataset['x'][:, :, trial]
        selectedZ = np.dot(projected_trial_data, selectedw.T)
        variances = np.var(selectedZ, axis=0)
        epsilon_var = 1e-9
        variances_reg = variances + epsilon_var
        sum_variances_reg = np.sum(variances_reg)
        if sum_variances_reg < epsilon_var: producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_variances_reg)
    return producedfeatur

def fitcsvm(X, Y, **kwargs):
    standardize = kwargs.get('Standardize', False)
    kernel = kwargs.get('KernelFunction', 'linear')
    kernel_map = {'linear': 'linear', 'rbf': 'rbf', 'gaussian': 'rbf', 'polynomial': 'poly'}
    model = SVC(kernel=kernel_map.get(kernel, kernel), probability=True, C=1.0)
    if X.shape[0] == 0:
        log_message("fitcsvm Error: Input X has 0 trials.")
        return None
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
        y_pred = model.predict(X_norm)
    else:
        y_pred = model.predict(X)
    return y_pred

def train_cspsvm(data):
    if data is None or 'x' not in data or data['x'].shape[2] == 0:
        return None, None
    X_data_csp = data['x']
    n_selected_channels_csp = X_data_csp.shape[1]
    features_dict, spatial_filters = featcrossval(data, ChRanking=None, numchannel=n_selected_channels_csp)
    if features_dict['x'].size == 0 : return None, spatial_filters
    X_features_train = features_dict['x'].T
    y_labels_train = features_dict['y']
    if X_features_train.shape[0] == 0: return None, spatial_filters
    model = fitcsvm(X_features_train, y_labels_train, Standardize=True, KernelFunction='linear')
    return model, spatial_filters

def evaluate_cspsvm(model, spatial_filters, data_eval):
    if model is None: return 0, np.zeros((2,2)), np.array([])
    if spatial_filters is None or spatial_filters.size == 0: return 0, np.zeros((2,2)), np.array([])
    if data_eval is None or 'x' not in data_eval or data_eval['x'].shape[2] == 0:
        return 0, np.zeros((2,2)), np.array([])
    X_data_eval_csp = data_eval['x']
    n_selected_channels_eval = X_data_eval_csp.shape[1]
    features_eval_dict = featcrostest(data_eval, ChRanking=None, numchannel=n_selected_channels_eval, selectedw=spatial_filters)
    if features_eval_dict['x'].size == 0:
        y_true = data_eval['y']
        cm = confusion_matrix(y_true, [], labels=[1,2]) if len(y_true) > 0 else np.zeros((2,2))
        return 0, cm, np.array([])
    X_features_eval = features_eval_dict['x'].T
    y_true = features_eval_dict['y']
    if X_features_eval.shape[0] == 0:
        cm = confusion_matrix(y_true, [], labels=[1,2]) if len(y_true) > 0 else np.zeros((2,2))
        return 0, cm, np.array([])
    y_pred = predict_svm(model, X_features_eval)
    accuracy = 0; cm = np.zeros((2,2))
    if len(y_true) > 0 and len(y_pred) > 0 and len(y_true) == len(y_pred):
        accuracy = np.mean(y_pred == y_true) * 100
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    elif len(y_true) > 0:
        cm = confusion_matrix(y_true, [], labels=[1,2])
    return accuracy, cm, y_pred


def compute_average_psd(data, labels, class_labels=[1, 2], sampling_rate=512):
    """Computes the average PSD for each class."""
    psd_results = {}
    for class_label in class_labels:
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) == 0:
            continue
        
        class_data = data[:, :, class_indices] # Shape (samples, channels, trials)
        
        # Transpose to (trials, channels, samples) for easier iteration
        class_data_transposed = class_data.transpose(2, 1, 0)
        
        psds = []
        for trial_data in class_data_transposed:
            # trial_data shape is (channels, samples)
            freqs, trial_psd = welch(trial_data, fs=sampling_rate, nperseg=min(256, trial_data.shape[1]))
            psds.append(np.mean(trial_psd, axis=0)) # Average across channels
        
        if psds:
            psd_results[f"class_{class_label}"] = (freqs, np.mean(psds, axis=0))
            
    return psd_results

# --- Conditional GAN (cGAN) Functions (Unchanged) ---
def build_generator_cgan(num_channels, time_points, latent_dim=LATENT_DIM_CGAN, num_classes=NUM_CLASSES_CGAN, embedding_dim=EMBEDDING_DIM_CGAN):
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
    return models.Model([noise_input, label_input], x, name="generator_cgan")

def build_critic_cgan(num_channels, time_points, num_classes=NUM_CLASSES_CGAN, embedding_dim=EMBEDDING_DIM_CGAN):
    data_input = layers.Input(shape=(num_channels, time_points))
    label_input = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(num_classes, embedding_dim)(label_input)
    x = layers.Permute((2, 1))(data_input)
    x = layers.Conv1D(64, 5, 2, "same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1D(128, 5, 2, "same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1D(256, 5, 2, "same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    data_features_flat = layers.Flatten()(x)
    label_embedding_flat = layers.Flatten()(label_embedding)
    merged_features = layers.Concatenate()([data_features_flat, label_embedding_flat])
    output_score = layers.Dense(1)(merged_features)
    return models.Model([data_input, label_input], output_score, name="critic_cgan")

def frequency_domain_loss(real_data, generated_data):
    real_data_t = tf.transpose(real_data, perm=[0, 2, 1])
    gen_data_t = tf.transpose(generated_data, perm=[0, 2, 1])
    real_fft = tf.abs(tf.signal.rfft(real_data_t))
    gen_fft = tf.abs(tf.signal.rfft(gen_data_t))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))

def smooth_eeg(data, window_size=5):
    smoothed_data = np.copy(data)
    if window_size % 2 == 0: window_size += 1
    if data.shape[2] <= window_size :
        log_message(f"smooth_eeg: window_size {window_size} >= data length {data.shape[2]}. Skipping.")
        return data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            smoothed_data[i, j, :] = signal.savgol_filter(data[i, j, :], window_size, 2)
    return smoothed_data

def gradient_penalty_cgan(critic, real_samples_eeg, fake_samples_eeg, real_labels_batch_0_1, lambda_gp=10):
    alpha = tf.random.uniform(shape=[tf.shape(real_samples_eeg)[0], 1, 1], minval=0., maxval=1.)
    interpolated_eeg = alpha * real_samples_eeg + (1 - alpha) * fake_samples_eeg
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_eeg)
        interpolated_predictions = critic([interpolated_eeg, real_labels_batch_0_1], training=True)
    gradients = gp_tape.gradient(interpolated_predictions, [interpolated_eeg])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
    gp_loss_val = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return lambda_gp * gp_loss_val

def train_wgan_gp_cgan(eeg_data, eeg_labels, epochs, batch_size, model_name_prefix, num_gan_channels,
                       output_dir_param, subject_display_idx_param, run_number,
                       time_points=500, latent_dim=LATENT_DIM_CGAN, n_critic_steps=5, freq_loss_weight=0.5,
                       num_classes_gan=NUM_CLASSES_CGAN, embedding_dim_gan=EMBEDDING_DIM_CGAN):
    if not isinstance(eeg_labels, np.ndarray): eeg_labels = np.array(eeg_labels)
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)
    if not (np.all(labels_0_1 >= 0) and np.all(labels_0_1 < num_classes_gan)):
        log_message(f"ERROR cGAN S{subject_display_idx_param} Run {run_number}: Labels not 0-{num_classes_gan-1}. Found: {np.unique(labels_0_1)}")
        return None
    labels_0_1_reshaped = labels_0_1[:, np.newaxis]

    generator = build_generator_cgan(num_channels=num_gan_channels, time_points=time_points, latent_dim=latent_dim, num_classes=num_classes_gan, embedding_dim=embedding_dim_gan)
    critic = build_critic_cgan(num_channels=num_gan_channels, time_points=time_points, num_classes=num_classes_gan, embedding_dim=embedding_dim_gan)
    model_name = f"{model_name_prefix}_cGAN"
    log_message(f"\n--- Training {model_name} for S{subject_display_idx_param}, Run {run_number} ---")
    gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step_wgan_cgan(real_eeg_batch, real_labels_batch_0_1_tf):
        batch_size_tf = tf.shape(real_eeg_batch)[0]

        # Train the Critic
        for _ in range(n_critic_steps):
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch_size_tf, latent_dim])
                fake_eeg = generator([noise, real_labels_batch_0_1_tf], training=True)
                real_output = critic([real_eeg_batch, real_labels_batch_0_1_tf], training=True)
                fake_output = critic([fake_eeg, real_labels_batch_0_1_tf], training=True)
                gp = gradient_penalty_cgan(critic, real_eeg_batch, fake_eeg, real_labels_batch_0_1_tf)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
            gradients_d = tape.gradient(d_loss, critic.trainable_variables)
            crit_opt.apply_gradients(zip(gradients_d, critic.trainable_variables))

        # Train the Generator
        with tf.GradientTape() as tape_g:
            noise_g = tf.random.normal([batch_size_tf, latent_dim])
            fake_eeg_for_g = generator([noise_g, real_labels_batch_0_1_tf], training=True)
            fake_output_for_g = critic([fake_eeg_for_g, real_labels_batch_0_1_tf], training=True)
            g_loss_wasserstein = -tf.reduce_mean(fake_output_for_g)
            freq_loss_val = frequency_domain_loss(real_eeg_batch, fake_eeg_for_g)
            total_g_loss = g_loss_wasserstein + (freq_loss_weight * freq_loss_val)
        
        gradients_g = tape_g.gradient(total_g_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))

        return d_loss, total_g_loss

    d_losses, g_losses = [], []
    if eeg_data.shape[0] == 0: return None
    
    dataset = tf.data.Dataset.from_tensor_slices((eeg_data.astype(np.float32), labels_0_1_reshaped.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=len(eeg_data)).batch(batch_size, drop_remainder=True)
    num_batches_per_epoch = len(eeg_data) // batch_size

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss_combined = 0.0, 0.0
        
        for real_eeg_batch, real_labels_batch in dataset:
            d_loss, g_loss_comb = train_step_wgan_cgan(real_eeg_batch, real_labels_batch)
            epoch_d_loss += d_loss
            epoch_g_loss_combined += g_loss_comb

        avg_epoch_d_loss = epoch_d_loss / num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        avg_epoch_g_loss = epoch_g_loss_combined / num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        d_losses.append(avg_epoch_d_loss)
        g_losses.append(avg_epoch_g_loss)

        if epoch % 100 == 0 or epoch == epochs - 1:
            log_message(f"Epoch {epoch}/{epochs} ({model_name}, S{subject_display_idx_param}, Run {run_number}): D Loss = {avg_epoch_d_loss:.4f}, G Loss (Comb) = {avg_epoch_g_loss:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(d_losses, label="Critic Loss"); plt.plot(g_losses, label="Gen Loss (W+Freq)")
    plt.title(f"cGAN Losses - {model_name_prefix} S{subject_display_idx_param} R{run_number}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plot_filename = os.path.join(output_dir_param, f"WGAN_GP_Losses_{model_name_prefix}_cGAN_S{subject_display_idx_param}_Run{run_number}.png")
    plt.savefig(plot_filename); plt.close()
    return generator

def generate_synthetic_data_cgan(generator_cgan_model, num_samples_gen, target_label_0_1_gen,
                                 latent_dim_gen=LATENT_DIM_CGAN, smooth=True, window_size=5):
    if generator_cgan_model is None or num_samples_gen == 0: return np.array([])
    noise = tf.random.normal([num_samples_gen, latent_dim_gen])
    labels_for_generation = tf.ones((num_samples_gen, 1), dtype=tf.int32) * target_label_0_1_gen
    synthetic_data = generator_cgan_model([noise, labels_for_generation], training=False)
    synthetic_data_np = synthetic_data.numpy()
    if smooth and synthetic_data_np.size > 0 :
        synthetic_data_np = smooth_eeg(synthetic_data_np, window_size)
    return synthetic_data_np


def plot_final_accuracies(baseline_acc, synth_only_acc, best_aug_acc, subject_id, output_dir, dataset_name):
    """Plots a bar chart comparing baseline, synth-only, and best augmented accuracies."""
    labels = ['Baseline', 'Synth-Only', 'Best Augmented']
    accuracies = [baseline_acc, synth_only_acc, best_aug_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=['blue', 'red', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'Final Accuracy Comparison for Subject {subject_id} on {dataset_name}')
    plt.ylim(0, 100)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    plot_filename = os.path.join(output_dir, f"final_accuracies_S{subject_id}.png")
    plt.savefig(plot_filename)
    plt.close()
    log_message(f"Saved final accuracy plot to {plot_filename}")


def plot_final_accuracies(baseline_acc, synth_only_acc, best_aug_acc, subject_id, output_dir, dataset_name):
    """Plots a bar chart comparing baseline, synth-only, and best augmented accuracies."""
    labels = ['Baseline', 'Synth-Only', 'Best Augmented']
    accuracies = [baseline_acc, synth_only_acc, best_aug_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=['blue', 'red', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'Final Accuracy Comparison for Subject {subject_id} on {dataset_name}')
    plt.ylim(0, 100)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    plot_filename = os.path.join(output_dir, f"final_accuracies_S{subject_id}.png")
    plt.savefig(plot_filename)
    plt.close()
    log_message(f"Saved final accuracy plot to {plot_filename}")


# --- Plotting Functions (Unchanged) ---
def compute_grand_average(data_plot, labels_plot, channel_idx_plot, class_labels_plot=[1, 2],
                          sampling_rate_plot=250, title_prefix_plot="", confidence_plot=0.95):
    results = {}
    labels_plot = np.asarray(labels_plot).squeeze()
    for class_label_plot in class_labels_plot:
        class_indices_plot = np.where(labels_plot == class_label_plot)[0]
        if len(class_indices_plot) == 0:
            log_message(f"    {title_prefix_plot} Data: No trials for class {class_label_plot} for GA, channel {channel_idx_plot}.")
            continue
        class_data_plot = data_plot[class_indices_plot, channel_idx_plot, :]
        grand_avg_plot = np.mean(class_data_plot, axis=0)
        sem_plot = stats.sem(class_data_plot, axis=0, nan_policy='omit')
        df_plot = np.sum(~np.isnan(class_data_plot), axis=0) -1
        df_plot[df_plot < 1] = 1
        ci_plot = np.zeros_like(grand_avg_plot)
        valid_sem_mask_plot = (sem_plot > 0) & ~np.isnan(sem_plot) & (df_plot > 0)
        if np.any(valid_sem_mask_plot):
            ci_plot[valid_sem_mask_plot] = sem_plot[valid_sem_mask_plot] * stats.t.ppf((1 + confidence_plot) / 2., df_plot[valid_sem_mask_plot])
        results[f"class_{class_label_plot}"] = {"grand_avg": grand_avg_plot, "lower_ci": grand_avg_plot - ci_plot, "upper_ci": grand_avg_plot + ci_plot, "n_trials": len(class_indices_plot)}
    return results

def plot_grand_average_comparison(real_data_train_plot, synthetic_data_plot, real_labels_train_plot, synthetic_labels_plot,
                                 channel_idx_plot, class_labels_plot=[1, 2], sampling_rate_plot=250,
                                 output_dir_param=None, subject_display_idx_param=None,
                                 augmented_data=None, augmented_labels=None):
    original_channel_number_plot = SELECTED_CHANNELS_1_BASED[channel_idx_plot]
    log_message(f"\n--- Plotting GA Comparison for Sel.ChIdx {channel_idx_plot} (OrigCh {original_channel_number_plot}) (S{subject_display_idx_param}) ---")
    real_results_plot = compute_grand_average(real_data_train_plot, real_labels_train_plot, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Real (Train Split)")
    synth_results_plot = compute_grand_average(synthetic_data_plot, synthetic_labels_plot, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Synthetic (cGAN)")
    aug_results_plot = None
    if augmented_data is not None and augmented_labels is not None:
        aug_results_plot = compute_grand_average(augmented_data, augmented_labels, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Augmented (100%)")

    time_points_count_plot = real_data_train_plot.shape[2]
    time_vector_plot = np.arange(time_points_count_plot) / sampling_rate_plot

    for class_label_plot in class_labels_plot:
        class_key_plot = f"class_{class_label_plot}"
        real_class_result_plot = real_results_plot.get(class_key_plot)
        synth_class_result_plot = synth_results_plot.get(class_key_plot)
        aug_class_result_plot = aug_results_plot.get(class_key_plot) if aug_results_plot else None
        plt.figure(figsize=(12, 6))
        plot_successful_ga = False
        if real_class_result_plot:
            plt.plot(time_vector_plot, real_class_result_plot["grand_avg"], 'b-', linewidth=2, label=f'Real (n={real_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, real_class_result_plot["lower_ci"], real_class_result_plot["upper_ci"], color='blue', alpha=0.2)
            plot_successful_ga = True
        if synth_class_result_plot:
            plt.plot(time_vector_plot, synth_class_result_plot["grand_avg"], 'r-', linewidth=2, label=f'Synthetic (n={synth_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, synth_class_result_plot["lower_ci"], synth_class_result_plot["upper_ci"], color='red', alpha=0.2)
            plot_successful_ga = True
        if aug_class_result_plot:
            plt.plot(time_vector_plot, aug_class_result_plot["grand_avg"], 'g-', linewidth=2, label=f'Augmented (n={aug_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, aug_class_result_plot["lower_ci"], aug_class_result_plot["upper_ci"], color='green', alpha=0.2)
            plot_successful_ga = True

        if plot_successful_ga and output_dir_param and subject_display_idx_param:
            plt.title(f"GA Comparison - Cls {class_label_plot} - Sel.ChIdx {channel_idx_plot} (OrigCh {original_channel_number_plot}) - S{subject_display_idx_param}")
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude (Norm)"); plt.legend(); plt.grid(True); plt.tight_layout()
            plot_filename = os.path.join(output_dir_param, f"Grand_Avg_Cls{class_label_plot}_SelChIdx{channel_idx_plot}_OrigCh{original_channel_number_plot}_S{subject_display_idx_param}.png")
            plt.savefig(plot_filename); plt.close()
            if real_class_result_plot and synth_class_result_plot:
                 log_message(f"  GA Stats (Real vs Synth) Cls {class_label_plot}: Corr: {np.corrcoef(real_class_result_plot['grand_avg'], synth_class_result_plot['grand_avg'])[0, 1]:.4f}, RMSE: {np.sqrt(np.mean((real_class_result_plot['grand_avg'] - synth_class_result_plot['grand_avg'])**2)):.4f}")
                 real_fft_abs = np.abs(fft(real_class_result_plot["grand_avg"])); synth_fft_abs = np.abs(fft(synth_class_result_plot["grand_avg"]))
                 log_message(f"  Freq W-dist (Real vs Synth): {wasserstein_distance(real_fft_abs[:len(real_fft_abs)//2], synth_fft_abs[:len(synth_fft_abs)//2]):.4f}")
            if real_class_result_plot and aug_class_result_plot:
                 log_message(f"  GA Stats (Real vs Aug) Cls {class_label_plot}: Corr: {np.corrcoef(real_class_result_plot['grand_avg'], aug_class_result_plot['grand_avg'])[0, 1]:.4f}, RMSE: {np.sqrt(np.mean((real_class_result_plot['grand_avg'] - aug_class_result_plot['grand_avg'])**2)):.4f}")
                 real_fft_abs = np.abs(fft(real_class_result_plot["grand_avg"])); aug_fft_abs = np.abs(fft(aug_class_result_plot["grand_avg"]))
                 log_message(f"  Freq W-dist (Real vs Aug): {wasserstein_distance(real_fft_abs[:len(real_fft_abs)//2], aug_fft_abs[:len(aug_fft_abs)//2]):.4f}")
        else:
            log_message(f"  No data to plot GA for Cls {class_label_plot}, Sel.ChIdx {channel_idx_plot}. Skipping.")
            plt.close()

def plot_psd_comparison(real_data_train_psd, synthetic_data_psd, real_labels_train_psd, synthetic_labels_psd,
                        channel_idx_psd, class_labels_psd=[1, 2], sampling_rate_psd=250,
                        output_dir_param=None, subject_display_idx_param=None,
                        augmented_data=None, augmented_labels=None):
    original_channel_number_psd = SELECTED_CHANNELS_1_BASED[channel_idx_psd]
    log_message(f"\n--- PSD ANALYSIS for Sel.ChIdx {channel_idx_psd} (OrigCh {original_channel_number_psd}) (S{subject_display_idx_param}) ---")
    for class_label_psd in class_labels_psd:
        plt.figure(figsize=(12, 6))
        common_freq_axis_psd = None; plot_successful_psd_flag = False
        avg_real_psd_val, avg_synth_psd_val, avg_aug_psd_val = (None, None, None)

        labels_real_psd = np.asarray(real_labels_train_psd).squeeze()
        real_indices_psd = np.where(labels_real_psd == class_label_psd)[0]
        if len(real_indices_psd) > 0:
            real_trials_ch_psd = real_data_train_psd[real_indices_psd, channel_idx_psd, :]
            real_psds_list_psd = []; current_common_freq_axis_real_psd = None
            for i in range(real_trials_ch_psd.shape[0]):
                nperseg_val = min(256, real_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, real_trials_ch_psd.shape[1] // 2 -1)
                if nperseg_val <= 0 or noverlap_val < 0 or noverlap_val >= nperseg_val : continue
                f_psd, psd_val = signal.welch(real_trials_ch_psd[i,:], fs=sampling_rate_psd, nperseg=nperseg_val, noverlap=noverlap_val)
                if current_common_freq_axis_real_psd is None: current_common_freq_axis_real_psd = f_psd
                if len(f_psd) == len(current_common_freq_axis_real_psd) and np.allclose(f_psd, current_common_freq_axis_real_psd): real_psds_list_psd.append(psd_val)
                else: real_psds_list_psd.append(np.interp(current_common_freq_axis_real_psd, f_psd, psd_val))
            if real_psds_list_psd:
                common_freq_axis_psd = current_common_freq_axis_real_psd
                avg_real_psd_val = np.mean(np.array(real_psds_list_psd), axis=0); sem_real_psd_val = stats.sem(np.array(real_psds_list_psd), axis=0, nan_policy='omit')
                freq_mask_psd = common_freq_axis_psd <= 50
                plt.plot(common_freq_axis_psd[freq_mask_psd], avg_real_psd_val[freq_mask_psd], 'b-', linewidth=2, label=f'Real (n={len(real_indices_psd)})')
                plt.fill_between(common_freq_axis_psd[freq_mask_psd], np.nan_to_num(avg_real_psd_val[freq_mask_psd] - 1.96 * sem_real_psd_val[freq_mask_psd]), np.nan_to_num(avg_real_psd_val[freq_mask_psd] + 1.96 * sem_real_psd_val[freq_mask_psd]), color='blue', alpha=0.2)
                plot_successful_psd_flag = True

        labels_synth_psd = np.asarray(synthetic_labels_psd).squeeze()
        synth_indices_psd = np.where(labels_synth_psd == class_label_psd)[0]
        if len(synth_indices_psd) > 0:
            synth_trials_ch_psd = synthetic_data_psd[synth_indices_psd, channel_idx_psd, :]
            synth_psds_list_psd = []
            for i in range(synth_trials_ch_psd.shape[0]):
                nperseg_val = min(256, synth_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, synth_trials_ch_psd.shape[1] // 2 -1)
                if nperseg_val <= 0 or noverlap_val < 0 or noverlap_val >= nperseg_val: continue
                f_psd, psd_val = signal.welch(synth_trials_ch_psd[i,:], fs=sampling_rate_psd, nperseg=nperseg_val, noverlap=noverlap_val)
                if common_freq_axis_psd is None: continue
                if len(f_psd) == len(common_freq_axis_psd) and np.allclose(f_psd, common_freq_axis_psd): synth_psds_list_psd.append(psd_val)
                else: synth_psds_list_psd.append(np.interp(common_freq_axis_psd, f_psd, psd_val))
            if synth_psds_list_psd and common_freq_axis_psd is not None:
                avg_synth_psd_val = np.mean(np.array(synth_psds_list_psd), axis=0); sem_synth_psd_val = stats.sem(np.array(synth_psds_list_psd), axis=0, nan_policy='omit')
                freq_mask_psd = common_freq_axis_psd <= 50
                plt.plot(common_freq_axis_psd[freq_mask_psd], avg_synth_psd_val[freq_mask_psd], 'r-', linewidth=2, label=f'Synthetic (n={len(synth_indices_psd)})')
                plt.fill_between(common_freq_axis_psd[freq_mask_psd], np.nan_to_num(avg_synth_psd_val[freq_mask_psd] - 1.96 * sem_synth_psd_val[freq_mask_psd]), np.nan_to_num(avg_synth_psd_val[freq_mask_psd] + 1.96 * sem_synth_psd_val[freq_mask_psd]), color='red', alpha=0.2)
                plot_successful_psd_flag = True

        if augmented_data is not None and augmented_labels is not None:
            labels_aug_psd = np.asarray(augmented_labels).squeeze()
            aug_indices_psd = np.where(labels_aug_psd == class_label_psd)[0]
            if len(aug_indices_psd) > 0:
                aug_trials_ch_psd = augmented_data[aug_indices_psd, channel_idx_psd, :]
                aug_psds_list_psd = []
                for i in range(aug_trials_ch_psd.shape[0]):
                    nperseg_val = min(256, aug_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, aug_trials_ch_psd.shape[1] // 2 - 1)
                    if nperseg_val <= 0 or noverlap_val < 0 or noverlap_val >= nperseg_val: continue
                    f_psd, psd_val = signal.welch(aug_trials_ch_psd[i,:], fs=sampling_rate_psd, nperseg=nperseg_val, noverlap=noverlap_val)
                    if common_freq_axis_psd is None: continue
                    if len(f_psd) == len(common_freq_axis_psd) and np.allclose(f_psd, common_freq_axis_psd): aug_psds_list_psd.append(psd_val)
                    else: aug_psds_list_psd.append(np.interp(common_freq_axis_psd, f_psd, psd_val))
                if aug_psds_list_psd and common_freq_axis_psd is not None:
                    avg_aug_psd_val = np.mean(np.array(aug_psds_list_psd), axis=0); sem_aug_psd_val = stats.sem(np.array(aug_psds_list_psd), axis=0, nan_policy='omit')
                    freq_mask_psd = common_freq_axis_psd <= 50
                    plt.plot(common_freq_axis_psd[freq_mask_psd], avg_aug_psd_val[freq_mask_psd], 'g-', linewidth=2, label=f'Augmented (n={len(aug_indices_psd)})')
                    plt.fill_between(common_freq_axis_psd[freq_mask_psd], np.nan_to_num(avg_aug_psd_val[freq_mask_psd] - 1.96 * sem_aug_psd_val[freq_mask_psd]), np.nan_to_num(avg_aug_psd_val[freq_mask_psd] + 1.96 * sem_aug_psd_val[freq_mask_psd]), color='green', alpha=0.2)
                    plot_successful_psd_flag = True

        if plot_successful_psd_flag and common_freq_axis_psd is not None and output_dir_param and subject_display_idx_param:
            plt.xlabel('Freq (Hz)'); plt.ylabel('PSD (µV²/Hz approx.)')
            plt.title(f'PSD Comparison - Cls {class_label_psd}, Sel.ChIdx {channel_idx_psd} (OrigCh {original_channel_number_psd}) - S{subject_display_idx_param}')
            plt.legend(); plt.grid(True); plt.xlim(0, 50); plt.ylim(bottom=0); plt.tight_layout()
            plot_filename = os.path.join(output_dir_param, f"PSD_Cls{class_label_psd}_SelChIdx{channel_idx_psd}_OrigCh{original_channel_number_psd}_S{subject_display_idx_param}.png")
            plt.savefig(plot_filename); plt.close()
        else:
            log_message(f"  No data to plot PSD for Cls {class_label_psd}. Skipping.")
            plt.close()

# --- List of Datasets to Process ---
datasets_to_process = [f'DATA{i}.mat' for i in range(3, 6)]

# --- Main Loop for Processing Each Dataset ---
for dataset_name in datasets_to_process:
    print(f"\n\n{'='*20} STARTING PROCESSING FOR DATASET: {dataset_name} {'='*20}")
    
    # --- Load the .mat file ---
    try:
        data_mat = scipy.io.loadmat(dataset_name)
        xsubi_all_data = data_mat['xsubi_all']
    except (FileNotFoundError, KeyError) as e:
        print(f"FATAL: Could not load data from {dataset_name}. Error: {e}. SKIPPING.")
        continue

    NUM_SUBJECTS_TRAIN = xsubi_all_data.shape[1]
    dataset_prefix = dataset_name.split('.')[0]

    # --- Configuration for subject processing ---
    TARGET_SUBJECTS = "all"
    all_subject_indices_available = list(range(NUM_SUBJECTS_TRAIN))
    subjects_to_process_indices = []

    if isinstance(TARGET_SUBJECTS, str) and TARGET_SUBJECTS.lower() == "all":
        subjects_to_process_indices = all_subject_indices_available
    elif isinstance(TARGET_SUBJECTS, int):
        subjects_to_process_indices = [TARGET_SUBJECTS - 1] if 1 <= TARGET_SUBJECTS <= NUM_SUBJECTS_TRAIN else all_subject_indices_available
    elif isinstance(TARGET_SUBJECTS, list) and all(isinstance(i, int) for i in TARGET_SUBJECTS):
        valid_ids_0_based = [s_id - 1 for s_id in TARGET_SUBJECTS if 1 <= s_id <= NUM_SUBJECTS_TRAIN]
        subjects_to_process_indices = sorted(list(set(valid_ids_0_based))) if valid_ids_0_based else all_subject_indices_available
    else:
        subjects_to_process_indices = all_subject_indices_available

    if not subjects_to_process_indices:
        print(f"CRITICAL: No subjects selected for processing in {dataset_name}. SKIPPING.")
        continue
    else:
        print(f"INFO: For {dataset_name}, will process {len(subjects_to_process_indices)} subject(s). 1-based IDs: {[i+1 for i in subjects_to_process_indices]}")

    # --- Main Loop for Processing Each Subject ---
    print(f"\nStarting processing for {len(subjects_to_process_indices)} selected subject(s) from {dataset_name}.")
    print(f"Global random seed set to: {SEED_VALUE}")

    for subject_idx in subjects_to_process_indices:
        current_subject_display_id = subject_idx + 1
        output_dir = f"{dataset_prefix}_results/Subject_{current_subject_display_id}"
        os.makedirs(output_dir, exist_ok=True)
        current_log_file_path = os.path.join(output_dir, "run_log.txt")
        with open(current_log_file_path, 'w') as f:
            f.write(f"Log for Subject {current_subject_display_id} from {dataset_name} (0-indexed: {subject_idx})\n")
            f.write(f"Output directory: {output_dir}\nGlobal seed: {SEED_VALUE}\n")
            f.write(f"BASELINE PARAMS: Channels={SELECTED_CHANNELS_1_BASED}, Freq=[{LOWCUT_BASELINE}-{HIGHCUT_BASELINE} Hz], FS={FS_BASELINE} Hz, Trim={TIME_TRIM_SECONDS_BASELINE}s\n")
            f.write(f"Train/Valid/Test Split: {NUM_TRAIN_TRIALS_PER_CLASS_BASELINE}/{NUM_VALID_TRIALS_PER_CLASS_BASELINE} trials per class.\n")
            f.write(f"GAN Runs: {NUM_RUNS_PER_SUBJECT}\n")
            f.write("="*70 + "\n\n")

        log_message(f"PROCESSING SUBJECT {current_subject_display_id}/{NUM_SUBJECTS_TRAIN} from {dataset_name}")

        try:
            xsubi1_struct = xsubi_all_data[0, subject_idx]

            log_message(f"\n--- Step 1: PERFORMING FIXED TRAIN/VALID/TEST SPLIT ---")
            np.random.seed(SEED_VALUE)

            x_train_gan_fmt, y_train_labels, final_train_csp, \
            final_valid_csp, final_test_csp, train_min, train_max, num_synthetic_samples_per_class = \
                preprocess_and_split_subject_data(
                    xsubi1_struct, str(current_subject_display_id), dataset_name,
                    SELECTED_CHANNELS_0_BASED, NUM_TRAIN_TRIALS_PER_CLASS_BASELINE,
                    NUM_VALID_TRIALS_PER_CLASS_BASELINE
                )

            if x_train_gan_fmt is None:
                log_message(f"CRITICAL ERROR: Data preprocessing and splitting failed for S{current_subject_display_id}. Skipping.")
                continue

            # Save the complete training set
            training_set_filename = os.path.join(output_dir, f"training_{current_subject_display_id}.mat")
            scipy.io.savemat(training_set_filename, {
                'training_set_x': final_train_csp['x'],
                'training_set_y': final_train_csp['y']
            })
            log_message(f"Saved complete training set to {training_set_filename}")

            log_message(f"\n--- Step 2: Baseline Evaluation ---")
            train_valid_csp = {
                'x': np.concatenate((final_train_csp['x'], final_valid_csp['x']), axis=2),
                'y': np.concatenate((final_train_csp['y'], final_valid_csp['y']))
            }
            log_message(f"Created combined train+valid set for baseline model training. Shape: {train_valid_csp['x'].shape}")
            
            model_baseline, sf_baseline = train_cspsvm(train_valid_csp)
            accuracy_baseline = 0.0
            if model_baseline:
                accuracy_baseline, _, _ = evaluate_cspsvm(model_baseline, sf_baseline, final_test_csp)
            log_message(f"Baseline Accuracy (Train+Valid -> Test): {accuracy_baseline:.2f}%")

            log_message(f"\n--- Step 3: {NUM_RUNS_PER_SUBJECT} cGAN RUNS & BATCH SELECTION ---")
            all_generated_batches = []
            for run_idx in range(NUM_RUNS_PER_SUBJECT):
                run_number = run_idx + 1
                cgan_generator_trained = train_wgan_gp_cgan(
                    eeg_data=x_train_gan_fmt, eeg_labels=y_train_labels,
                    epochs=1500, batch_size=min(64, x_train_gan_fmt.shape[0]),
                    model_name_prefix='MI', num_gan_channels=NUM_SELECTED_CHANNELS,
                    time_points=x_train_gan_fmt.shape[2], freq_loss_weight=1.5,
                    output_dir_param=output_dir, subject_display_idx_param=current_subject_display_id,
                    run_number=run_number
                )

                if not cgan_generator_trained:
                    log_message(f"    Run {run_number}: Failed to train the generator. Skipping batch generation.")
                    continue

                log_message(f"  --- Generating & Evaluating Batches with Validation Accuracy (Run {run_number}) ---")
                num_synth_per_class_eval = final_valid_csp['x'].shape[2] // 2

                for batch_idx in range(30):
                    synth_c0 = generate_synthetic_data_cgan(cgan_generator_trained, num_synth_per_class_eval, 0)
                    synth_c1 = generate_synthetic_data_cgan(cgan_generator_trained, num_synth_per_class_eval, 1)

                    if not (synth_c0.size > 0 and synth_c1.size > 0):
                        log_message(f"    Run {run_number}, Batch {batch_idx+1}: Failed to generate synthetic data.")
                        continue

                    synth_data_gan_fmt = np.concatenate((synth_c0, synth_c1), axis=0)
                    synth_labels_gan_fmt = np.concatenate((np.ones(synth_c0.shape[0]), np.ones(synth_c1.shape[0]) + 1))
                    synth_data_csp_fmt = {
                        'x': np.transpose(synth_data_gan_fmt, (2, 1, 0)),
                        'y': synth_labels_gan_fmt
                    }

                    # Train on synthetic batch and evaluate on validation set
                    model_synth_batch, sf_synth_batch = train_cspsvm(synth_data_csp_fmt)
                    batch_accuracy = 0.0
                    if model_synth_batch:
                        batch_accuracy, _, _ = evaluate_cspsvm(model_synth_batch, sf_synth_batch, final_valid_csp)

                    log_message(f"    Run {run_number}, Batch {batch_idx+1}: Validation Accuracy: {batch_accuracy:.4f}")

                    all_generated_batches.append({
                        'run_index': run_number,
                        'batch_index': batch_idx + 1,
                        'selection_score': batch_accuracy,
                        'synthetic_features': synth_data_csp_fmt
                    })

            log_message("\n\n--- Step 4: Selecting Best Augmentation Strategy ---")
            if not all_generated_batches:
                log_message("CRITICAL ERROR: No synthetic batches were generated. Cannot proceed."); continue

            all_generated_batches.sort(key=lambda x: x['selection_score'], reverse=True)
            top_10_batches = all_generated_batches[:10]
            log_message(f"Selected top {len(top_10_batches)} synthetic batches based on validation accuracy.")

            all_evaluated_strategies = []
            mix_ratios = [0, 0.25, 0.50, 1.0]  # 0 represents 'synth-only'

            log_message("\n--- Evaluating strategies on the validation set using classification accuracy ---")
            for batch_info in top_10_batches:
                synth_features_for_aug = batch_info['synthetic_features']
                for ratio in mix_ratios:
                    if ratio == 0:
                        training_data_strat = synth_features_for_aug
                        strategy_name = 'Synth Only'
                    else:
                        strategy_name = f"Augmented ({int(ratio*100)}%)"
                        num_real_trials = final_train_csp['x'].shape[2]
                        num_synth_to_add = int(num_real_trials * ratio)
                        if num_synth_to_add == 0: continue

                        synth_x_to_add = synth_features_for_aug['x'][:, :, :num_synth_to_add]
                        synth_y_to_add = synth_features_for_aug['y'][:num_synth_to_add]

                        augmented_x = np.concatenate((final_train_csp['x'], synth_x_to_add), axis=2)
                        augmented_y = np.concatenate((final_train_csp['y'], synth_y_to_add))
                        training_data_strat = {'x': augmented_x, 'y': augmented_y}

                    model_strategy, sf_strategy = train_cspsvm(training_data_strat)
                    validation_accuracy = 0.0
                    if model_strategy:
                        validation_accuracy, _, _ = evaluate_cspsvm(model_strategy, sf_strategy, final_valid_csp)

                    log_message(f"    Run {batch_info['run_index']}, Batch {batch_info['batch_index']}, Strategy '{strategy_name}': Validation Acc: {validation_accuracy:.2f}%")
                    all_evaluated_strategies.append({
                        'run_index': batch_info['run_index'],
                        'batch_index': batch_info['batch_index'],
                        'mix_ratio': ratio,
                        'validation_accuracy': validation_accuracy,
                        'selection_score': batch_info['selection_score'],
                        'synthetic_features': batch_info['synthetic_features']
                    })



            if not all_evaluated_strategies: log_message("CRITICAL: No strategies evaluated."); continue

            max_validation_accuracy = max(s['validation_accuracy'] for s in all_evaluated_strategies)
            top_accuracy_strategies = [s for s in all_evaluated_strategies if s['validation_accuracy'] == max_validation_accuracy]

            if len(top_accuracy_strategies) > 1:
                log_message(f"\nFound {len(top_accuracy_strategies)} strategies with top validation accuracy of {max_validation_accuracy:.2f}%. Using initial synthetic-only accuracy as tie-breaker.")
                top_accuracy_strategies.sort(key=lambda x: x['selection_score'], reverse=True)
            
            best_strategy_details = top_accuracy_strategies[0]
            best_strategy_name = 'Synth Only' if best_strategy_details['mix_ratio'] == 0 else f"Augmented ({int(best_strategy_details['mix_ratio']*100)}%)"
            log_message(f"\nSelected best strategy: Run {best_strategy_details['run_index']}, Batch {best_strategy_details['batch_index']}, Strategy: {best_strategy_name} with Val Acc {best_strategy_details['validation_accuracy']:.2f}% and initial synth-only accuracy: {best_strategy_details['selection_score']:.4f}")

            # --- SAVE AND PLOT BEST BATCH ---
            best_synthetic_batch_csp = best_strategy_details['synthetic_features']
            
            # Save the final selected synthetic batch
            synthetic_batch_filename = os.path.join(output_dir, f"synthetic_{current_subject_display_id}.mat")
            scipy.io.savemat(synthetic_batch_filename, {
                'synthetic_x': best_synthetic_batch_csp['x'],
                'synthetic_y': best_synthetic_batch_csp['y']
            })
            log_message(f"Saved final selected synthetic batch to {synthetic_batch_filename}")

            # Transpose synthetic data for plotting functions
            best_synthetic_batch_gan = np.transpose(best_synthetic_batch_csp['x'], (2, 1, 0))
            best_synthetic_labels = best_synthetic_batch_csp['y']

            # Create 100% augmented data for plotting comparison
            num_real_trials_plot = x_train_gan_fmt.shape[0]
            num_synth_to_add_plot = min(num_real_trials_plot, best_synthetic_batch_gan.shape[0])
            synth_x_to_add_plot = best_synthetic_batch_gan[:num_synth_to_add_plot, :, :]
            synth_y_to_add_plot = best_synthetic_labels[:num_synth_to_add_plot]
            
            augmented_x_plot = np.concatenate((x_train_gan_fmt, synth_x_to_add_plot), axis=0)
            augmented_y_plot = np.concatenate((y_train_labels, synth_y_to_add_plot))

            # Generate comparison plots for a few channels
            for ch_idx_to_plot in [0, 5, 11]: # Example: plot for 3 channels
                if ch_idx_to_plot < NUM_SELECTED_CHANNELS:
                    plot_grand_average_comparison(
                        real_data_train_plot=x_train_gan_fmt,
                        synthetic_data_plot=best_synthetic_batch_gan,
                        real_labels_train_plot=y_train_labels,
                        synthetic_labels_plot=best_synthetic_labels,
                        channel_idx_plot=ch_idx_to_plot,
                        sampling_rate_plot=FS_BASELINE,
                        output_dir_param=output_dir,
                        subject_display_idx_param=current_subject_display_id,
                        augmented_data=augmented_x_plot,
                        augmented_labels=augmented_y_plot
                    )
                    plot_psd_comparison(
                        real_data_train_psd=x_train_gan_fmt,
                        synthetic_data_psd=best_synthetic_batch_gan,
                        real_labels_train_psd=y_train_labels,
                        synthetic_labels_psd=best_synthetic_labels,
                        channel_idx_psd=ch_idx_to_plot,
                        sampling_rate_psd=FS_BASELINE,
                        output_dir_param=output_dir,
                        subject_display_idx_param=current_subject_display_id,
                        augmented_data=augmented_x_plot,
                        augmented_labels=augmented_y_plot
                    )

            log_message("\n\n--- Step 5: Final Evaluation on Test Set ---")
            
            # A) Evaluate the best 'synth-only' strategy on the test set
            synth_only_strategies = [s for s in all_evaluated_strategies if s['mix_ratio'] == 0]
            accuracy_best_synth_on_test = 0.0
            if synth_only_strategies:
                best_synth_only_strategy = max(synth_only_strategies, key=lambda x: x['validation_accuracy'])
                model_best_synth, sf_best_synth = train_cspsvm(best_synth_only_strategy['synthetic_features'])
                if model_best_synth:
                    accuracy_best_synth_on_test, _, _ = evaluate_cspsvm(model_best_synth, sf_best_synth, final_test_csp)
            log_message(f"BEST SYNTHETIC-ONLY (Val Acc: {best_synth_only_strategy['validation_accuracy']:.2f}%) -> REAL test accuracy: {accuracy_best_synth_on_test:.2f}%")

            # B) Evaluate the overall best strategy on the test set
            log_message("Retraining best model on combined Train+Validation data for final evaluation.")
            final_model_train_data = None
            if best_strategy_details['mix_ratio'] == 0:
                final_model_train_data = best_strategy_details['synthetic_features']
            else:
                synth_features = best_strategy_details['synthetic_features']
                num_real = train_valid_csp['x'].shape[2]
                num_synth_to_add = int(num_real * best_strategy_details['mix_ratio'])
                if num_synth_to_add > synth_features['x'].shape[2]: num_synth_to_add = synth_features['x'].shape[2]

                synth_x_to_add = synth_features['x'][:, :, :num_synth_to_add]
                synth_y_to_add = synth_features['y'][:num_synth_to_add]

                aug_x = np.concatenate((train_valid_csp['x'], synth_x_to_add), axis=2)
                aug_y = np.concatenate((train_valid_csp['y'], synth_y_to_add))
                final_model_train_data = {'x': aug_x, 'y': aug_y}

            model_final, sf_final = train_cspsvm(final_model_train_data)
            accuracy_best_strategy_on_test = 0.0
            if model_final:
                accuracy_best_strategy_on_test, _, _ = evaluate_cspsvm(model_final, sf_final, final_test_csp)
            log_message(f"BEST OVERALL STRATEGY ({best_strategy_name}) -> REAL test accuracy: {accuracy_best_strategy_on_test:.2f}%")

            log_message(f"\n\n=== FINAL SUMMARY FOR S{current_subject_display_id} FROM {dataset_name} ===")
            log_message(f"BASELINE ACCURACY (Real train+val -> Real test): {accuracy_baseline:.2f}%")
            log_message(f"BEST SYNTHETIC-ONLY train → REAL test accuracy: {accuracy_best_synth_on_test:.2f}%")
            if best_strategy_details['mix_ratio'] != 0:
                log_message(f"BEST AUGMENTED ({int(best_strategy_details['mix_ratio']*100)}%) train → REAL test accuracy: {accuracy_best_strategy_on_test:.2f}%")

            # Plot final accuracies
            plot_final_accuracies(
                baseline_acc=accuracy_baseline,
                synth_only_acc=accuracy_best_synth_on_test,
                best_aug_acc=accuracy_best_strategy_on_test,
                subject_id=current_subject_display_id,
                output_dir=output_dir,
                dataset_name=dataset_prefix
            )

        except Exception as e_main:
            log_message(f"CRITICAL ERROR S{current_subject_display_id}: {e_main}\n{traceback.format_exc()}")
            log_message(f"--- Subject {current_subject_display_id} processing FAILED. ---")
