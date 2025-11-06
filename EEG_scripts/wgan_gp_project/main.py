# --- START OF FILE main.py ---

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# import tensorflow.keras.backend as K # Not explicitly used in the provided snippet
from scipy.signal import ellip, ellipord, filtfilt, welch # Added welch for PSD
from scipy import signal # For PSD, already implicitly available but good to be clear
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import traceback
from sklearn.svm import SVC
from scipy.linalg import eigh
import scipy.stats as stats # For grand average CI and SEM
from scipy.fft import fft # For grand average frequency analysis
from scipy.stats import wasserstein_distance # For grand average frequency analysis
import os # Added for directory and file operations
import csv # <<< NEW >>>: Imported for CSV writing

# --- Define Selected Channels (1-based) ---
SELECTED_CHANNELS_1_BASED = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17]
# Convert to 0-based indices for Python
SELECTED_CHANNELS_0_BASED = [idx - 1 for idx in SELECTED_CHANNELS_1_BASED]
NUM_SELECTED_CHANNELS = len(SELECTED_CHANNELS_0_BASED)

# --- Fixed trial counts for train/validation split across different files ---
NUM_TRAIN_TRIALS_PER_CLASS = 40
NUM_VALID_TRIALS_PER_CLASS = 10
NUM_RUNS_PER_SUBJECT = 1 # Number of GAN training runs per subject

# --- cGAN Specific Constants ---
NUM_CLASSES_CGAN = 2  # For left (class 0) and right (class 1) motor imagery
LATENT_DIM_CGAN = 100 # Latent dimension for cGAN noise
EMBEDDING_DIM_CGAN = 25 # Embedding dimension for class labels in cGAN

# --- Set Random Seed ---
SEED_VALUE = 42  # Example seed value
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE) # Also set TensorFlow seed for reproducibility

# --- Load the .mat files ---
data_mat_for_training = scipy.io.loadmat('data1.mat') # Source for TRAINING data
data_mat_for_validation_test = scipy.io.loadmat('data2.mat') # Source for VALIDATION and TEST data

# Access the nested 'data' key
xsubi_all_train_source = data_mat_for_training['xsubi_all']
xsubi_all_val_test_source = data_mat_for_validation_test['txsubi_all']

NUM_SUBJECTS_TRAIN = xsubi_all_train_source.shape[1]
NUM_SUBJECTS_VAL_TEST = xsubi_all_val_test_source.shape[1]

# Global variable for log file path, to be updated in the loop for each subject
current_log_file_path = None

def log_message(message):
    """Prints to console and appends to the current subject's log file."""
    print(message)
    global current_log_file_path
    if current_log_file_path:
        with open(current_log_file_path, 'a') as f:
            f.write(str(message) + "\n")
    else:
        print(f"WARNING: current_log_file_path not set. Message not logged to file: {message}")


# Bandpass filter definition using elliptical filter
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

# --- Robust Data Extraction Helper ---
def _get_eeg_data_from_field_main(subject_struct_item, field_name, subject_id_str, data_type_str="Data"):
    if not hasattr(subject_struct_item, 'dtype') or subject_struct_item.dtype.names is None:
        log_message(f"  ERROR S{subject_id_str} ({data_type_str}): Input is not a structured array. Cannot find field '{field_name}'.")
        return None
    if field_name not in subject_struct_item.dtype.names:
        log_message(f"  INFO S{subject_id_str} ({data_type_str}): Field '{field_name}' not found in subject data struct.")
        return None
    field_content = subject_struct_item[field_name]
    actual_data = None
    if isinstance(field_content, np.ndarray) and field_content.shape == (1,1) and \
       hasattr(field_content[0,0], 'shape'):
        actual_data = field_content[0,0]
    else:
        actual_data = field_content
    if not isinstance(actual_data, np.ndarray):
        log_message(f"  ERROR S{subject_id_str} ({data_type_str}): Field '{field_name}' content is not a NumPy array. Type: {type(actual_data)}")
        return None
    if field_name == 'y':
        return actual_data.flatten()
    if actual_data.ndim < 2:
         log_message(f"  ERROR S{subject_id_str} ({data_type_str}): Field '{field_name}' data has too few dims ({actual_data.ndim}). Shape: {actual_data.shape}")
         return None
    if actual_data.ndim == 2:
        s, c = actual_data.shape
        if s > c and c < 100 : # Heuristic: samples > channels, and few channels
            log_message(f"  INFO S{subject_id_str} ({data_type_str}): Field '{field_name}' data is 2D ({s}x{c}). Assuming (samples, channels) for single trial, expanding to 3D.")
            actual_data = np.expand_dims(actual_data, axis=2)
        elif c > s and s < 100: # Heuristic: channels > samples, and few samples (likely one channel's data for many trials, or transposed trial)
            log_message(f"  INFO S{subject_id_str} ({data_type_str}): Field '{field_name}' data is 2D ({s}x{c}). Assuming (channels, samples) for single trial, transposing and expanding to 3D.")
            actual_data = np.expand_dims(actual_data.T, axis=2)
        else: # Ambiguous or large 2D array
            log_message(f"  INFO S{subject_id_str} ({data_type_str}): Field '{field_name}' data is 2D ({s}x{c}). Assuming (samples, channels) for single trial, expanding to 3D.")
            actual_data = np.expand_dims(actual_data, axis=2)
    if actual_data.ndim != 3:
        log_message(f"  ERROR S{subject_id_str} ({data_type_str}): Field '{field_name}' data is not 3D after processing. Shape: {actual_data.shape}")
        return None
    if 0 in actual_data.shape:
        log_message(f"  ERROR S{subject_id_str} ({data_type_str}): Field '{field_name}' data has a zero dimension. Shape: {actual_data.shape}")
        return None
    return actual_data

# --- New Preprocessing Functions ---
def preprocess_training_data_final(raw_subject_struct, subject_id_str,
                                   num_train_per_class, selected_channels_0_based,
                                   lowfreq=8, highfreq=35, fs=250,
                                   startSample=125, endSample=624):
    try:
        log_message(f"  Preprocessing TRAINING data for S{subject_id_str} from data2.mat...")
        all_x_data_orig_ch = _get_eeg_data_from_field_main(raw_subject_struct, 'x', subject_id_str, "Train-'x'")
        if all_x_data_orig_ch is None: return (None,) * 5
        all_y_labels_orig = _get_eeg_data_from_field_main(raw_subject_struct, 'y', subject_id_str, "Train-'y'")
        num_total_trials_orig = all_x_data_orig_ch.shape[2]
        if all_y_labels_orig is None:
            log_message(f"  INFO S{subject_id_str} (Train): 'y' labels not found. Inferring...")
            if num_total_trials_orig < 2: return (None,) * 5
            num_class1_orig = num_total_trials_orig // 2
            num_class2_orig = num_total_trials_orig - num_class1_orig
            all_y_labels_orig = np.concatenate((np.ones(num_class1_orig), np.ones(num_class2_orig) + 1))
        if len(all_y_labels_orig) != num_total_trials_orig: return (None,) * 5

        class1_indices_orig = np.where(all_y_labels_orig == 1)[0]
        class2_indices_orig = np.where(all_y_labels_orig == 2)[0]

        if len(class1_indices_orig) < num_train_per_class or len(class2_indices_orig) < num_train_per_class:
            log_message(f"  ERROR S{subject_id_str} (Train): Not enough trials. Need {num_train_per_class}, have C1:{len(class1_indices_orig)}, C2:{len(class2_indices_orig)}.")
            return (None,) * 5

        train_c1_idx = class1_indices_orig[:num_train_per_class]
        train_c2_idx = class2_indices_orig[:num_train_per_class]

        x_c1_train = all_x_data_orig_ch[:, :, train_c1_idx]
        x_c2_train = all_x_data_orig_ch[:, :, train_c2_idx]
        x_train_combined = np.concatenate((x_c1_train, x_c2_train), axis=2)
        y_train_combined = np.concatenate((np.ones(num_train_per_class), np.ones(num_train_per_class) + 1))

        filtered_x_train = elliptical_filter(x_train_combined, lowcut=lowfreq, highcut=highfreq, fs=fs)
        windowed_x_train = filtered_x_train[startSample:endSample + 1, :, :]
        selected_channels_x_train = windowed_x_train[:, selected_channels_0_based, :]
        if selected_channels_x_train.size == 0: return (None,) * 5

        train_min_global = np.min(selected_channels_x_train)
        train_max_global = np.max(selected_channels_x_train)
        train_range_global = train_max_global - train_min_global
        epsilon_norm = 1e-8
        if train_range_global < epsilon_norm:
            normalized_x_train = np.zeros_like(selected_channels_x_train)
        else:
            normalized_x_train = 2 * (selected_channels_x_train - train_min_global) / train_range_global - 1

        x_train_gan = np.transpose(normalized_x_train, (2, 1, 0))
        y_train_gan = y_train_combined
        finaltrn_csp = {'x': normalized_x_train, 'y': y_train_combined}
        log_message(f"  S{subject_id_str} (Train): GAN data shape {x_train_gan.shape}, CSP data shape {finaltrn_csp['x'].shape}")

        return x_train_gan, y_train_gan, finaltrn_csp, train_min_global, train_max_global
    except Exception as e:
        log_message(f"ERROR S{subject_id_str} (Train) in preprocess_training_data_final: {e}\n{traceback.format_exc()}")
        return (None,) * 5

def preprocess_validation_test_data_final(raw_subject_struct, subject_id_str,
                                          num_valid_per_class, train_min_stat, train_max_stat,
                                          selected_channels_0_based,
                                          lowfreq=8, highfreq=35, fs=250,
                                          startSample=115, endSample=614):
    try:
        log_message(f"  Preprocessing VALIDATION/TEST data for S{subject_id_str} from data1.mat...")
        all_x_data_orig_ch = _get_eeg_data_from_field_main(raw_subject_struct, 'x', subject_id_str, "Val/Test-'x'")
        if all_x_data_orig_ch is None: return None, None
        all_y_labels_orig = _get_eeg_data_from_field_main(raw_subject_struct, 'y', subject_id_str, "Val/Test-'y'")
        num_total_trials_orig = all_x_data_orig_ch.shape[2]
        if all_y_labels_orig is None:
            log_message(f"  INFO S{subject_id_str} (Val/Test): 'y' labels not found. Inferring...")
            if num_total_trials_orig < 2: return None, None
            num_class1_orig = num_total_trials_orig // 2
            num_class2_orig = num_total_trials_orig - num_class1_orig
            all_y_labels_orig = np.concatenate((np.ones(num_class1_orig), np.ones(num_class2_orig) + 1))
        if len(all_y_labels_orig) != num_total_trials_orig: return None, None

        class1_indices_orig = np.where(all_y_labels_orig == 1)[0]
        class2_indices_orig = np.where(all_y_labels_orig == 2)[0]

        if len(class1_indices_orig) <= num_valid_per_class or len(class2_indices_orig) <= num_valid_per_class:
            log_message(f"  ERROR S{subject_id_str} (Val/Test): Not enough trials for val/test split. "
                        f"Need >{num_valid_per_class}, have C1:{len(class1_indices_orig)}, C2:{len(class2_indices_orig)}.")
            return None, None

        valid_c1_idx = class1_indices_orig[:num_valid_per_class]
        test_c1_idx = class1_indices_orig[num_valid_per_class:]
        valid_c2_idx = class2_indices_orig[:num_valid_per_class]
        test_c2_idx = class2_indices_orig[num_valid_per_class:]
        
        if len(test_c1_idx) == 0 or len(test_c2_idx) == 0:
            log_message(f"  WARNING S{subject_id_str} (Test): No test trials left after taking {num_valid_per_class} for validation.")

        def process_subset(x_data, y_labels):
            if x_data.size == 0: return None
            filtered_x = elliptical_filter(x_data, lowcut=lowfreq, highcut=highfreq, fs=fs)
            windowed_x = filtered_x[startSample:endSample + 1, :, :]
            selected_channels_x = windowed_x[:, selected_channels_0_based, :]
            if selected_channels_x.size == 0: return None
            train_range_stat = train_max_stat - train_min_stat
            epsilon_norm = 1e-8
            if train_range_stat < epsilon_norm:
                normalized_x = np.zeros_like(selected_channels_x)
            else:
                normalized_x = 2 * (selected_channels_x - train_min_stat) / train_range_stat - 1
                normalized_x = np.clip(normalized_x, -1, 1)
            return {'x': normalized_x, 'y': y_labels.squeeze()}

        x_valid_combined = np.concatenate((all_x_data_orig_ch[:, :, valid_c1_idx], all_x_data_orig_ch[:, :, valid_c2_idx]), axis=2)
        y_valid_combined = np.concatenate((np.ones(len(valid_c1_idx)), np.ones(len(valid_c2_idx)) + 1))
        finalval_csp = process_subset(x_valid_combined, y_valid_combined)
        if finalval_csp:
            log_message(f"  S{subject_id_str} (Validation): Final CSP data shape {finalval_csp['x'].shape}")
        else:
            log_message(f"  S{subject_id_str} (Validation): Processing failed.")

        x_test_combined = np.concatenate((all_x_data_orig_ch[:, :, test_c1_idx], all_x_data_orig_ch[:, :, test_c2_idx]), axis=2)
        y_test_combined = np.concatenate((np.ones(len(test_c1_idx)), np.ones(len(test_c2_idx)) + 1))
        finaltest_csp = process_subset(x_test_combined, y_test_combined)
        if finaltest_csp:
            log_message(f"  S{subject_id_str} (Test): Final CSP data shape {finaltest_csp['x'].shape}")
        else:
            log_message(f"  S{subject_id_str} (Test): Processing failed or no test trials.")

        return finalval_csp, finaltest_csp
    except Exception as e:
        log_message(f"ERROR S{subject_id_str} (Val/Test) in preprocess_validation_test_data_final: {e}\n{traceback.format_exc()}")
        return None, None

# --- CSP-SVM Functions ---
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
        current_trial_data = X1[:,:,trial] # (samples, channels)
        cov_trial = np.cov(current_trial_data, rowvar=False) # (channels, channels)
        if not np.all(np.isnan(cov_trial)):
            cov1 += cov_trial
            valid_trials_c1 +=1

    if valid_trials_c1 > 0: cov1 /= valid_trials_c1
    else: log_message("CSP Warning: All trials for class 1 had NaN covariance or 0 valid trials.")

    cov2 = np.zeros((n_channels, n_channels))
    valid_trials_c2 = 0
    for trial in range(X2.shape[2]):
        current_trial_data = X2[:,:,trial] # (samples, channels)
        cov_trial = np.cov(current_trial_data, rowvar=False) # (channels, channels)
        if not np.all(np.isnan(cov_trial)):
            cov2 += cov_trial
            valid_trials_c2 += 1
    if valid_trials_c2 > 0: cov2 /= valid_trials_c2
    else: log_message("CSP Warning: All trials for class 2 had NaN covariance or 0 valid trials.")

    epsilon_reg = 1e-9 # Regularization
    cov1_reg = cov1 + epsilon_reg * np.eye(n_channels)
    cov2_reg = cov2 + epsilon_reg * np.eye(n_channels)
    try:
        evals, evecs = eigh(cov1_reg, cov1_reg + cov2_reg)
    except np.linalg.LinAlgError:
        log_message("CSP Error: Generalized eigenvalue problem failed. Using regularized cov1.")
        evals, evecs = eigh(cov1_reg) # Fallback
    idx = np.argsort(evals)[::-1] # Sort in descending order
    evecs = evecs[:, idx]
    W = evecs.T # Spatial filters, rows are filters
    return W

def featcrossval(finaldataset, ChRanking, numchannel): # ChRanking seems unused
    a = finaldataset
    W = csp(a) # W has shape (num_filters, num_channels)
    if numchannel > 6: # numchannel is the actual number of channels in data
        if W.shape[0] < 6 : selectedw = W # If less than 6 filters, use all
        else: selectedw = np.vstack((W[0:3, :], W[-3:, :])) # First 3 and last 3 filters
    elif W.shape[0] >=2 : selectedw = np.vstack((W[0, :][np.newaxis,:], W[-1, :][np.newaxis,:])) # First and last
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
        variances = np.var(selectedZ, axis=0) # Variance along time axis for each CSP component
        epsilon_var = 1e-9
        variances_reg = variances + epsilon_var
        sum_variances_reg = np.sum(variances_reg)
        if sum_variances_reg < epsilon_var : producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_variances_reg)
    return producedfeatur, selectedw

def featcrostest(finaldataset, ChRanking, numchannel, selectedw): # ChRanking unused
    a = finaldataset
    if selectedw is None or selectedw.shape[0] == 0:
        log_message("featcrostest Warning: No spatial filters provided.")
        return {'x': np.zeros((0, a['x'].shape[2] if a['x'].ndim == 3 and a['x'].shape[2] > 0 else 0)),
                'y': finaldataset['y']}
    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]
    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}
    for trial in range(ntrial):
        projected_trial_data = finaldataset['x'][:, :, trial] # (samples, channels)
        selectedZ = np.dot(projected_trial_data, selectedw.T) # (samples, num_csp_filters)
        variances = np.var(selectedZ, axis=0)
        epsilon_var = 1e-9
        variances_reg = variances + epsilon_var
        sum_variances_reg = np.sum(variances_reg)
        if sum_variances_reg < epsilon_var: producedfeatur['x'][:, trial] = np.zeros(num_features)
        else: producedfeatur['x'][:, trial] = np.log(variances_reg / sum_variances_reg)
    return producedfeatur

def fitcsvm(X, Y, **kwargs): # X here is (trials, features)
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
        X_std[X_std < 1e-8] = 1e-8 # Avoid division by zero
        X_norm = (X - X_mean) / X_std
        model.X_mean_ = X_mean
        model.X_std_ = X_std
        model.fit(X_norm, Y)
    else:
        model.fit(X, Y)
    return model

def predict_svm(model, X): # X is (trials, features)
    if model is None: return np.array([])
    if X.shape[0] == 0: return np.array([])
    if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
        X_norm = (X - model.X_mean_) / model.X_std_
        y_pred = model.predict(X_norm)
    else:
        y_pred = model.predict(X)
    return y_pred

def train_cspsvm(data): # data['x'] is (samples, channels, trials)
    if data is None or 'x' not in data or data['x'].shape[2] == 0:
        log_message("CSP-SVM training skipped: Input data is invalid or has zero trials.")
        return None, None
    X_data_csp = data['x']
    n_selected_channels_csp = X_data_csp.shape[1]
    features_dict, spatial_filters = featcrossval(data, ChRanking=None, numchannel=n_selected_channels_csp)
    if features_dict['x'].size == 0 : return None, spatial_filters
    X_features_train = features_dict['x'].T # Transpose to (trials, features) for SVM
    y_labels_train = features_dict['y']
    if X_features_train.shape[0] == 0: return None, spatial_filters
    model = fitcsvm(X_features_train, y_labels_train, Standardize=True, KernelFunction='linear')
    return model, spatial_filters

def evaluate_cspsvm(model, spatial_filters, data_test):
    if model is None: return 0, np.zeros((2,2)), np.array([])
    if spatial_filters is None or spatial_filters.size == 0: return 0, np.zeros((2,2)), np.array([])
    if data_test is None or 'x' not in data_test or data_test['x'].shape[2] == 0:
        log_message("Evaluation skipped: Test data is invalid or has zero trials.")
        return 0, np.zeros((2,2)), np.array([])
    X_data_test_csp = data_test['x']
    n_selected_channels_test = X_data_test_csp.shape[1]
    features_test_dict = featcrostest(data_test, ChRanking=None, numchannel=n_selected_channels_test, selectedw=spatial_filters)
    if features_test_dict['x'].size == 0:
        y_true_test = data_test['y']
        cm = confusion_matrix(y_true_test, [], labels=[1,2]) if len(y_true_test) > 0 else np.zeros((2,2))
        return 0, cm, np.array([])
    X_features_test = features_test_dict['x'].T # (trials, features)
    y_true_test = features_test_dict['y']
    if X_features_test.shape[0] == 0:
        cm = confusion_matrix(y_true_test, [], labels=[1,2]) if len(y_true_test) > 0 else np.zeros((2,2))
        return 0, cm, np.array([])
    y_pred_test = predict_svm(model, X_features_test)
    accuracy = 0; cm = np.zeros((2,2))
    if len(y_true_test) > 0 and len(y_pred_test) > 0 and len(y_true_test) == len(y_pred_test):
        accuracy = np.mean(y_pred_test == y_true_test) * 100
        cm = confusion_matrix(y_true_test, y_pred_test, labels=[1, 2])
    elif len(y_true_test) > 0:
        cm = confusion_matrix(y_true_test, [], labels=[1,2])
    return accuracy, cm, y_pred_test

# --- Conditional GAN (cGAN) Functions ---
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

def build_critic_cgan(num_channels, time_points=500, num_classes=NUM_CLASSES_CGAN, embedding_dim=EMBEDDING_DIM_CGAN):
    data_input = layers.Input(shape=(num_channels, time_points), name="critic_data_input")
    label_input = layers.Input(shape=(1,), name="critic_label_input")
    label_embedding = layers.Embedding(num_classes, embedding_dim, name="critic_label_embedding")(label_input)
    x = layers.Permute((2, 1), name="critic_permute_in")(data_input) # (batch, samples, channels)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", name="critic_conv1d_1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="critic_leakyrelu_1")(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="same", name="critic_conv1d_2")(x)
    x = layers.LayerNormalization(name="critic_layernorm_1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="critic_leakyrelu_2")(x)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding="same", name="critic_conv1d_3")(x)
    x = layers.LayerNormalization(name="critic_layernorm_2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="critic_leakyrelu_3")(x)
    data_features_flat = layers.Flatten(name="critic_flatten_data")(x)
    label_embedding_flat = layers.Flatten(name="critic_flatten_label")(label_embedding)
    merged_features = layers.Concatenate(name="critic_merged_features")([data_features_flat, label_embedding_flat])
    output_score = layers.Dense(1, name="critic_dense_out")(merged_features)
    return models.Model([data_input, label_input], output_score, name="critic_cgan")

def frequency_domain_loss(real_data, generated_data):
    real_data_t = tf.transpose(real_data, perm=[0, 2, 1]) # (batch, time, chan)
    gen_data_t = tf.transpose(generated_data, perm=[0, 2, 1]) # (batch, time, chan)
    real_fft = tf.abs(tf.signal.rfft(real_data_t))
    gen_fft = tf.abs(tf.signal.rfft(gen_data_t))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))

def smooth_eeg(data, window_size=5): # data is (trials, channels, samples)
    smoothed_data = np.copy(data)
    if window_size % 2 == 0: window_size += 1
    if data.shape[2] <= window_size : # samples dim
        log_message(f"smooth_eeg: window_size {window_size} >= data length {data.shape[2]}. Skipping.")
        return data
    for i in range(data.shape[0]): # trials
        for j in range(data.shape[1]): # channels
            smoothed_data[i, j, :] = signal.savgol_filter(data[i, j, :], window_size, 2) # filter along samples
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
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32) # Map 1,2 to 0,1
    if not (np.all(labels_0_1 >= 0) and np.all(labels_0_1 < num_classes_gan)):
        log_message(f"ERROR cGAN S{subject_display_idx_param} Run {run_number}: Labels not 0-{num_classes_gan-1}. Found: {np.unique(labels_0_1)}")
        return None
    labels_0_1_reshaped = labels_0_1[:, np.newaxis]

    @tf.function
    def train_step_wgan_cgan(real_eeg_batch, real_labels_batch_0_1_tf, generator, critic, gen_opt, crit_opt, lambda_gp=10, latent_dim_cgan_step=LATENT_DIM_CGAN, n_critic=5, num_cls_step=NUM_CLASSES_CGAN):
        batch_size_tf = tf.shape(real_eeg_batch)[0]
        for _ in range(n_critic):
            noise = tf.random.normal([batch_size_tf, latent_dim_cgan_step])
            random_fake_labels_batch = tf.random.uniform(shape=[batch_size_tf, 1], minval=0, maxval=num_cls_step, dtype=tf.int32)
            with tf.GradientTape() as tape:
                fake_eeg = generator([noise, random_fake_labels_batch], training=True)
                real_output = critic([real_eeg_batch, real_labels_batch_0_1_tf], training=True)
                fake_output = critic([fake_eeg, random_fake_labels_batch], training=True)
                gp = gradient_penalty_cgan(critic, real_eeg_batch, fake_eeg, real_labels_batch_0_1_tf, lambda_gp)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
            gradients_d = tape.gradient(d_loss, critic.trainable_variables)
            crit_opt.apply_gradients(zip(gradients_d, critic.trainable_variables))
        noise_g = tf.random.normal([batch_size_tf, latent_dim_cgan_step])
        random_labels_for_g_train = tf.random.uniform(shape=[batch_size_tf, 1], minval=0, maxval=num_cls_step, dtype=tf.int32)
        with tf.GradientTape() as tape_g:
            fake_eeg_for_g = generator([noise_g, random_labels_for_g_train], training=True)
            fake_output_for_g = critic([fake_eeg_for_g, random_labels_for_g_train], training=True)
            g_loss_wasserstein = -tf.reduce_mean(fake_output_for_g)
        gradients_g = tape_g.gradient(g_loss_wasserstein, generator.trainable_variables) # Only Wasserstein for base G loss
        gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))
        return d_loss, g_loss_wasserstein, fake_eeg_for_g # Return base G loss for logging

    generator = build_generator_cgan(num_channels=num_gan_channels, time_points=time_points, latent_dim=latent_dim, num_classes=num_classes_gan, embedding_dim=embedding_dim_gan)
    critic = build_critic_cgan(num_channels=num_gan_channels, time_points=time_points, num_classes=num_classes_gan, embedding_dim=embedding_dim_gan)
    model_name = f"{model_name_prefix}_cGAN"
    log_message(f"\n--- Training {model_name} for S{subject_display_idx_param}, Run {run_number} ---")
    log_message(f"Generator input spec: {generator.input_spec}, output shape: {generator.output_shape}")
    log_message(f"Critic input spec: {critic.input_spec}, output shape: {critic.output_shape}")
    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    d_losses, g_losses = [], []
    if eeg_data.shape[0] == 0: return None
    if eeg_data.shape[0] < batch_size: batch_size = eeg_data.shape[0]
    num_batches = eeg_data.shape[0] // batch_size
    if num_batches == 0 and eeg_data.shape[0] > 0: num_batches = 1

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss_combined = 0.0, 0.0
        permuted_indices = np.random.permutation(eeg_data.shape[0])
        eeg_data_shuffled = eeg_data[permuted_indices]
        labels_0_1_shuffled = labels_0_1_reshaped[permuted_indices]
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size; end_idx = start_idx + batch_size
            real_eeg_batch_np = eeg_data_shuffled[start_idx:end_idx].astype(np.float32)
            real_labels_batch_0_1_np = labels_0_1_shuffled[start_idx:end_idx]
            if real_eeg_batch_np.shape[0] == 0: continue
            real_eeg_batch_tf = tf.convert_to_tensor(real_eeg_batch_np, dtype=tf.float32)
            real_labels_batch_0_1_tf = tf.convert_to_tensor(real_labels_batch_0_1_np, dtype=tf.int32)

            d_loss, g_loss_base, _ = train_step_wgan_cgan(
                real_eeg_batch_tf, real_labels_batch_0_1_tf, generator, critic, gen_opt, crit_opt,
                latent_dim_cgan_step=latent_dim, n_critic=n_critic_steps, num_cls_step=num_classes_gan
            )
            current_g_loss_total_for_batch = g_loss_base.numpy()
            if freq_loss_weight > 0:
                noise_for_freq = tf.random.normal([tf.shape(real_eeg_batch_tf)[0], latent_dim])
                labels_for_freq_fake = real_labels_batch_0_1_tf # Use real labels for comparable generation
                with tf.GradientTape() as freq_tape:
                    fake_data_for_freq_loss = generator([noise_for_freq, labels_for_freq_fake], training=True) # training=True
                    freq_loss_val = frequency_domain_loss(real_eeg_batch_tf, fake_data_for_freq_loss)
                    weighted_freq_loss = freq_loss_weight * freq_loss_val
                freq_gradients = freq_tape.gradient(weighted_freq_loss, generator.trainable_variables)
                if any(g is None for g in freq_gradients): log_message(f"Epoch {epoch} Warning: Freq_gradients None.")
                else: gen_opt.apply_gradients(zip(freq_gradients, generator.trainable_variables))
                current_g_loss_total_for_batch += weighted_freq_loss.numpy()
            epoch_d_loss += d_loss.numpy()
            epoch_g_loss_combined += current_g_loss_total_for_batch
        avg_epoch_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
        avg_epoch_g_loss = epoch_g_loss_combined / num_batches if num_batches > 0 else 0
        d_losses.append(avg_epoch_d_loss); g_losses.append(avg_epoch_g_loss)
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
    synthetic_data_np = synthetic_data.numpy() # (num_samples, num_channels, time_points)
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

# --- Plotting Functions (modified for clarity) ---
def compute_grand_average(data_plot, labels_plot, channel_idx_plot, class_labels_plot=[1, 2],
                          sampling_rate_plot=250, title_prefix_plot="", confidence_plot=0.95):
    results = {}
    labels_plot = np.asarray(labels_plot).squeeze()
    for class_label_plot in class_labels_plot:
        class_indices_plot = np.where(labels_plot == class_label_plot)[0]
        if len(class_indices_plot) == 0:
            log_message(f"    {title_prefix_plot} Data: No trials for class {class_label_plot} for GA, channel {channel_idx_plot}.")
            continue
        class_data_plot = data_plot[class_indices_plot, channel_idx_plot, :] # (trials, samples)
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
                                 augmented_data=None, augmented_labels=None): # MODIFIED: Added augmented data args
    """
    Plots a comparison of Grand Average ERPs for real, synthetic, and optionally, augmented data.
    """
    original_channel_number_plot = SELECTED_CHANNELS_1_BASED[channel_idx_plot]
    log_message(f"\n--- Plotting GA Comparison for Sel.ChIdx {channel_idx_plot} (OrigCh {original_channel_number_plot}) (S{subject_display_idx_param}) ---")

    # Compute grand average for each dataset
    real_results_plot = compute_grand_average(real_data_train_plot, real_labels_train_plot, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Real (data1.mat train)")
    synth_results_plot = compute_grand_average(synthetic_data_plot, synthetic_labels_plot, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Synthetic (cGAN)")

    # NEW: Compute grand average for augmented data if provided
    aug_results_plot = None
    if augmented_data is not None and augmented_labels is not None:
        aug_results_plot = compute_grand_average(augmented_data, augmented_labels, channel_idx_plot, class_labels_plot, sampling_rate_plot, "Augmented (100%)")

    time_points_count_plot = real_data_train_plot.shape[2]
    time_vector_plot = np.arange(time_points_count_plot) / sampling_rate_plot

    for class_label_plot in class_labels_plot:
        class_key_plot = f"class_{class_label_plot}"
        real_class_result_plot = real_results_plot.get(class_key_plot)
        synth_class_result_plot = synth_results_plot.get(class_key_plot)
        # NEW: Get augmented data result
        aug_class_result_plot = aug_results_plot.get(class_key_plot) if aug_results_plot else None

        plt.figure(figsize=(12, 6))
        plot_successful_ga = False

        # Plot Real Data
        if real_class_result_plot:
            plt.plot(time_vector_plot, real_class_result_plot["grand_avg"], 'b-', linewidth=2, label=f'Real (n={real_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, real_class_result_plot["lower_ci"], real_class_result_plot["upper_ci"], color='blue', alpha=0.2)
            plot_successful_ga = True

        # Plot Synthetic Data
        if synth_class_result_plot:
            plt.plot(time_vector_plot, synth_class_result_plot["grand_avg"], 'r-', linewidth=2, label=f'Synthetic (n={synth_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, synth_class_result_plot["lower_ci"], synth_class_result_plot["upper_ci"], color='red', alpha=0.2)
            plot_successful_ga = True

        # NEW: Plot Augmented Data
        if aug_class_result_plot:
            plt.plot(time_vector_plot, aug_class_result_plot["grand_avg"], 'g-', linewidth=2, label=f'Augmented (n={aug_class_result_plot["n_trials"]})')
            plt.fill_between(time_vector_plot, aug_class_result_plot["lower_ci"], aug_class_result_plot["upper_ci"], color='green', alpha=0.2)
            plot_successful_ga = True

        if plot_successful_ga and output_dir_param and subject_display_idx_param:
            plt.title(f"GA Comparison - Cls {class_label_plot} - Sel.ChIdx {channel_idx_plot} (OrigCh {original_channel_number_plot}) - S{subject_display_idx_param}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (Norm)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_filename = os.path.join(output_dir_param, f"Grand_Avg_Cls{class_label_plot}_SelChIdx{channel_idx_plot}_OrigCh{original_channel_number_plot}_S{subject_display_idx_param}.png")
            plt.savefig(plot_filename)
            plt.close()

            # MODIFIED: Log statistics for all pairs
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
                        augmented_data=None, augmented_labels=None): # MODIFIED: Added augmented data args
    """
    Plots a comparison of Power Spectral Density (PSD) for real, synthetic, and optionally, augmented data.
    """
    original_channel_number_psd = SELECTED_CHANNELS_1_BASED[channel_idx_psd]
    log_message(f"\n--- PSD ANALYSIS for Sel.ChIdx {channel_idx_psd} (OrigCh {original_channel_number_psd}) (S{subject_display_idx_param}) ---")

    for class_label_psd in class_labels_psd:
        plt.figure(figsize=(12, 6))
        common_freq_axis_psd = None
        plot_successful_psd_flag = False
        avg_real_psd_val, avg_synth_psd_val, avg_aug_psd_val = (None, None, None)

        # --- Process Real Data ---
        labels_real_psd = np.asarray(real_labels_train_psd).squeeze()
        real_indices_psd = np.where(labels_real_psd == class_label_psd)[0]
        if len(real_indices_psd) > 0:
            real_trials_ch_psd = real_data_train_psd[real_indices_psd, channel_idx_psd, :]
            real_psds_list_psd = []
            current_common_freq_axis_real_psd = None
            for i in range(real_trials_ch_psd.shape[0]):
                nperseg_val = min(256, real_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, real_trials_ch_psd.shape[1] // 2 -1)
                if nperseg_val == 0 or noverlap_val < 0 or noverlap_val >= nperseg_val : continue
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

        # --- Process Synthetic Data ---
        labels_synth_psd = np.asarray(synthetic_labels_psd).squeeze()
        synth_indices_psd = np.where(labels_synth_psd == class_label_psd)[0]
        if len(synth_indices_psd) > 0:
            synth_trials_ch_psd = synthetic_data_psd[synth_indices_psd, channel_idx_psd, :]
            synth_psds_list_psd = []
            for i in range(synth_trials_ch_psd.shape[0]):
                nperseg_val = min(256, synth_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, synth_trials_ch_psd.shape[1] // 2 -1)
                if nperseg_val == 0 or noverlap_val < 0 or noverlap_val >= nperseg_val: continue
                f_psd, psd_val = signal.welch(synth_trials_ch_psd[i,:], fs=sampling_rate_psd, nperseg=nperseg_val, noverlap=noverlap_val)
                if common_freq_axis_psd is None: continue # Cannot proceed without a reference frequency axis
                if len(f_psd) == len(common_freq_axis_psd) and np.allclose(f_psd, common_freq_axis_psd): synth_psds_list_psd.append(psd_val)
                else: synth_psds_list_psd.append(np.interp(common_freq_axis_psd, f_psd, psd_val))
            if synth_psds_list_psd and common_freq_axis_psd is not None:
                avg_synth_psd_val = np.mean(np.array(synth_psds_list_psd), axis=0); sem_synth_psd_val = stats.sem(np.array(synth_psds_list_psd), axis=0, nan_policy='omit')
                freq_mask_psd = common_freq_axis_psd <= 50
                plt.plot(common_freq_axis_psd[freq_mask_psd], avg_synth_psd_val[freq_mask_psd], 'r-', linewidth=2, label=f'Synthetic (n={len(synth_indices_psd)})')
                plt.fill_between(common_freq_axis_psd[freq_mask_psd], np.nan_to_num(avg_synth_psd_val[freq_mask_psd] - 1.96 * sem_synth_psd_val[freq_mask_psd]), np.nan_to_num(avg_synth_psd_val[freq_mask_psd] + 1.96 * sem_synth_psd_val[freq_mask_psd]), color='red', alpha=0.2)
                plot_successful_psd_flag = True

        # --- NEW: Process Augmented Data ---
        if augmented_data is not None and augmented_labels is not None:
            labels_aug_psd = np.asarray(augmented_labels).squeeze()
            aug_indices_psd = np.where(labels_aug_psd == class_label_psd)[0]
            if len(aug_indices_psd) > 0:
                aug_trials_ch_psd = augmented_data[aug_indices_psd, channel_idx_psd, :]
                aug_psds_list_psd = []
                for i in range(aug_trials_ch_psd.shape[0]):
                    nperseg_val = min(256, aug_trials_ch_psd.shape[1]); noverlap_val = min(nperseg_val // 2, aug_trials_ch_psd.shape[1] // 2 - 1)
                    if nperseg_val == 0 or noverlap_val < 0 or noverlap_val >= nperseg_val: continue
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

        # --- Finalize and Save Plot ---
        if plot_successful_psd_flag and common_freq_axis_psd is not None and output_dir_param and subject_display_idx_param:
            plt.xlabel('Freq (Hz)')
            plt.ylabel('PSD (µV²/Hz approx.)')
            plt.title(f'PSD Comparison - Cls {class_label_psd}, Sel.ChIdx {channel_idx_psd} (OrigCh {original_channel_number_psd}) - S{subject_display_idx_param}')
            plt.legend()
            plt.grid(True)
            plt.xlim(0, 50)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plot_filename = os.path.join(output_dir_param, f"PSD_Cls{class_label_psd}_SelChIdx{channel_idx_psd}_OrigCh{original_channel_number_psd}_S{subject_display_idx_param}.png")
            plt.savefig(plot_filename)
            plt.close()
        else:
            log_message(f"  No data to plot PSD for Cls {class_label_psd}. Skipping.")
            plt.close()

def compute_average_psd(data, labels, class_labels=[1, 2], sampling_rate=250):
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


if __name__ == '__main__':
    # --- Configuration for subject processing ---
    TARGET_SUBJECTS = "all" # "all", subject_id (1-based int), or [id1, id2, ...] (1-based list)

    all_subject_indices_available = list(range(NUM_SUBJECTS_TRAIN))
    subjects_to_process_indices = [] # 0-based indices

    if isinstance(TARGET_SUBJECTS, str) and TARGET_SUBJECTS.lower() == "all":
        subjects_to_process_indices = all_subject_indices_available
    elif isinstance(TARGET_SUBJECTS, int):
        if 1 <= TARGET_SUBJECTS <= NUM_SUBJECTS_TRAIN:
            subjects_to_process_indices = [TARGET_SUBJECTS - 1]
        else: 
            print(f"WARNING: Subject ID {TARGET_SUBJECTS} out of range (1-{NUM_SUBJECTS_TRAIN}). Defaulting to all.")
            subjects_to_process_indices = all_subject_indices_available
    elif isinstance(TARGET_SUBJECTS, list) and all(isinstance(i, int) for i in TARGET_SUBJECTS):
        valid_ids_0_based = [s_id - 1 for s_id in TARGET_SUBJECTS if 1 <= s_id <= NUM_SUBJECTS_TRAIN]
        if valid_ids_0_based:
            subjects_to_process_indices = sorted(list(set(valid_ids_0_based)))
        else: 
            print(f"WARNING: No valid subject IDs in {TARGET_SUBJECTS} (range 1-{NUM_SUBJECTS_TRAIN}). Defaulting to all.")
            subjects_to_process_indices = all_subject_indices_available
    else:
        print(f"WARNING: Invalid TARGET_SUBJECTS format ('{TARGET_SUBJECTS}'). Defaulting to all.")
        subjects_to_process_indices = all_subject_indices_available

    if not subjects_to_process_indices:
        print("CRITICAL: No subjects selected for processing. Exiting.")
        exit()
    else:
        print(f"INFO: Will process {len(subjects_to_process_indices)} subject(s). 1-based IDs: {[i+1 for i in subjects_to_process_indices]}")

    print(f"Global random seed set to: {SEED_VALUE}")

    for subject_idx in subjects_to_process_indices:
        current_subject_display_id = subject_idx + 1
        output_dir = f"Subject_{current_subject_display_id}_D1T_50trials"
        os.makedirs(output_dir, exist_ok=True)
        current_log_file_path = os.path.join(output_dir, "run_log.txt")
        with open(current_log_file_path, 'w') as f:
            f.write(f"Log for Subject {current_subject_display_id} (0-indexed: {subject_idx})\n")
            f.write(f"Output directory: {output_dir}\nGlobal seed: {SEED_VALUE}\nSelected channels (0-based): {SELECTED_CHANNELS_0_BASED}\n")
            f.write(f"Train trials/class: {NUM_TRAIN_TRIALS_PER_CLASS}, Valid trials/class: {NUM_VALID_TRIALS_PER_CLASS}, GAN Runs: {NUM_RUNS_PER_SUBJECT}\n")
            f.write("="*70 + "\n\n")

        log_message(f"PROCESSING SUBJECT {current_subject_display_id}/{NUM_SUBJECTS_TRAIN} (Selected index {subject_idx})")

        try:
            xsubi_train_struct = xsubi_all_train_source[0, subject_idx]
            xsubi_val_test_struct = xsubi_all_val_test_source[0, subject_idx] if subject_idx < NUM_SUBJECTS_VAL_TEST else None

            log_message(f"\n--- Step 1: Data Preprocessing & Splitting ---")
            np.random.seed(SEED_VALUE)

            x_train_gan_fmt, y_train_labels, final_train_csp, train_min_global_stat, train_max_global_stat = \
                preprocess_training_data_final(
                    xsubi_train_struct, str(current_subject_display_id),
                    NUM_TRAIN_TRIALS_PER_CLASS, SELECTED_CHANNELS_0_BASED
                )

            if x_train_gan_fmt is None:
                log_message(f"CRITICAL: Training data preprocessing failed for S{current_subject_display_id}. Skipping."); continue

            # Save the complete training set
            training_set_filename = os.path.join(output_dir, f"training_{current_subject_display_id}.mat")
            scipy.io.savemat(training_set_filename, {
                'training_set_x': final_train_csp['x'],
                'training_set_y': final_train_csp['y']
            })
            log_message(f"Saved complete training set to {training_set_filename}")

            final_valid_csp, final_test_csp = (None, None)
            if xsubi_val_test_struct is not None:
                final_valid_csp, final_test_csp = preprocess_validation_test_data_final(
                    xsubi_val_test_struct, str(current_subject_display_id),
                    NUM_VALID_TRIALS_PER_CLASS, train_min_global_stat, train_max_global_stat,
                    SELECTED_CHANNELS_0_BASED
                )
            if final_valid_csp is None or final_test_csp is None:
                log_message(f"CRITICAL: Validation or Test data preprocessing failed for S{current_subject_display_id}. Skipping."); continue

            log_message(f"\n--- Step 2: Baseline Evaluation ---")
            train_valid_csp = {
                'x': np.concatenate((final_train_csp['x'], final_valid_csp['x']), axis=2),
                'y': np.concatenate((final_train_csp['y'], final_valid_csp['y']))
            }
            model_baseline, sf_baseline = train_cspsvm(train_valid_csp)
            accuracy_baseline, _, _ = evaluate_cspsvm(model_baseline, sf_baseline, final_test_csp)
            log_message(f"Baseline Accuracy (Train+Valid -> Test): {accuracy_baseline:.2f}%")

            log_message(f"\n--- Step 3: {NUM_RUNS_PER_SUBJECT} cGAN RUNS & BATCH SELECTION ---")
            all_generated_batches = []
            for run_idx in range(NUM_RUNS_PER_SUBJECT):
                run_number = run_idx + 1
                cgan_generator_trained = train_wgan_gp_cgan(
                    eeg_data=x_train_gan_fmt, eeg_labels=y_train_labels,
                    epochs=1, batch_size=min(64, x_train_gan_fmt.shape[0]),
                    model_name_prefix='MI', num_gan_channels=NUM_SELECTED_CHANNELS,
                    time_points=x_train_gan_fmt.shape[2], freq_loss_weight=1.5,
                    output_dir_param=output_dir, subject_display_idx_param=current_subject_display_id,
                    run_number=run_number
                )
                if not cgan_generator_trained: continue

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
                log_message("CRITICAL: No synthetic batches generated.")
                continue

            all_generated_batches.sort(key=lambda x: x['selection_score'], reverse=True)
            top_10_batches = all_generated_batches[:10]
            log_message(f"Selected top {len(top_10_batches)} batches based on GA similarity.")

            all_evaluated_strategies = []
            mix_ratios = [0, 0.25, 0.50, 1.0]

            log_message("\n--- Evaluating strategies on the validation set ---")
            for batch_info in top_10_batches:
                for ratio in mix_ratios:
                    if ratio == 0:
                        train_data_strat = batch_info['synthetic_features']
                        strategy_name = 'Synth Only'
                    else:
                        strategy_name = f"Augmented ({int(ratio*100)}%)"
                        num_real_trials = final_train_csp['x'].shape[2]
                        num_synth_to_add = int(num_real_trials * ratio)
                        synth_x_to_add = batch_info['synthetic_features']['x'][:, :, :num_synth_to_add]
                        synth_y_to_add = batch_info['synthetic_features']['y'][:num_synth_to_add]
                        aug_x = np.concatenate((final_train_csp['x'], synth_x_to_add), axis=2)
                        aug_y = np.concatenate((final_train_csp['y'], synth_y_to_add))
                        train_data_strat = {'x': aug_x, 'y': aug_y}

                    model_strategy, sf_strategy = train_cspsvm(train_data_strat)
                    validation_accuracy, _, _ = evaluate_cspsvm(model_strategy, sf_strategy, final_valid_csp)
                    log_message(f"    Run {batch_info['run_index']}, Batch {batch_info['batch_index']}, Strategy '{strategy_name}': Validation Acc: {validation_accuracy:.2f}%")
                    all_evaluated_strategies.append({
                        'validation_accuracy': validation_accuracy, **batch_info, 'mix_ratio': ratio
                    })


            if not all_evaluated_strategies:
                log_message("CRITICAL: No strategies evaluated.")
                continue

            max_val_acc = max(s['validation_accuracy'] for s in all_evaluated_strategies)
            top_strategies = [s for s in all_evaluated_strategies if s['validation_accuracy'] == max_val_acc]

            if len(top_strategies) > 1:
                log_message(f"\nFound {len(top_strategies)} strategies with top validation accuracy of {max_val_acc:.2f}%. Using GA score as tie-breaker.")
                top_strategies.sort(key=lambda x: x['selection_score'], reverse=True)
            
            best_strategy_details = top_strategies[0]
            best_strategy_name = 'Synth Only' if best_strategy_details['mix_ratio'] == 0 else f"Augmented ({int(best_strategy_details['mix_ratio']*100)}%)"
            log_message(f"\nSelected best strategy: {best_strategy_name} from Run {best_strategy_details['run_index']}, Batch {best_strategy_details['batch_index']} (Val Acc: {best_strategy_details['validation_accuracy']:.2f}%, GA Score: {best_strategy_details['selection_score']:.4f})")

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
            synth_x_to_add_plot = best_synthetic_batch_gan[:num_real_trials_plot, :, :]
            synth_y_to_add_plot = best_synthetic_labels[:num_real_trials_plot]
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
                        sampling_rate_plot=250,
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
                        sampling_rate_psd=250,
                        output_dir_param=output_dir,
                        subject_display_idx_param=current_subject_display_id,
                        augmented_data=augmented_x_plot,
                        augmented_labels=augmented_y_plot
                    )

            log_message("\n\n--- Step 5: Final Evaluation on Test Set ---")
            synth_only_strategies = [s for s in all_evaluated_strategies if s['mix_ratio'] == 0]
            accuracy_best_synth_on_test = 0.0
            if synth_only_strategies:
                best_synth_only_strategy = max(synth_only_strategies, key=lambda x: x['validation_accuracy'])
                model_best_synth, sf_best_synth = train_cspsvm(best_synth_only_strategy['synthetic_features'])
                accuracy_best_synth_on_test, _, _ = evaluate_cspsvm(model_best_synth, sf_best_synth, final_test_csp)
            log_message(f"BEST SYNTHETIC-ONLY (Val Acc: {best_synth_only_strategy['validation_accuracy']:.2f}%) -> REAL test accuracy: {accuracy_best_synth_on_test:.2f}%")

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
            accuracy_best_strategy_on_test, _, _ = evaluate_cspsvm(model_final, sf_final, final_test_csp)
            log_message(f"BEST OVERALL STRATEGY ({best_strategy_name}) -> REAL test accuracy: {accuracy_best_strategy_on_test:.2f}%")

            log_message(f"\n\n=== FINAL SUMMARY FOR S{current_subject_display_id} ===")
            log_message(f"Baseline Accuracy (Train+Valid -> Test): {accuracy_baseline:.2f}%")
            log_message(f"Best Synthetic-Only Accuracy (Synth Train -> Test): {accuracy_best_synth_on_test:.2f}%")
            if best_strategy_details['mix_ratio'] != 0:
                log_message(f"Best Augmented Strategy ({best_strategy_name}) Accuracy (Augmented Train+Valid -> Test): {accuracy_best_strategy_on_test:.2f}%")

            plot_final_accuracies(
                baseline_acc=accuracy_baseline,
                synth_only_acc=accuracy_best_synth_on_test,
                best_aug_acc=accuracy_best_strategy_on_test,
                subject_id=current_subject_display_id,
                output_dir=output_dir,
                dataset_name="D2T_50trials"
            )

        except Exception as e_main:
            log_message(f"CRITICAL ERROR S{current_subject_display_id}: {e_main}\n{traceback.format_exc()}")
            log_message(f"--- Subject {current_subject_display_id} processing FAILED. ---")

    print("\n--- All selected subjects processed. Check individual log files and output directories. ---")
