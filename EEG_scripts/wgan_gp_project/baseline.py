import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings
import traceback

warnings.filterwarnings('ignore')

# --- Helper Functions ---

def butterworth_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def csp(data):
    class1_idx = np.where(data['y'] == 1)[0]
    class2_idx = np.where(data['y'] == 2)[0]
    if len(class1_idx) == 0 or len(class2_idx) == 0:
        raise ValueError("CSP requires data from both classes.")
    
    X1, X2 = data['x'][:, :, class1_idx], data['x'][:, :, class2_idx]
    n_channels = X1.shape[1]
    
    cov1 = sum(np.dot(trial.T, trial) / np.trace(np.dot(trial.T, trial)) for trial in np.rollaxis(X1, 2)) / len(class1_idx)
    cov2 = sum(np.dot(trial.T, trial) / np.trace(np.dot(trial.T, trial)) for trial in np.rollaxis(X2, 2)) / len(class2_idx)
    
    cov_sum = cov1 + cov2
    
    try:
        evals, evecs = eigh(cov1, cov_sum)
    except np.linalg.LinAlgError:
        print("  CSP Warning: Generalized eigenvalue problem failed. Using regularized fallback.")
        evals, evecs = eigh(cov1 + 1e-6 * np.eye(n_channels), cov_sum + 1e-6 * np.eye(n_channels))
        
    return evecs[:, np.argsort(evals)[::-1]].T

# --- CORRECTED FUNCTION ---
def apply_logvar_features(dataset, csp_filters):
    """
    Applies CSP filters and calculates the log-variance features.
    The single-trial data is transposed to match matrix dimensions.
    """
    features = np.zeros((dataset['x'].shape[2], csp_filters.shape[0]))
    for i in range(dataset['x'].shape[2]):
        # Transpose the single trial data from (samples, channels) to (channels, samples)
        single_trial_data = dataset['x'][:, :, i].T
        projected = np.dot(csp_filters, single_trial_data)
        
        variances = np.var(projected, axis=1)
        sum_variances = np.sum(variances)
        if sum_variances > 1e-9:
            features[i, :] = np.log(variances / sum_variances)
    return features

def fit_svm(X, Y):
    model = SVC(kernel='linear', C=1.0)
    model.X_mean = np.mean(X, axis=0)
    model.X_std = np.std(X, axis=0)
    model.X_std[model.X_std < 1e-8] = 1.0 # Avoid division by zero
    X_norm = (X - model.X_mean) / model.X_std
    model.fit(X_norm, Y)
    return model

def predict_svm(model, X):
    if X.shape[0] == 0: return np.array([])
    X_norm = (X - model.X_mean) / model.X_std
    return model.predict(X_norm)

def _get_eeg_data_from_field(subject_struct, field_name):
    try:
        field_data = subject_struct[field_name]
        while isinstance(field_data, np.ndarray) and field_data.shape == (1, 1):
            field_data = field_data[0, 0]

        if not isinstance(field_data, np.ndarray):
            raise TypeError(f"Content of '{field_name}' is not a NumPy array.")

        if field_name == 'y':
            return field_data.flatten()
        elif field_name == 'x':
            if field_data.ndim != 3:
                raise ValueError(f"Field '{field_name}' is not 3D. Shape: {field_data.shape}")
            return field_data
        return field_data

    except (KeyError, TypeError, ValueError, IndexError) as e:
        print(f"  ERROR processing field '{field_name}': {e}")
        return None

# --- Main Parameters ---
mat_file_path = "DATA1.mat"
mat_variable_name = 'xsubi_all'
num_train_trials_per_class = 40
lowfreq, highfreq, fs = 8, 30, 512
time_trim_seconds = 0.5

# --- MODIFICATION: Use 1-based indexing for channels ---
# Define the specific channels to be used using 1-based indexing for convenience.
selected_channels_1based = [14, 13, 12, 48, 49, 50, 51, 17, 18, 19, 56, 54, 55]
# Convert the 1-based list to 0-based indices for use with Python/NumPy.
selected_channels = [ch - 1 for ch in selected_channels_1based]
# --- END MODIFICATION ---

# --- Set Random Seed ---
SEED_VALUE = 52  # Example seed value
np.random.seed(SEED_VALUE)

try:
    all_subjects_data = scipy.io.loadmat(mat_file_path)[mat_variable_name]
    print(f"Data loaded from '{mat_file_path}' (variable: '{mat_variable_name}')")
except (FileNotFoundError, KeyError) as e:
    print(f"FATAL: Could not load data. {e}")
    exit()

num_subjects = all_subjects_data.shape[1]
print(f"Found {num_subjects} subjects.")
subject_accuracies = np.full(num_subjects, np.nan)

for subi in range(num_subjects):
    subject_id = subi + 1
    print(f"\n{'='*10} Processing Subject {subject_id}/{num_subjects} {'='*10}")

    try:
        subject_raw_data = all_subjects_data[0, subi]
        all_x = _get_eeg_data_from_field(subject_raw_data, 'x')
        all_y = _get_eeg_data_from_field(subject_raw_data, 'y')

        if all_x is None or all_y is None:
            print(f"  Skipping subject {subject_id} due to data loading failure.")
            continue
        print(f"  Loaded S{subject_id}: x_shape={all_x.shape}, y_shape={all_y.shape}")
        
        # Select specific channels using the 0-based index list
        print(f"  Selecting {len(selected_channels)} specific channels.")
        all_x = all_x[:, selected_channels, :]
        print(f"  New x_shape after channel selection: {all_x.shape}")

        # Preprocessing
        filtered_x = butterworth_filter(all_x, lowfreq, highfreq, fs)
        start_sample = int(fs * time_trim_seconds)
        end_sample = -start_sample if start_sample > 0 else filtered_x.shape[0]
        if start_sample * 2 >= filtered_x.shape[0]:
            raise ValueError("Trial length is too short for the specified time window.")
        processed_x = filtered_x[start_sample:end_sample, :, :]

        # Train/Test Split
        idx_c1, idx_c2 = np.where(all_y == 1)[0], np.where(all_y == 2)[0]
        if len(idx_c1) < num_train_trials_per_class or len(idx_c2) < num_train_trials_per_class:
            raise ValueError("Not enough trials per class for the split.")
        
        np.random.shuffle(idx_c1); np.random.shuffle(idx_c2)
        train_idx = np.concatenate((idx_c1[:num_train_trials_per_class], idx_c2[:num_train_trials_per_class]))
        test_idx = np.setdiff1d(np.arange(len(all_y)), train_idx)
        
        train_data = {'x': processed_x[:, :, train_idx], 'y': all_y[train_idx]}
        test_data = {'x': processed_x[:, :, test_idx], 'y': all_y[test_idx]}
        print(f"  Split complete. Train: {len(train_data['y'])}, Test: {len(test_data['y'])}")

        # Training
        W = csp(train_data)
        csp_filters = np.vstack((W[:3], W[-3:])) if W.shape[0] >= 6 else W
        
        train_features = apply_logvar_features(train_data, csp_filters)
        model = fit_svm(train_features, train_data['y'])

        # Evaluation
        test_features = apply_logvar_features(test_data, csp_filters)
        predictions = predict_svm(model, test_features)
        
        accuracy = np.mean(predictions == test_data['y']) * 100
        cm = confusion_matrix(test_data['y'], predictions, labels=[1, 2])
        
        subject_accuracies[subi] = accuracy
        print(f"  Subject {subject_id}: Test Accuracy = {accuracy:.2f}%")
        print(f"  Confusion Matrix:\n{cm}")

    except Exception as e:
        print(f"  UNEXPECTED ERROR for Subject {subject_id}: {e}")
        traceback.print_exc()

# --- Final Results ---
print(f"\n{'='*10} Processing Complete {'='*10}")
valid_accuracies = subject_accuracies[~np.isnan(subject_accuracies)]
if len(valid_accuracies) > 0:
    print(f"Overall Average Accuracy across {len(valid_accuracies)} subjects: {np.mean(valid_accuracies):.2f}%")
    print(f"Standard Deviation of Accuracy: {np.std(valid_accuracies):.2f}%")
else:
    print("\nNo subjects were processed successfully.")