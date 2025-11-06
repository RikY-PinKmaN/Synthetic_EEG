import h5py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings
import traceback

warnings.filterwarnings('ignore')

# --- Helper Functions (Unchanged) ---

def reshape_for_svm(eeg_data_3d):
    """
    Reshapes EEG data for classic ML models like SVM.
    Expects data in [samples, channels, trials] format.
    """
    if eeg_data_3d.ndim != 3:
        raise ValueError("Input data must be 3D [samples, channels, trials]")

    n_samples, n_channels, n_trials = eeg_data_3d.shape
    data_transposed = np.transpose(eeg_data_3d, (2, 0, 1))
    n_features = n_samples * n_channels
    data_reshaped = data_transposed.reshape(n_trials, n_features)
    return data_reshaped

def fit_svm(X, Y):
    """
    Fits an SVM model with data standardization.
    """
    model = SVC(kernel='linear', C=1.0)
    model.X_mean = np.mean(X, axis=0)
    model.X_std = np.std(X, axis=0)
    model.X_std[model.X_std < 1e-8] = 1.0
    X_norm = (X - model.X_mean) / model.X_std
    model.fit(X_norm, Y)
    return model

def predict_svm(model, X):
    """Predicts labels using the trained SVM model and its standardization parameters."""
    if X.shape[0] == 0: return np.array([])
    X_norm = (X - model.X_mean) / model.X_std
    return model.predict(X_norm)

# --- Main Parameters ---
feature_mat_path = "P300_feature.mat"
feature_var_name = "p300_feature"
label_mat_path = "P300_label.mat"
label_var_name = "p300_labels"

num_train_trials_total = 30
# SEED_VALUE = 52
# np.random.seed(SEED_VALUE)

# --- Main Script Logic ---
print(f"{'='*10} P300 Classification Run {'='*10}")

try:
    # --- Step 1: Load Data and Labels using h5py ---
    print(f"Loading features from '{feature_mat_path}' (variable: '{feature_var_name}') using h5py")
    with h5py.File(feature_mat_path, 'r') as f:
        all_x_raw = f[feature_var_name][:]

    print(f"Loading labels from '{label_mat_path}' (variable: '{label_var_name}') using h5py")
    with h5py.File(label_mat_path, 'r') as f:
        all_y_raw = f[label_var_name][:]

    # --- Step 2: Inspect and Preprocess the Data ---
    print(f"Original data shape from .mat file: {all_x_raw.shape}")

    # **MODIFICATION:** Correct the transpose operation based on the observed shape.
    # The logs showed the input shape is (90, 28, 8) -> [Trials, Timepoints, Channels].
    # We need to convert it to the script's expected format: [Timepoints, Channels, Trials].
    # The correct transpose is (1, 2, 0).
    processed_x = np.transpose(all_x_raw, (1, 2, 0))
    print(f"Transposed data shape for Python: {processed_x.shape} [Timepoints, Channels, Trials]")
    
    # Ensure labels are a 1D array
    all_y = all_y_raw.flatten()
    print(f"Labels shape: {all_y.shape}")

    # --- Step 3: Train/Test Split ---
    num_total_trials = processed_x.shape[2]
    if num_total_trials <= num_train_trials_total:
        raise ValueError(f"Total trials ({num_total_trials}) is not greater than the number of training trials ({num_train_trials_total}).")

    print(f"\nSplitting data: {num_train_trials_total} for training, {num_total_trials - num_train_trials_total} for testing.")
    rand_indices = np.random.permutation(num_total_trials)
    train_idx = rand_indices[:num_train_trials_total]
    test_idx = rand_indices[num_train_trials_total:]

    x_train_3d = processed_x[:, :, train_idx]
    y_train = all_y[train_idx]
    x_test_3d = processed_x[:, :, test_idx]
    y_test = all_y[test_idx]
    
    print(f"Split complete. Train trials: {len(y_train)}, Test trials: {len(y_test)}")

    # --- Step 4: P300 Feature Extraction (Reshaping) ---
    print("\nReshaping data for SVM...")
    train_features = reshape_for_svm(x_train_3d)
    test_features = reshape_for_svm(x_test_3d)
    print(f"  Training features shape: {train_features.shape} [Trials, Features]")
    print(f"  Testing features shape:  {test_features.shape} [Trials, Features]")

    # --- Step 5: Train the SVM Classifier ---
    print("\nTraining the SVM model...")
    model = fit_svm(train_features, y_train)
    print("Training complete.")

    # --- Step 6: Evaluate the Model ---
    print("\nEvaluating model on the test set...")
    predictions = predict_svm(model, test_features)
    
    accuracy = np.mean(predictions == y_test) * 100
    unique_labels = np.unique(np.concatenate((y_test, predictions)))
    cm = confusion_matrix(y_test, predictions, labels=unique_labels)
    
    print(f"\n{'='*10} Results {'='*10}")
    print(f"Test Accuracy = {accuracy:.2f}%")
    print(f"Confusion Matrix (Labels: {unique_labels}):\n{cm}")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find a file. Please check paths. Details: {e}")
except KeyError as e:
    print(f"FATAL ERROR: Variable name '{e}' not found in .mat file. Please check the `feature_var_name` and `label_var_name` variables in the script.")
except Exception as e:
    print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
    traceback.print_exc()