import numpy as np
from scipy.signal import ellip, ellipord, filtfilt
import src.config as cfg

# Bandpass filter definition using elliptical filter
def elliptical_filter(data, lowcut=cfg.LOWCUT, highcut=cfg.HIGHCUT, fs=cfg.SAMPLING_RATE, rp=cfg.FILTER_RIPPLE, rs=cfg.FILTER_ATTENUATION):
    nyq = 0.5 * fs
    wp = [lowcut / nyq, highcut / nyq]  # Passband
    ws = [(lowcut - 1) / nyq, (highcut + 1) / nyq]  # Stopband
    n, wn = ellipord(wp, ws, rp, rs)  # Filter order and natural frequency
    b, a = ellip(n, rp, rs, wn, btype='band')  # Coefficients
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

# Function to preprocess and split data
def preprocess_and_split_data(x_data, y_data, train_trials=cfg.TRAIN_TRIALS, val_trials=cfg.VAL_TRIALS, lowcut=cfg.LOWCUT, highcut=cfg.HIGHCUT, fs=cfg.SAMPLING_RATE):
    # Separate data by class
    right_data = x_data[:, :, y_data.ravel() == 2]
    left_data = x_data[:, :, y_data.ravel() == 1]
    
    # Print total available trials per class
    print(f"Available left hand trials: {left_data.shape[2]}")
    print(f"Available right hand trials: {right_data.shape[2]}")

    # Ensure we only use up to 500 time points
    right_data = right_data[115:615, :, :]
    left_data = left_data[115:615, :, :]

    # Training data (first 20 trials per class)
    train_right = right_data[:, :, :train_trials]
    train_left = left_data[:, :, :train_trials]

    # Validation data (next 5 trials per class)
    val_right = right_data[:, :, train_trials:train_trials+val_trials]
    val_left = left_data[:, :, train_trials:train_trials+val_trials]

    # Test data (remaining trials per class)
    test_right = right_data[:, :, train_trials+val_trials:]
    test_left = left_data[:, :, train_trials+val_trials:]

    # Combine data for training, validation, and testing
    x_train = np.concatenate([train_right, train_left], axis=2)
    y_train = np.concatenate([np.zeros(train_right.shape[2]) + 2, np.zeros(train_left.shape[2]) + 1])

    x_val = np.concatenate([val_right, val_left], axis=2)
    y_val = np.concatenate([np.zeros(val_right.shape[2]) + 2, np.zeros(val_left.shape[2]) + 1])

    x_test = np.concatenate([test_right, test_left], axis=2)
    y_test = np.concatenate([np.zeros(test_right.shape[2]) + 2, np.zeros(test_left.shape[2]) + 1])

    # Apply elliptical filter to all datasets
    process_data = lambda data: np.array([elliptical_filter(trial, lowcut, highcut, fs)
                                        for trial in np.transpose(data, (2, 0, 1))])

    x_train_filtered = process_data(x_train)
    x_val_filtered = process_data(x_val)
    x_test_filtered = process_data(x_test)

    # Transpose data to match CSP requirements: (n_trials, n_channels, n_time_points)
    x_train_final = np.transpose(x_train_filtered, (0, 2, 1))
    x_val_final = np.transpose(x_val_filtered, (0, 2, 1))
    x_test_final = np.transpose(x_test_filtered, (0, 2, 1))

    # Normalize data to range [-1, 1]
    normalize = lambda data: 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

    x_train_final = normalize(x_train_final)
    x_val_final = normalize(x_val_final)
    x_test_final = normalize(x_test_final)

    return x_train_final, y_train, x_val_final, y_val, x_test_final, y_test



# # Function to preprocess training data from data.mat (all trials used for training)
# def preprocess_training_data(x_data, y_data, lowcut=cfg.LOWCUT, highcut=cfg.HIGHCUT, fs=cfg.SAMPLING_RATE):
#     # Separate data by class
#     right_data = x_data[:, :, y_data.ravel() == 2]
#     left_data = x_data[:, :, y_data.ravel() == 1]

#     # Ensure we only use up to 500 time points
#     right_data = right_data[115:615, :, :]
#     left_data = left_data[115:615, :, :]

#     # Use all trials for training (combine all right and left trials)
#     x_train = np.concatenate([right_data, left_data], axis=2)
#     y_train = np.concatenate([np.zeros(right_data.shape[2]) + 2, np.zeros(left_data.shape[2]) + 1])

#     # Apply elliptical filter
#     x_train_filtered = np.array([elliptical_filter(trial, lowcut, highcut, fs)
#                                for trial in np.transpose(x_train, (2, 0, 1))])

#     # Transpose data to match CSP requirements: (n_trials, n_channels, n_time_points)
#     x_train_final = np.transpose(x_train_filtered, (0, 2, 1))

#     # Normalize data to range [-1, 1]
#     x_train_final = 2 * (x_train_final - np.min(x_train_final)) / (np.max(x_train_final) - np.min(x_train_final)) - 1

#     return x_train_final, y_train

# # Function to preprocess validation and test data from data1.mat
# def preprocess_val_test_data(x_data, y_data, lowcut=cfg.LOWCUT, highcut=cfg.HIGHCUT, fs=cfg.SAMPLING_RATE, val_trials=cfg.VAL_TRIALS):
#     # Separate data by class
#     right_data = x_data[:, :, y_data.ravel() == 2]
#     left_data = x_data[:, :, y_data.ravel() == 1]

#     # Ensure we only use up to 500 time points
#     right_data = right_data[115:615, :, :]
#     left_data = left_data[115:615, :, :]

#     # Validation data (10 trials per class)
#     x_val = np.concatenate([right_data[:, :, :val_trials],
#                            left_data[:, :, :val_trials]], axis=2)
#     y_val = np.concatenate([np.zeros(val_trials) + 2, np.zeros(val_trials) + 1])

#     # Test data (remaining trials per class)
#     x_test = np.concatenate([right_data[:, :, val_trials:],
#                             left_data[:, :, val_trials:]], axis=2)
#     y_test = np.concatenate([np.zeros(right_data.shape[2] - val_trials) + 2,
#                             np.zeros(left_data.shape[2] - val_trials) + 1])

#     # Apply elliptical filter to validation and test datasets
#     x_val_filtered = np.array([elliptical_filter(trial, lowcut, highcut, fs)
#                              for trial in np.transpose(x_val, (2, 0, 1))])
#     x_test_filtered = np.array([elliptical_filter(trial, lowcut, highcut, fs)
#                               for trial in np.transpose(x_test, (2, 0, 1))])

#     # Transpose data to match CSP requirements: (n_trials, n_channels, n_time_points)
#     x_val_final = np.transpose(x_val_filtered, (0, 2, 1))
#     x_test_final = np.transpose(x_test_filtered, (0, 2, 1))

#     # Normalize data to range [-1, 1]
#     x_val_final = 2 * (x_val_final - np.min(x_val_final)) / (np.max(x_val_final) - np.min(x_val_final)) - 1
#     x_test_final = 2 * (x_test_final - np.min(x_test_final)) / (np.max(x_test_final) - np.min(x_test_final)) - 1

#     return x_val_final, y_val, x_test_final, y_test