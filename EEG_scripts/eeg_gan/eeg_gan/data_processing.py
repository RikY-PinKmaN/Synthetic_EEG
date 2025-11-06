import numpy as np
import pandas as pd
from scipy.signal import ellip, ellipord, filtfilt


class EEGDataProcessor:
    def __init__(self, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
        """
        Initialize the EEGDataProcessor with filter parameters.
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.rp = rp
        self.rs = rs

    def elliptical_filter(self, data):
        """
        Apply a bandpass elliptical filter to the data.

        Parameters:
            data (ndarray): Input data (time points, electrodes).

        Returns:
            ndarray: Filtered data.
        """
        nyq = 0.5 * self.fs
        wp = [self.lowcut / nyq, self.highcut / nyq]  # Passband
        ws = [(self.lowcut - 1) / nyq, (self.highcut + 1) / nyq]  # Stopband
        n, wn = ellipord(wp, ws, self.rp, self.rs)  # Filter order and natural frequency
        b, a = ellip(n, self.rp, self.rs, wn, btype="band")  # Coefficients
        filtered_data = filtfilt(b, a, data, axis=0)  # Apply filter across time points
        return filtered_data

    def preprocess_data(self, x_data, y_data, train_trials=50):
        """
        Preprocess the data: filter, normalize, and split into train/test sets.

        Parameters:
            x_data (ndarray): EEG data (time points, electrodes, trials).
            y_data (ndarray): Labels for each trial.
            train_trials (int): Number of trials to use for training.

        Returns:
            tuple: Processed (x_train, y_train, x_test, y_test).
        """
        # Split data by labels
        right_data = x_data[:, :, y_data.ravel() == 2]
        left_data = x_data[:, :, y_data.ravel() == 1]

        # Combine and split data into train/test sets
        x_train = np.concatenate(
            [right_data[:, :, :train_trials], left_data[:, :, :train_trials]], axis=2
        )
        y_train = np.concatenate(
            [np.zeros(train_trials) + 2, np.zeros(train_trials) + 1]
        )

        x_test = np.concatenate(
            [right_data[:, :, train_trials:], left_data[:, :, train_trials:]], axis=2
        )
        y_test = np.concatenate(
            [np.zeros(right_data.shape[2] - train_trials) + 2, np.zeros(left_data.shape[2] - train_trials) + 1]
        )

        # Apply elliptical filter trial-by-trial
        x_train_filtered = np.array(
            [self.elliptical_filter(trial) for trial in np.transpose(x_train, (2, 0, 1))]
        )
        x_test_filtered = np.array(
            [self.elliptical_filter(trial) for trial in np.transpose(x_test, (2, 0, 1))]
        )

        # Transpose back to (time points, electrodes, trials) for compatibility
        x_train_filtered = np.transpose(x_train_filtered, (1, 2, 0))
        x_test_filtered = np.transpose(x_test_filtered, (1, 2, 0))

        # Normalize data to range [-1, 1]
        x_train_normalized = 2 * (x_train_filtered - np.min(x_train_filtered)) / (
            np.max(x_train_filtered) - np.min(x_train_filtered)
        ) - 1
        x_test_normalized = 2 * (x_test_filtered - np.min(x_test_filtered)) / (
            np.max(x_test_filtered) - np.min(x_test_filtered)
        ) - 1

        return x_train_normalized, y_train, x_test_normalized, y_test

    def create_csv_structure(self, x_data, y_data, kw_time="Time", kw_channel="Electrode", kw_condition="Condition"):
        """
        Convert processed EEG data into a CSV-compatible structure.

        Parameters:
            x_data (ndarray): EEG data (time points, electrodes, trials).
            y_data (ndarray): Labels for each trial.

        Returns:
            DataFrame: Processed data with columns for metadata and time points.
        """
        rows = []
        n_time_points, n_channels, n_trials = x_data.shape  # Time points, electrodes, trials

        # Generate time column headers
        time_columns = [f"{kw_time}{i+1}" for i in range(n_time_points)]

        # Loop through each trial
        for trial_idx in range(n_trials):
            condition = y_data[trial_idx]  # Condition for this trial
            for channel_idx in range(n_channels):
                # Extract metadata and time-series data
                electrode = channel_idx + 1  # Electrode ID (1-indexed)
                time_series = x_data[:, channel_idx, trial_idx]  # Time-series for this electrode and trial

                # Combine metadata and time-series into one row
                row = [condition, electrode] + list(time_series)
                rows.append(row)

        # Create DataFrame
        column_headers = [kw_condition, kw_channel] + time_columns
        df = pd.DataFrame(rows, columns=column_headers)

        return df
