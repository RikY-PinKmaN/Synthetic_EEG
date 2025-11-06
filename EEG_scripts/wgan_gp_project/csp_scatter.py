import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import os

def log_message(message):
    """Prints a message to the console."""
    print(message)

def csp(data):
    """
    Custom Common Spatial Pattern (CSP) implementation based on main.py.
    """
    class1_indices = np.where(data['y'] == 1)[0]
    class2_indices = np.where(data['y'] == 2)[0]

    if len(class1_indices) == 0 or len(class2_indices) == 0:
        log_message("CSP Error: One class has no trials.")
        # Return identity matrix if one class is missing
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
    if valid_trials_c1 > 0:
        cov1 /= valid_trials_c1
    else:
        log_message("CSP Warning: All trials for class 1 had NaN covariance.")

    cov2 = np.zeros((n_channels, n_channels))
    valid_trials_c2 = 0
    for trial in range(X2.shape[2]):
        current_trial_data = X2[:,:,trial]
        cov_trial = np.cov(current_trial_data, rowvar=False)
        if not np.all(np.isnan(cov_trial)):
            cov2 += cov_trial
            valid_trials_c2 += 1
    if valid_trials_c2 > 0:
        cov2 /= valid_trials_c2
    else:
        log_message("CSP Warning: All trials for class 2 had NaN covariance.")

    # Regularization to avoid singularity
    epsilon_reg = 1e-9
    cov1_reg = cov1 + epsilon_reg * np.eye(n_channels)
    cov2_reg = cov2 + epsilon_reg * np.eye(n_channels)

    try:
        # Solve the generalized eigenvalue problem
        evals, evecs = eigh(cov1_reg, cov1_reg + cov2_reg)
    except np.linalg.LinAlgError:
        log_message("CSP Error: Generalized eigenvalue problem failed. Using regularized fallback.")
        # Fallback to standard eigenvalue decomposition of the regularized covariance
        evals, evecs = eigh(cov1_reg)

    # Sort eigenvectors in descending order of eigenvalues
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # The spatial filters are the rows of the transposed eigenvector matrix
    W = evecs.T
    return W

def main():
    """
    Main function to load data, compute CSP filters, and create a scatter plot.
    """
    # --- File Paths ---
    # It's good practice to define paths at the top
    real_data_path = 'training_8.mat'
    # The synthetic data could be in a subfolder, so we check there first
    synthetic_data_path_local = 'best_synthetic_batch_S8.mat'
   #synthetic_data_path_results = os.path.join('BCI_2a_results', 'Subject_2_D1Train_D2Test_20trials', 'best_synthetic_batch_cGAN_S2.mat')

    # --- Load Real Data ---
    try:
        log_message(f"Loading real data from: {real_data_path}")
        real_data_mat = scipy.io.loadmat(real_data_path)
        # Based on new.py, the keys are 'best_training_data' and 'best_training_labels'
        real_x_key = 'best_training_data'
        real_y_key = 'best_training_labels'

        if real_x_key not in real_data_mat or real_y_key not in real_data_mat:
             raise KeyError(f"Could not find '{real_x_key}' or '{real_y_key}' in the real data .mat file.")

        real_data = {'x': real_data_mat[real_x_key], 'y': real_data_mat[real_y_key].flatten()}
        log_message("Real data loaded successfully.")
    except (FileNotFoundError, KeyError) as e:
        log_message(f"FATAL: Could not load or parse real data from '{real_data_path}'. Error: {e}")
        return

    # --- Load Synthetic Data ---
    synthetic_data_path = None
    if os.path.exists(synthetic_data_path_local):
        synthetic_data_path = synthetic_data_path_local
   #elif os.path.exists(synthetic_data_path_results):
       #synthetic_data_path = synthetic_data_path_results
    else:
        log_message(f"FATAL: Synthetic data file not found at '{synthetic_data_path_local}'.")
        log_message("Please ensure you have run the converter.py script if you only have a .npz file.")
        return

    try:
        log_message(f"Loading synthetic data from: {synthetic_data_path}")
        synthetic_data_mat = scipy.io.loadmat(synthetic_data_path)
        # For synthetic data from the converter, the keys are based on the .npz file.
        # Common keys might be 'class0_data_gan_fmt' and 'class1_data_gan_fmt'.
        synth_c0_key = 'class0_data_gan_fmt'
        synth_c1_key = 'class1_data_gan_fmt'

        if synth_c0_key not in synthetic_data_mat or synth_c1_key not in synthetic_data_mat:
            raise KeyError("Could not find expected data keys in the synthetic .mat file.")

        synth_c0_gan_fmt = synthetic_data_mat[synth_c0_key] # (trials, channels, samples)
        synth_c1_gan_fmt = synthetic_data_mat[synth_c1_key] # (trials, channels, samples)

        # Convert from GAN format to CSP format (samples, channels, trials)
        synth_c0_csp_fmt = np.transpose(synth_c0_gan_fmt, (2, 1, 0))
        synth_c1_csp_fmt = np.transpose(synth_c1_gan_fmt, (2, 1, 0))

        # Combine into a single dictionary for the CSP function
        synth_x = np.concatenate((synth_c0_csp_fmt, synth_c1_csp_fmt), axis=2)
        synth_y = np.concatenate((np.ones(synth_c0_gan_fmt.shape[0]), np.ones(synth_c1_gan_fmt.shape[0]) + 1))
        synthetic_data = {'x': synth_x, 'y': synth_y}
        log_message("Synthetic data loaded and formatted successfully.")

    except (FileNotFoundError, KeyError) as e:
        log_message(f"FATAL: Could not load or parse synthetic data from '{synthetic_data_path}'. Error: {e}")
        return

    # --- Compute CSP Filters ---
    log_message("\nComputing CSP filters for REAL data...")
    W_real = csp(real_data)
    log_message("Computing CSP filters for SYNTHETIC data...")
    W_synth = csp(synthetic_data)

    if W_real.shape[0] < 2 or W_synth.shape[0] < 2:
        log_message("Not enough CSP filters were generated to create a scatter plot. Need at least 2.")
        return

    # --- Create Scatter Plot ---
    log_message("\nCreating scatter plot...")
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # We will plot the first vs the last filter for both real and synthetic data
    # These filters are expected to capture the most discriminative variance.
    plt.scatter(W_real[0, :], W_real[-1, :],
                marker='o', s=100, alpha=0.7, edgecolors='k',
                label='Real Data CSP Filters')

    plt.scatter(W_synth[0, :], W_synth[-1, :],
                marker='^', s=100, alpha=0.7, edgecolors='k',
                label='Synthetic Data CSP Filters')

    plt.title('Comparison of CSP Filters: Real vs. Synthetic Data', fontsize=16)
    plt.xlabel('CSP Filter 1 (Most discriminative for Class 1)', fontsize=12)
    plt.ylabel('CSP Filter N (Most discriminative for Class 2)', fontsize=12)
    plt.legend(fontsize=12)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.tight_layout()

    # Save the figure
    output_filename = 'csp_filter_scatter_comparison.png'
    plt.savefig(output_filename, dpi=300)
    log_message(f"\nScatter plot saved as '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    main()
