import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import sem
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")

# --- Configuration ---
# Define the H folders and session pairs to process
H_FOLDERS = [f'H{i}' for i in range(1, 13)]
SESSION_PAIRS = ['1-2', '3-4', '5-6', '7-8']
CHANNEL_TO_PLOT = 'Cz'  # Specify the channel name to plot
CH_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']
CHANNEL_IDX = CH_NAMES.index(CHANNEL_TO_PLOT)

# Preprocessing and Plotting Parameters
FS = 80
TS_START_SEC, TS_END_SEC = -0.1, 0.6
OUTPUT_DIR = "Combined_Grand_Average_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# UTILITY AND CALCULATION FUNCTIONS
# =============================================================================

def log_message(message):
    """Prints a message to the console."""
    print(message)


def compute_grand_average(data, channel_idx):
    """Computes the grand average and standard error for a given channel."""
    if data.shape[0] == 0:
        return None
    # Select data for the specific channel
    class_data = data[:, channel_idx, :]
    grand_avg = np.mean(class_data, axis=0)
    err = sem(class_data, axis=0, nan_policy='omit')
    return {"grand_avg": grand_avg, "error": err, "n_trials": data.shape[0]}


def compute_cohens_d(target_data, nontarget_data, channel_idx):
    """
    Computes Cohen's d effect size for each time point between target and non-target data.
    """
    if target_data.shape[0] < 2 or nontarget_data.shape[0] < 2:
        return None

    # Select data for the specific channel
    target_ch_data = target_data[:, channel_idx, :]
    nontarget_ch_data = nontarget_data[:, channel_idx, :]

    n_target = target_ch_data.shape[0]
    n_nontarget = nontarget_ch_data.shape[0]

    mean_target = np.mean(target_ch_data, axis=0)
    mean_nontarget = np.mean(nontarget_ch_data, axis=0)

    std_target = np.std(target_ch_data, axis=0, ddof=1)
    std_nontarget = np.std(nontarget_ch_data, axis=0, ddof=1)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(
        ((n_target - 1) * std_target**2 + (n_nontarget - 1) * std_nontarget**2) /
        (n_target + n_nontarget - 2)
    )
    
    # Calculate Cohen's d, handle potential division by zero
    d = (mean_target - mean_nontarget) / pooled_std
    d = np.nan_to_num(d) # Replace NaN or Inf with 0 if pooled_std is 0

    return d


def shaded_error_bar(x, y, err_bar, line_props=None, transparent=False, ax=None):
    """A helper function to plot a line with a shaded error region."""
    if ax is None: ax = plt.gca()
    y, x = np.ravel(y), np.ravel(x)
    if err_bar.ndim == 1: err_bar = np.vstack([err_bar, err_bar])
    if line_props is None: line_props = {'color': 'k', 'linestyle': '-'}
    main_line, = ax.plot(x, y, **line_props)
    col = main_line.get_color()
    face_alpha = 0.15 if transparent else 1.0
    patch_color = col if transparent else tuple(c + (1 - c) * 0.85 for c in plt.cm.colors.to_rgb(col))
    uE, lE = y + err_bar[0, :], y - err_bar[1, :]
    patch = ax.fill_between(x, lE, uE, color=patch_color, alpha=face_alpha, edgecolor='none')
    edge_color = tuple(c + (1 - c) * 0.45 for c in plt.cm.colors.to_rgb(col))
    edge1, = ax.plot(x, lE, '-', color=edge_color); edge2, = ax.plot(x, uE, '-', color=edge_color)
    main_line.remove(); main_line, = ax.plot(x, y, **line_props)
    return {'mainLine': main_line, 'patch': patch, 'edge': (edge1, edge2)}

# =============================================================================
# COMBINED PLOTTING FUNCTION
# =============================================================================

def plot_erp_and_effect_size_combined(
    real_target_data, synthetic_target_data,
    d_real, d_synthetic,
    session_pair, output_dir, time_vector,
    y_axis_label="Amplitude (Normalized)"
):
    """
    Generates and saves a single figure with two subplots:
    1. Top: Grand Average ERP of Target trials (Real vs. Synthetic).
    2. Bottom: Effect Size (Cohen's d) for (Target vs. Non-Target).
    """
    log_message(f"\n--- Generating Combined Plot for Session Pair {session_pair} ---")

    fig, axes = plt.subplots(2, 1, figsize=(15, 16), sharex=True)
    ax_erp, ax_effect = axes[0], axes[1]
    
    b_plot, a_plot = butter(4, 16 / FS * 2, btype='low')

    # --- 1. Top Plot: Grand Average ERP ---
    erp_plot_successful = False
    datasets = {"Real": real_target_data, "Synthetic": synthetic_target_data}
    style_map = {"Real": {'color': 'blue', 'linestyle': '-'}, "Synthetic": {'color': 'green', 'linestyle': '--'}}

    for data_type, data in datasets.items():
        if data is None or data.shape[0] == 0:
            log_message(f"Skipping ERP plot for {data_type} data as it is empty.")
            continue

        results = compute_grand_average(data, CHANNEL_IDX)
        if results:
            grand_avg_filt = filtfilt(b_plot, a_plot, results["grand_avg"])
            line_props = style_map[data_type].copy()
            line_props['label'] = f'Combined {data_type} Target ({results["n_trials"]} trials)'
            line_props['linewidth'] = 2.5
            shaded_error_bar(x=time_vector, y=grand_avg_filt, err_bar=results["error"], line_props=line_props, transparent=True, ax=ax_erp)
            erp_plot_successful = True
    
    if erp_plot_successful:
        ax_erp.set_title("Grand Average ERP (Target Stimuli)", fontsize=16)
        ax_erp.set_ylabel(y_axis_label, fontsize=14)
        ax_erp.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_erp.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax_erp.legend(fontsize=12)
        ax_erp.grid(True)

    # --- 2. Bottom Plot: Effect Size (Cohen's d) ---
    effect_plot_successful = False
    if d_real is not None:
        ax_effect.plot(time_vector, d_real, color='blue', linestyle='-', linewidth=2.5, label='Real Data Effect Size')
        effect_plot_successful = True
    
    if d_synthetic is not None:
        ax_effect.plot(time_vector, d_synthetic, color='green', linestyle='--', linewidth=2.5, label='Synthetic Data Effect Size')
        effect_plot_successful = True

    if effect_plot_successful:
        ax_effect.set_title("Effect Size (Target vs. Non-Target)", fontsize=16)
        ax_effect.set_ylabel("Effect Size (Cohen's d)", fontsize=14)
        ax_effect.axhline(0.2, color='gray', linewidth=0.7, linestyle=':', label='Small (d=0.2)')
        ax_effect.axhline(0.5, color='gray', linewidth=0.7, linestyle='--', label='Medium (d=0.5)')
        ax_effect.axhline(0.8, color='gray', linewidth=0.7, linestyle='-', label='Large (d=0.8)')
        ax_effect.legend(fontsize=12)
        ax_effect.grid(True)

    # --- Global Figure Formatting ---
    if erp_plot_successful or effect_plot_successful:
        fig.suptitle(f"ERP and Effect Size Comparison - Session Pair {session_pair} (H1-H12) - Channel {CHANNEL_TO_PLOT}", fontsize=20)
        ax_effect.set_xlabel("Time (s)", fontsize=14)
        ax_effect.set_xlim([TS_START_SEC, TS_END_SEC])
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
        plot_filename = os.path.join(output_dir, f"Combined_ERP_EffectSize_S{session_pair}.png")
        plt.savefig(plot_filename)
        log_message(f"Saved combined plot to: {plot_filename}")
    else:
        log_message(f"No data was available to plot for Session Pair {session_pair}.")
    
    plt.close()

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def load_data_from_path(file_path):
    """Helper to load .mat data and handle exceptions."""
    if os.path.exists(file_path):
        try:
            mat_contents = scipy.io.loadmat(file_path)
            # Find a key that contains 'data'
            data_key = [k for k in mat_contents if not k.startswith('__') and 'data' in k][0]
            data = mat_contents[data_key]
            
            train_min = mat_contents.get('train_min')
            train_max = mat_contents.get('train_max')
            
            log_message(f"  Loaded {data.shape[0]} trials from {os.path.basename(file_path)}")

            min_val, max_val = None, None
            if train_min is not None and train_min.size > 0:
                min_val = train_min.item()
            if train_max is not None and train_max.size > 0:
                max_val = train_max.item()

            return data, min_val, max_val
        except Exception as e:
            log_message(f"  ERROR loading data from {file_path}: {e}")
    else:
        log_message(f"  Data file not found: {file_path}")
    return None, None, None

def main():
    """
    Main function to load, combine, and plot data for all specified session pairs.
    """
    for pair in SESSION_PAIRS:
        log_message(f"\n{'='*20} Processing Session Pair: {pair} {'='*20}")
        
        all_real_target_trials, all_real_nontarget_trials = [], []
        all_synth_target_trials, all_synth_nontarget_trials = [], []
        train_min, train_max = None, None

        for h_folder in H_FOLDERS:
            log_message(f"-> Loading data from {h_folder} for session pair {pair}...")
            base_path = os.path.join(f"{h_folder}_results", f"Subject_{pair}_results")
            
            real_target_data, t_min, t_max = load_data_from_path(os.path.join(base_path, f"target_training_data_S{pair}.mat"))
            if real_target_data is not None: 
                all_real_target_trials.append(real_target_data)
                if t_min is not None and train_min is None:
                    train_min, train_max = t_min, t_max

            real_nontarget_data, t_min, t_max = load_data_from_path(os.path.join(base_path, f"nontarget_training_data_S{pair}.mat"))
            if real_nontarget_data is not None: 
                all_real_nontarget_trials.append(real_nontarget_data)
                if t_min is not None and train_min is None:
                    train_min, train_max = t_min, t_max

            synth_target_data, _, _ = load_data_from_path(os.path.join(base_path, f"target_synthetic_data_S{pair}.mat"))
            if synth_target_data is not None: all_synth_target_trials.append(synth_target_data)

            synth_nontarget_data, _, _ = load_data_from_path(os.path.join(base_path, f"nontarget_synthetic_data_S{pair}.mat"))
            if synth_nontarget_data is not None: all_synth_nontarget_trials.append(synth_nontarget_data)

        # Concatenate all loaded data
        final_real_target = np.concatenate(all_real_target_trials, axis=0) if all_real_target_trials else np.array([])
        final_real_nontarget = np.concatenate(all_real_nontarget_trials, axis=0) if all_real_nontarget_trials else np.array([])
        final_synth_target = np.concatenate(all_synth_target_trials, axis=0) if all_synth_target_trials else np.array([])
        final_synth_nontarget = np.concatenate(all_synth_nontarget_trials, axis=0) if all_synth_nontarget_trials else np.array([])
        
        log_message(f"Total real trials -> Target: {final_real_target.shape[0]}, Non-Target: {final_real_nontarget.shape[0]}")
        log_message(f"Total synthetic trials -> Target: {final_synth_target.shape[0]}, Non-Target: {final_synth_nontarget.shape[0]}")

        # Proceed only if there is data to process
        if final_real_target.size > 0:
            time_vector = np.linspace(TS_START_SEC, TS_END_SEC, final_real_target.shape[2])
            
            # Calculate effect sizes on normalized data
            d_real = compute_cohens_d(final_real_target, final_real_nontarget, CHANNEL_IDX)
            d_synthetic = compute_cohens_d(final_synth_target, final_synth_nontarget, CHANNEL_IDX)

            # Denormalize data for plotting if scaling factors are available
            y_axis_label = "Amplitude (Normalized)"
            if train_min is not None and train_max is not None and train_max > train_min:
                log_message(f"Denormalizing ERP data using min={train_min:.4f}, max={train_max:.4f} for plotting.")
                denormalize = lambda y: (y + 1) * (train_max - train_min) / 2 + train_min
                if final_real_target.size > 0: final_real_target = denormalize(final_real_target)
                if final_synth_target.size > 0: final_synth_target = denormalize(final_synth_target)
                y_axis_label = "Amplitude (μV)"

            # Generate the combined plot
            plot_erp_and_effect_size_combined(
                real_target_data=final_real_target,
                synthetic_target_data=final_synth_target,
                d_real=d_real,
                d_synthetic=d_synthetic,
                session_pair=pair,
                output_dir=OUTPUT_DIR,
                time_vector=time_vector,
                y_axis_label=y_axis_label
            )
        else:
            log_message(f"Skipping plots for session pair {pair} due to missing essential data.")

if __name__ == '__main__':
    main()
    log_message("\n--- Script finished ---")