# =================================================================================================
#
#               P300 SYNTHETIC DATA GENERATION & EVALUATION PIPELINE (v5 - Definitive)
#
# This script integrates a Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)
# for generating synthetic P300 EEG data.
#
# V5 CORRECTIONS:
# - Added `matplotlib.use('Agg')` at the top of the script. This forces a non-interactive
#   backend for matplotlib, resolving the critical conflict between the multiprocessing
#   used by `mlxtend` and the default GUI backend (Tkinter). This prevents the
#   "main thread is not in main loop" crash.
#
# =================================================================================================

import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, welch, savgol_filter
from scipy.stats import mode, sem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
import traceback
import scipy.io

# ==================================================================
#                           *** FIX ***
# Force a non-interactive backend for Matplotlib BEFORE importing pyplot.
# This prevents conflicts between multiprocessing (from mlxtend) and
# GUI toolkits (like Tkinter), which was causing the crash.
# ==================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")

# --- P300 Processing & Pipeline Configuration ---
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Data and Subject Configuration
SESSION_IDS = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
            
# Preprocessing Parameters
FS = 80
ARTIFTH = 50
TS_START_SEC, TS_END_SEC = -0.1, 0.6
BASELINE_START_SEC, BASELINE_END_SEC = -0.1, 0.0

# Data Splitting Parameters
# Non-target trial counts are now set dynamically within the main_analysis function
# based on the subject group (2x for subjects 1-4, 4x for subjects 5-8).
NUM_TRAIN_TARGET_TRIALS = 60
NUM_VALID_TARGET_TRIALS = 15

# GAN & Evaluation Parameters
LATENT_DIM_CGAN = 100
NUM_CLASSES_CGAN = 2
EMBEDDING_DIM_CGAN = 25
NUM_RUNS_PER_SUBJECT = 5
GAN_EPOCHS = 1000
BATCH_SIZE_GAN = 64
N_CRITIC_STEPS = 2
NUM_GENERATION_BATCHES = 30

# Loss Weights
FREQ_LOSS_WEIGHT = 0.4
P300_LOSS_WEIGHT = 0.9  

# P300 Window Definition for Loss Calculation
P300_WINDOW_START_SEC = 0.15
P300_WINDOW_END_SEC = 0.5

# Global variable for log file path
current_log_file_path = None

# =============================================================================
# PART 1: UTILITY AND LOGGING FUNCTIONS
# =============================================================================

def log_message(message):
    print(message)
    global current_log_file_path
    if current_log_file_path:
        with open(current_log_file_path, 'a') as f: f.write(str(message) + "\n")

def shaded_error_bar(x, y, err_bar, line_props=None, transparent=False, ax=None):
    if ax is None: ax = plt.gca()
    y, x = np.ravel(y), np.ravel(x)
    if len(err_bar.shape) == 1: err_bar = np.vstack([err_bar, err_bar])
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
# PART 2: P300 CLASSIFIER & FEATURE ENGINEERING
# =============================================================================

def create_averaged_features_from_trials(single_trials_data, single_trials_labels, n_average=15):
    feature_list, label_list = [], []
    for label_class in [1, 2]:
        class_indices = np.where(single_trials_labels == label_class)[0]
        class_trials = single_trials_data[class_indices]
        num_features_for_class = len(class_trials) // n_average
        for i in range(num_features_for_class):
            trial_group = class_trials[i * n_average : (i + 1) * n_average]
            averaged_feature = np.mean(trial_group, axis=0)
            feature_list.append(averaged_feature)
            label_list.append(label_class)
    if not feature_list:
        return {'x': np.empty((single_trials_data.shape[1], single_trials_data.shape[2], 0)), 'y': np.empty((0,))}
    p300_features = np.array(feature_list).transpose(1, 2, 0)
    p300_labels = np.array(label_list)
    return {'x': p300_features, 'y': p300_labels}

def train_p300_lda(training_features):
    p300_feature, p300_labels = training_features['x'], training_features['y'].ravel()
    if p300_feature.shape[2] == 0: return None, None
    num_channels, num_samples, num_trials = p300_feature.shape
    p300_feature_reshaped = p300_feature.transpose(2, 1, 0).reshape(num_trials, -1)
    sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), k_features=5, forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=-1)
    sfs.fit(p300_feature_reshaped, p300_labels)
    selected_feature_indices = list(sfs.k_feature_idx_)
    train_data_selected = p300_feature_reshaped[:, selected_feature_indices]
    final_lda_model = LinearDiscriminantAnalysis()
    final_lda_model.fit(train_data_selected, p300_labels)
    return final_lda_model, selected_feature_indices

def evaluate_p300_lda(model, feature_indices, evaluation_features):
    if model is None or feature_indices is None: return 0.0
    p300_feature, true_labels = evaluation_features['x'], evaluation_features['y'].ravel()
    if p300_feature.shape[2] == 0: return 0.0
    num_channels, num_samples, num_trials = p300_feature.shape
    p300_feature_reshaped = p300_feature.transpose(2, 1, 0).reshape(num_trials, -1)
    test_data_selected = p300_feature_reshaped[:, feature_indices]
    predicted_labels = model.predict(test_data_selected)
    return np.mean(predicted_labels == true_labels) * 100

# =============================================================================
# PART 3: CONDITIONAL GAN (cWGAN-GP) FUNCTIONS (Identical to c_MAIN.py)
# =============================================================================

def smooth_eeg(data, window_size=11):
    smoothed_data = np.copy(data)
    if window_size % 2 == 0: window_size += 1
    if data.shape[2] <= window_size: return data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            smoothed_data[i, j, :] = savgol_filter(data[i, j, :], window_size, 2)
    return smoothed_data

def frequency_domain_loss(real_data, generated_data):
    real_data_t = tf.transpose(real_data, perm=[0, 2, 1])
    gen_data_t = tf.transpose(generated_data, perm=[0, 2, 1])
    real_fft = tf.abs(tf.signal.rfft(real_data_t))
    gen_fft = tf.abs(tf.signal.rfft(gen_data_t))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))

def p300_loss(real_data, generated_data):
    # Calculate sample indices for the P300 window
    start_idx = int((P300_WINDOW_START_SEC - TS_START_SEC) * FS)
    end_idx = int((P300_WINDOW_END_SEC - TS_START_SEC) * FS)
    
    # Slice the tensors to get the data only within the P300 window
    real_p300_window = real_data[:, :, start_idx:end_idx]
    fake_p300_window = generated_data[:, :, start_idx:end_idx]
    
    # Return the Mean Absolute Error (L1 Loss)
    return tf.reduce_mean(tf.abs(real_p300_window - fake_p300_window))


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


def gradient_penalty_cgan(critic, real_samples, fake_samples, real_labels, lambda_gp=10):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return lambda_gp * gp

def train_wgan_gp_cgan(eeg_data, eeg_labels, epochs, batch_size, subject_id_str, run_number, output_dir):
    num_channels, time_points = eeg_data.shape[1], eeg_data.shape[2]
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)
    labels_0_1_reshaped = labels_0_1[:, np.newaxis]

    generator = build_generator_cgan(num_channels, time_points)
    critic = build_critic_cgan(num_channels, time_points)
    gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step_wgan_cgan(real_eeg_batch, real_labels_batch_0_1_tf):
        batch_size_tf = tf.shape(real_eeg_batch)[0]

        # -------------------------
        #  Train the Critic
        # -------------------------
        for _ in range(N_CRITIC_STEPS):
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch_size_tf, LATENT_DIM_CGAN])
                # Use real labels for fake data generation during critic training for stability
                fake_eeg = generator([noise, real_labels_batch_0_1_tf], training=True)
                real_output = critic([real_eeg_batch, real_labels_batch_0_1_tf], training=True)
                fake_output = critic([fake_eeg, real_labels_batch_0_1_tf], training=True)
                gp = gradient_penalty_cgan(critic, real_eeg_batch, fake_eeg, real_labels_batch_0_1_tf)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
            gradients_d = tape.gradient(d_loss, critic.trainable_variables)
            crit_opt.apply_gradients(zip(gradients_d, critic.trainable_variables))

        # -------------------------
        #  Train the Generator
        # -------------------------
        with tf.GradientTape() as tape_g:
            noise_g = tf.random.normal([batch_size_tf, LATENT_DIM_CGAN])
            # Use real labels for generator training to ensure generated data matches the batch context
            fake_eeg_for_g = generator([noise_g, real_labels_batch_0_1_tf], training=True)
            fake_output_for_g = critic([fake_eeg_for_g, real_labels_batch_0_1_tf], training=True)

            # --- Calculate all generator losses ---
            g_loss_wasserstein = -tf.reduce_mean(fake_output_for_g)
            p300_loss_val = p300_loss(real_eeg_batch, fake_eeg_for_g)
            freq_loss_val = frequency_domain_loss(real_eeg_batch, fake_eeg_for_g)

            # --- Combine losses into a single total loss ---
            total_g_loss = g_loss_wasserstein + \
                           (P300_LOSS_WEIGHT * p300_loss_val) + \
                           (FREQ_LOSS_WEIGHT * freq_loss_val)

        # --- Apply gradients for the combined loss once ---
        gradients_g = tape_g.gradient(total_g_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))

        return d_loss, total_g_loss

    log_message(f"\n--- Training cWGAN-GP for Subject {subject_id_str}, Run {run_number} ---")
    d_losses, g_losses = [], []
    
    # Create a tf.data.Dataset for efficient batching
    dataset = tf.data.Dataset.from_tensor_slices((eeg_data.astype(np.float32), labels_0_1_reshaped.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=len(eeg_data)).batch(batch_size, drop_remainder=True)
    
    num_batches_per_epoch = len(eeg_data) // batch_size

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss_comb = 0.0, 0.0
        
        for real_eeg_batch, real_labels_batch in dataset:
            d_loss, g_loss_comb = train_step_wgan_cgan(real_eeg_batch, real_labels_batch)
            epoch_d_loss += d_loss
            epoch_g_loss_comb += g_loss_comb

        avg_d_loss = epoch_d_loss / num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        avg_g_loss = epoch_g_loss_comb / num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:
            log_message(f"Epoch {epoch}/{epochs}: D Loss={avg_d_loss:.4f}, G Loss (Comb)={avg_g_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Critic Loss"); plt.plot(g_losses, label="Gen Loss (W+Freq+P300)")
    plt.title(f"cGAN Losses - Subject {subject_id_str} Run {run_number}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"GAN_Losses_S{subject_id_str}_Run{run_number}.png")); plt.close()
    return generator

def generate_synthetic_data_cgan(generator, num_samples, target_label_1_2, smooth=True):
    if generator is None or num_samples == 0: return np.array([])
    target_label_0_1 = target_label_1_2 - 1
    noise = tf.random.normal([num_samples, LATENT_DIM_CGAN])
    labels_for_gen = tf.ones((num_samples, 1), dtype=tf.int32) * target_label_0_1
    synthetic_data = generator([noise, labels_for_gen], training=False).numpy()
    if smooth and synthetic_data.size > 0:
        synthetic_data = smooth_eeg(synthetic_data)
    return synthetic_data

# =============================================================================
# PART 4: PLOTTING AND ANALYSIS FUNCTIONS
# =============================================================================

def compute_grand_average(data, labels, channel_idx, class_labels):
    results = {}
    for class_label in class_labels:
        class_indices = np.where(np.asarray(labels).squeeze() == class_label)[0]
        if len(class_indices) == 0:
            results[f"class_{class_label}"] = None
            continue
        class_data = data[class_indices, channel_idx, :]
        grand_avg = np.mean(class_data, axis=0)
        err = sem(class_data, axis=0, nan_policy='omit')
        results[f"class_{class_label}"] = {"grand_avg": grand_avg, "error": err, "n_trials": len(class_indices)}
    return results

def calculate_erp_similarity_score(real_data, real_labels, synth_data, synth_labels):
    """
    Calculates a similarity score based on the Mean Absolute Error (MAE) between
    the grand average ERPs of real and synthetic data. A higher score is better.
    """
    # Using all channels by passing slice(None)
    real_avg_results = compute_grand_average(real_data, real_labels, channel_idx=slice(None), class_labels=[1, 2])
    synth_avg_results = compute_grand_average(synth_data, synth_labels, channel_idx=slice(None), class_labels=[1, 2])

    total_error = 0
    num_comparisons = 0

    # Compare target ERPs (class 1)
    real_target_res = real_avg_results.get('class_1')
    synth_target_res = synth_avg_results.get('class_1')
    if real_target_res and synth_target_res:
        real_target_erp = real_target_res['grand_avg']  # Shape: (n_channels, n_samples)
        synth_target_erp = synth_target_res['grand_avg']
        total_error += np.mean(np.abs(real_target_erp - synth_target_erp))
        num_comparisons += 1

    # Compare non-target ERPs (class 2)
    real_nontarget_res = real_avg_results.get('class_2')
    synth_nontarget_res = synth_avg_results.get('class_2')
    if real_nontarget_res and synth_nontarget_res:
        real_nontarget_erp = real_nontarget_res['grand_avg']
        synth_nontarget_erp = synth_nontarget_res['grand_avg']
        total_error += np.mean(np.abs(real_nontarget_erp - synth_nontarget_erp))
        num_comparisons += 1

    if num_comparisons == 0:
        return -np.inf  # Return a very low score if no comparison could be made

    # Return negative average MAE, so that a higher score is better
    return -(total_error / num_comparisons)


def plot_grand_average_comparison(real_data_train_plot, synthetic_data_plot, real_labels_train_plot, synthetic_labels_plot,
                                 channel_idx_plot, class_labels_plot=[1, 2], sampling_rate_plot=FS,
                                 output_dir_param=None, subject_display_idx_param=None,
                                 augmented_data=None, augmented_labels=None,
                                 train_min=None, train_max=None):
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']
    channel_name = ch_names[channel_idx_plot]
    log_message(f"\n--- Plotting Combined Grand Average ERP Comparison for Channel {channel_name} (Session {subject_display_idx_param}) ---")

    # --- Denormalization logic ---
    denormalize = None
    y_axis_label = "Amplitude (normalized)"
    if train_min is not None and train_max is not None and train_max > train_min:
        denormalize = lambda y: (y + 1) * (train_max - train_min) / 2 + train_min
        y_axis_label = "Amplitude (μV)"
        log_message("Data will be rescaled to original amplitude (μV) for plotting.")

    datasets = {
        "Real": (real_data_train_plot, real_labels_train_plot),
        "Synthetic": (synthetic_data_plot, synthetic_labels_plot),
    }
    # if augmented_data is not None and augmented_labels is not None:
    #     datasets["Augmented"] = (augmented_data, augmented_labels)

    time_vector = np.linspace(TS_START_SEC, TS_END_SEC, real_data_train_plot.shape[2])
    b_plot, a_plot = butter(4, 16 / sampling_rate_plot * 2, btype='low')

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_successful = False

    # Define colors and styles to distinguish data types and classes
    style_map = {
        "Real":      {'target': {'color': 'blue', 'linestyle': '-'}, 'nontarget': {'color': 'red', 'linestyle': '-'}},
        "Synthetic": {'target': {'color': 'green', 'linestyle': '--'}, 'nontarget': {'color': 'orange', 'linestyle': '--'}},
        # "Augmented": {'target': {'color': 'purple', 'linestyle': ':'}, 'nontarget': {'color': 'brown', 'linestyle': ':'}},
    }

    for data_type, (data, labels) in datasets.items():
        if data is None or labels is None or len(data) == 0:
            log_message(f"Skipping {data_type} data as it is empty.")
            continue

        # Denormalize data if a denormalization function is available
        plot_data = denormalize(data) if denormalize and data.size > 0 else data

        results = compute_grand_average(plot_data, labels, channel_idx_plot, class_labels_plot)
        
        target_results = results.get('class_1')
        nontarget_results = results.get('class_2')

        if target_results:
            grand_avg_filt = filtfilt(b_plot, a_plot, target_results["grand_avg"])
            line_props = style_map[data_type]['target'].copy()
            line_props['label'] = f'{data_type} Target ({target_results["n_trials"]} trials)'
            line_props['linewidth'] = 2.5
            shaded_error_bar(
                x=time_vector, y=grand_avg_filt, err_bar=target_results["error"],
                line_props=line_props,
                transparent=True, ax=ax
            )
            plot_successful = True

        if nontarget_results and data_type == "Real":
            grand_avg_filt = filtfilt(b_plot, a_plot, nontarget_results["grand_avg"])
            line_props = style_map[data_type]['nontarget'].copy()
            line_props['label'] = f'{data_type} Non-Target ({nontarget_results["n_trials"]} trials)'
            line_props['linewidth'] = 2.5
            shaded_error_bar(
                x=time_vector, y=grand_avg_filt, err_bar=nontarget_results["error"],
                line_props=line_props,
                transparent=True, ax=ax
            )
            plot_successful = True

    if plot_successful:
        ax.set_title(f"Combined Grand Average ERP Comparison - Session {subject_display_idx_param} - Channel {channel_name}", fontsize=18)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel(y_axis_label, fontsize=16)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlim([TS_START_SEC, TS_END_SEC])
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir_param, f"GA_ERP_Combined_S{subject_display_idx_param}_Ch{channel_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        log_message(f"Saved combined grand average plot to: {plot_filename}")
    else:
        log_message("No data was plotted for any dataset.")
        plt.close()


def plot_accuracy_chart(accuracies, subject_id_str, output_dir):
    # This function is adapted from c_MAIN.py for a consistent, clear summary plot.
    try:
        labels = list(accuracies.keys())
        values = list(accuracies.values())

        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, values, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
        plt.ylabel('Classification Accuracy (%)', fontsize=12)
        plt.title(f'S{subject_id_str}: Classification Accuracy Comparison on Test Set', fontsize=14, fontweight='bold')
        plt.xticks(rotation=15, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(max(0, min(values) - 10 if values else 0), 100)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"accuracy_comparison_S{subject_id_str}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        log_message(f"\nSaved accuracy comparison plot to: {plot_filename}")
    except Exception as e_plot:
        log_message(f"\nERROR creating accuracy plot: {e_plot}")


# =============================================================================
# PART 5: MAIN ANALYSIS SCRIPT
# =============================================================================

def main_analysis(data_path):
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']
    for s_idx, s in enumerate(range(0, 7, 2)):
        subject_id_str = f"{s+1}-{s+2}"

        # Dynamically set non-target trial counts based on subject group
        if s < 4:  # For subjects 1-4 (pairs 1-2, 3-4)
            ratio = 2
            num_train_nontarget_trials = NUM_TRAIN_TARGET_TRIALS * ratio
            num_valid_nontarget_trials = NUM_VALID_TARGET_TRIALS * ratio
        else:  # For subjects 5-8 (pairs 5-6, 7-8)
            ratio = 4
            num_train_nontarget_trials = NUM_TRAIN_TARGET_TRIALS * ratio
            num_valid_nontarget_trials = NUM_VALID_TARGET_TRIALS * ratio
        
        base_output_dir = f"{data_path}_results"
        output_dir = os.path.join(base_output_dir, f"Subject_{subject_id_str}_results")
        os.makedirs(output_dir, exist_ok=True)

        global current_log_file_path
        current_log_file_path = os.path.join(output_dir, "run_log.txt")
        with open(current_log_file_path, 'w') as f: f.write(f"Log for Subject Pair {subject_id_str} from {data_path}\n{'='*40}\n")
        log_message(f"\n\n{'='*25} PROCESSING SUBJECT PAIR: {subject_id_str} from {data_path} {'='*25}")
        log_message(f"Using 1:{ratio} Target:Non-Target ratio for this subject group.")

        try:
            # --- 1. DATA LOADING AND PREPROCESSING ---
            log_message("\n--- Step 1: Loading and Preprocessing Data ---")
            raw1 = mne.io.read_raw_gdf(os.path.join(data_path, f"{SESSION_IDS[s]}.gdf"), preload=True)
            raw2 = mne.io.read_raw_gdf(os.path.join(data_path, f"{SESSION_IDS[s+1]}.gdf"), preload=True)
            raw1.rename_channels({old: new for old, new in zip(raw1.ch_names, ch_names)})
            raw2.rename_channels({old: new for old, new in zip(raw2.ch_names, ch_names)})
            raw = mne.concatenate_raws([raw1, raw2])
            original_fs = raw.info['sfreq']
            data = raw.get_data()
            b_hp, a_hp = butter(4, 0.4 / (original_fs / 2), 'high')
            b_lp, a_lp = butter(4, 30 / (original_fs / 2), 'low')
            b_n1, a_n1 = iirnotch(4.3 if s in [0,4] else 5.8, 5, original_fs)
            b_n2, a_n2 = iirnotch((4.3 if s in [0,4] else 5.8)*2, 5, original_fs)
            for j in range(data.shape[0]):
                data[j, :] = filtfilt(b_hp, a_hp, filtfilt(b_lp, a_lp, data[j, :]))
                data[j, :] = filtfilt(b_n1, a_n1, filtfilt(b_n2, a_n2, data[j, :]))
            raw._data = data; raw.resample(FS)
            log_message(f"Filtering and resampling complete. New Fs: {raw.info['sfreq']} Hz")

            # --- 2. EPOCHING AND ARTIFACT REJECTION ---
            log_message("\n--- Step 2: Epoching and Artifact Rejection ---")
            events, event_id = mne.events_from_annotations(raw)
            id_to_desc = {v: k for k, v in event_id.items()}
            event_codes = np.array([int(id_to_desc[code]) for code in events[:, 2]])
            stim_codes = np.zeros_like(event_codes)
            stim_codes[event_codes == 33286] = 2; stim_codes[event_codes == 33285] = 1
            flash_indices = np.where(stim_codes != 0)[0]
            EEG = raw.get_data() * 1e6
            ts_samples = np.arange(round(TS_START_SEC * FS), round(TS_END_SEC * FS))
            bl_samples = np.arange(round(BASELINE_START_SEC * FS), round(BASELINE_END_SEC * FS))
            clean_trials_target, clean_trials_nontarget = [], []
            for i in flash_indices:
                if i + 1 >= len(events): continue
                latency = events[i + 1, 0]
                epoch_indices = latency + ts_samples
                if np.min(epoch_indices) < 0 or np.max(epoch_indices) >= EEG.shape[1]: continue
                ep = EEG[:, epoch_indices]
                bl_indices = events[i, 0] + bl_samples
                if np.min(bl_indices) < 0 or np.max(bl_indices) >= EEG.shape[1]: continue
                BLamp = np.mean(EEG[:, bl_indices], axis=1, keepdims=True)
                ep_bc = ep - BLamp
                if np.any(np.max(ep_bc, axis=1) - np.min(ep_bc, axis=1) > ARTIFTH): continue
                if stim_codes[i] == 1: clean_trials_target.append(ep_bc)
                elif stim_codes[i] == 2: clean_trials_nontarget.append(ep_bc)
            if not clean_trials_target or not clean_trials_nontarget:
                log_message("CRITICAL: Not enough clean trials found. Skipping subject."); continue
            all_clean_trials = np.array(clean_trials_target + clean_trials_nontarget)
            all_clean_labels = np.array([1]*len(clean_trials_target) + [2]*len(clean_trials_nontarget))
            log_message(f"Found {len(clean_trials_target)} clean Target and {len(clean_trials_nontarget)} clean Non-Target trials.")

            # --- 3. SEQUENTIAL DATA SPLIT ---
            log_message("\n--- Step 3: Performing Fixed Sequential Data Split ---")
            idx_c1 = np.where(all_clean_labels == 1)[0]; idx_c2 = np.where(all_clean_labels == 2)[0]
            total_needed_c1 = NUM_TRAIN_TARGET_TRIALS + NUM_VALID_TARGET_TRIALS
            total_needed_c2 = num_train_nontarget_trials + num_valid_nontarget_trials
            if len(idx_c1) < total_needed_c1 or len(idx_c2) < total_needed_c2:
                log_message(f"CRITICAL: Not enough trials for split. Have T:{len(idx_c1)}/NT:{len(idx_c2)}, need T:{total_needed_c1}/NT:{total_needed_c2}."); continue
            train_idx = np.concatenate((idx_c1[:NUM_TRAIN_TARGET_TRIALS], idx_c2[:num_train_nontarget_trials]))
            valid_idx = np.concatenate((idx_c1[NUM_TRAIN_TARGET_TRIALS:total_needed_c1], idx_c2[num_train_nontarget_trials:total_needed_c2]))
            test_idx = np.concatenate((idx_c1[total_needed_c1:], idx_c2[total_needed_c2:]))
            train_trials, train_labels = all_clean_trials[train_idx], all_clean_labels[train_idx]
            valid_trials, valid_labels = all_clean_trials[valid_idx], all_clean_labels[valid_idx]
            test_trials, test_labels = all_clean_trials[test_idx], all_clean_labels[test_idx]
            train_min, train_max = np.min(train_trials), np.max(train_trials)
            normalize = lambda x: 2 * (x - train_min) / (train_max - train_min) - 1
            train_trials_norm = normalize(train_trials)
            valid_trials_norm = np.clip(normalize(valid_trials), -1, 1)
            test_trials_norm = np.clip(normalize(test_trials), -1, 1)
            log_message(f"Split complete. Train: {train_trials.shape[0]}, Valid: {valid_trials.shape[0]}, Test: {test_trials.shape[0]}")

            # --- 4. CREATE AVERAGED FEATURES FOR CLASSIFIER ---
            log_message("\n--- Step 4: Creating Averaged Features for LDA Classifier ---")
            train_features = create_averaged_features_from_trials(train_trials_norm, train_labels)
            valid_features = create_averaged_features_from_trials(valid_trials_norm, valid_labels)
            test_features = create_averaged_features_from_trials(test_trials_norm, test_labels)
            log_message(f"Created averaged features. Train: {train_features['x'].shape[2]}, Valid: {valid_features['x'].shape[2]}, Test: {test_features['x'].shape[2]}")

            # --- 5. BASELINE EVALUATION & UNIFIED SELECTION SET ---
            log_message("\n--- Step 5: Baseline Evaluation & Creating Unified Selection Set ---")
            train_valid_features = {
                'x': np.concatenate((train_features['x'], valid_features['x']), axis=2),
                'y': np.concatenate((train_features['y'], valid_features['y']))
            }
            log_message(f"Created combined train+valid feature set with {train_valid_features['x'].shape[2]} averaged trials.")
            
            model_real, sf_real = train_p300_lda(train_valid_features)
            accuracy_baseline = evaluate_p300_lda(model_real, sf_real, test_features)
            log_message(f"Baseline Accuracy (Train+Valid -> Test): {accuracy_baseline:.2f}%")

            # --- 6. GAN TRAINING & BATCH GENERATION/SELECTION ---
            all_generated_batches = []
            for run_idx in range(NUM_RUNS_PER_SUBJECT):
                run_number = run_idx + 1
                generator = train_wgan_gp_cgan(train_trials_norm, train_labels, GAN_EPOCHS, BATCH_SIZE_GAN, subject_id_str, run_number, output_dir)
                log_message(f"\n--- Generating & Evaluating Batches with ERP Similarity (Run {run_number}) ---")

                # For batch selection, we generate a consistent number of trials
                num_synth_target_eval = NUM_TRAIN_TARGET_TRIALS + NUM_VALID_TARGET_TRIALS
                num_synth_nontarget_eval = num_train_nontarget_trials + num_valid_nontarget_trials

                for batch_idx in range(NUM_GENERATION_BATCHES):
                    synth_c1 = generate_synthetic_data_cgan(generator, num_synth_target_eval, 1)
                    synth_c2 = generate_synthetic_data_cgan(generator, num_synth_nontarget_eval, 2)
                    if not (synth_c1.size > 0 and synth_c2.size > 0):
                        log_message(f"    Run {run_number}, Batch {batch_idx+1}: Failed to generate synthetic data.")
                        continue

                    synth_trials_all = np.concatenate((synth_c1, synth_c2), axis=0)
                    synth_labels_all = np.array([1]*len(synth_c1) + [2]*len(synth_c2))

                    # Evaluate batch by comparing its ERP to the real validation data's ERP
                    similarity_score = calculate_erp_similarity_score(
                        valid_trials_norm, valid_labels,
                        synth_trials_all, synth_labels_all
                    )
                    log_message(f"    Run {run_number}, Batch {batch_idx+1}: ERP Similarity Score: {similarity_score:.4f}")

                    # Create averaged features for this batch for later use
                    synth_features = create_averaged_features_from_trials(synth_trials_all, synth_labels_all)
                    if synth_features['x'].shape[2] == 0:
                        log_message(f"    Run {run_number}, Batch {batch_idx+1}: Failed to create averaged features.")
                        continue

                    all_generated_batches.append({
                        'run_index': run_number,
                        'batch_index': batch_idx + 1,
                        'selection_score': similarity_score, # Higher is better
                        'synthetic_trials': synth_trials_all,
                        'synthetic_labels': synth_labels_all,
                        'synthetic_features': synth_features
                    })

            # --- 7. SELECTING BEST STRATEGY (TOP-10 & AUGMENTATION) ---
            log_message("\n\n--- Step 7: Selecting Best Augmentation Strategy ---")
            if not all_generated_batches:
                log_message("CRITICAL ERROR: No synthetic batches were generated or evaluated. Cannot proceed."); continue

            # 1. Select top 10 synthetic batches based on ERP similarity score
            all_generated_batches.sort(key=lambda x: x['selection_score'], reverse=True)
            top_10_batches = all_generated_batches[:10]
            log_message(f"Selected top {len(top_10_batches)} synthetic batches based on ERP similarity to the validation set.")
            for i, batch in enumerate(top_10_batches):
                log_message(f"  Top {i+1}: Run {batch['run_index']}, Batch {batch['batch_index']}, Score: {batch['selection_score']:.4f}")

            # 2. Evaluate 'synth-only' and 'augmented' strategies for these top 10 batches on the validation set
            all_evaluated_strategies = []
            mix_ratios = [0, 0.25, 0.50, 1.0]  # 0 represents 'synth-only'
            
            log_message("\n--- Evaluating strategies on the validation set using classification accuracy ---")
            for batch_info in top_10_batches:
                synth_features_for_aug = batch_info['synthetic_features']
                for ratio in mix_ratios:
                    if ratio == 0:  # 'synth-only' strategy
                        training_features = synth_features_for_aug
                        strategy_name = 'Synth Only'
                    else:  # 'augmented' strategy
                        strategy_name = f"Augmented ({int(ratio*100)}%)"
                        # Augment the REAL TRAINING data
                        num_real_features = train_features['x'].shape[2]
                        num_synth_to_add = int(num_real_features * ratio)
                        if num_synth_to_add == 0: continue

                        available_synth_features = synth_features_for_aug['x'].shape[2]
                        if num_synth_to_add > available_synth_features:
                            num_synth_to_add = available_synth_features

                        synth_x_to_add = synth_features_for_aug['x'][:, :, :num_synth_to_add]
                        synth_y_to_add = synth_features_for_aug['y'][:num_synth_to_add]

                        augmented_x = np.concatenate((train_features['x'], synth_x_to_add), axis=2)
                        augmented_y = np.concatenate((train_features['y'], synth_y_to_add))
                        training_features = {'x': augmented_x, 'y': augmented_y}

                    # Train on the strategy's training set and evaluate on the REAL VALIDATION set
                    model_strategy, sf_strategy = train_p300_lda(training_features)
                    validation_accuracy = evaluate_p300_lda(model_strategy, sf_strategy, valid_features)

                    log_message(f"    Run {batch_info['run_index']}, Batch {batch_info['batch_index']}, Strategy '{strategy_name}': Validation Acc: {validation_accuracy:.2f}%")
                    all_evaluated_strategies.append({
                        'run_index': batch_info['run_index'],
                        'batch_index': batch_info['batch_index'],
                        'mix_ratio': ratio,
                        'validation_accuracy': validation_accuracy,
                        'selection_score': batch_info['selection_score'],
                        'synthetic_trials': batch_info['synthetic_trials'],
                        'synthetic_labels': batch_info['synthetic_labels'],
                        'synthetic_features': batch_info['synthetic_features']
                    })

            # 3. Find the best overall strategy based on validation accuracy
            if not all_evaluated_strategies:
                log_message("CRITICAL ERROR: No strategies could be evaluated. Cannot proceed."); continue

            # Find the maximum validation accuracy achieved
            max_validation_accuracy = max(s['validation_accuracy'] for s in all_evaluated_strategies)

            # Get all strategies that achieved this top accuracy
            top_accuracy_strategies = [s for s in all_evaluated_strategies if s['validation_accuracy'] == max_validation_accuracy]

            log_message(f"\nFound {len(top_accuracy_strategies)} strategies with the top validation accuracy of {max_validation_accuracy:.2f}%.")

            # If multiple strategies have the same top accuracy, use ERP similarity as a tie-breaker
            if len(top_accuracy_strategies) > 1:
                log_message("Using ERP similarity score as a tie-breaker...")
                # Sort the tied strategies by their selection_score (higher is better)
                top_accuracy_strategies.sort(key=lambda x: x['selection_score'], reverse=True)
                for s in top_accuracy_strategies:
                    log_message(f"  - Strategy (Run {s['run_index']}, Batch {s['batch_index']}, Ratio {s['mix_ratio']}): Accuracy={s['validation_accuracy']:.2f}, ERP Score={s['selection_score']:.4f}")

            # The best strategy is the first one in the (potentially sorted) list
            best_strategy_details = top_accuracy_strategies[0]
            
            best_ratio = best_strategy_details['mix_ratio']
            best_strategy_name = 'Synth Only' if best_ratio == 0 else f"Augmented ({int(best_ratio*100)}%)"

            log_message(f"\nSelected best strategy: Run {best_strategy_details['run_index']}, Batch {best_strategy_details['batch_index']}, Strategy: {best_strategy_name} with a validation accuracy of {best_strategy_details['validation_accuracy']:.2f}%")
            log_message("This strategy will now be evaluated on the unseen test set.")

            # --- 8. FINAL EVALUATION ON TEST SET ---
            log_message(f"\n\n--- Step 8: Final Evaluation of Best Strategies on Test Set ---")

            # A) Evaluate the best 'synth-only' strategy on the test set for comparison
            synth_only_strategies = [s for s in all_evaluated_strategies if s['mix_ratio'] == 0]
            if not synth_only_strategies:
                log_message("Warning: No synth-only strategies were evaluated.")
                accuracy_best_synth_on_test = 0.0
            else:
                best_synth_only_strategy = max(synth_only_strategies, key=lambda x: x['validation_accuracy'])
                # Train on synth data, test on real test data
                model_best_synth, sf_best_synth = train_p300_lda(best_synth_only_strategy['synthetic_features'])
                accuracy_best_synth_on_test = evaluate_p300_lda(model_best_synth, sf_best_synth, test_features)
                log_message(f"BEST SYNTHETIC-ONLY (Val Acc: {best_synth_only_strategy['validation_accuracy']:.2f}%) -> REAL test accuracy: {accuracy_best_synth_on_test:.2f}%")

            # B) Evaluate the overall best strategy on the test set
            # For final evaluation, retrain the model using the best strategy on the combined (real train + real valid) data
            final_model_train_features = None
            best_mix_ratio = best_strategy_details['mix_ratio']
            
            log_message("Retraining best model on combined Train+Validation data for final evaluation.")
            base_real_features_final = train_valid_features

            if best_mix_ratio == 0:  # Best strategy was synth-only
                final_model_train_features = best_strategy_details['synthetic_features']
            else:  # Best strategy was augmentation
                synth_features = best_strategy_details['synthetic_features']
                num_real = base_real_features_final['x'].shape[2]
                num_synth_to_add = int(num_real * best_mix_ratio)
                
                if num_synth_to_add > synth_features['x'].shape[2]:
                    num_synth_to_add = synth_features['x'].shape[2]
                
                synth_x_to_add = synth_features['x'][:, :, :num_synth_to_add]
                synth_y_to_add = synth_features['y'][:num_synth_to_add]
                
                aug_x = np.concatenate((base_real_features_final['x'], synth_x_to_add), axis=2)
                aug_y = np.concatenate((base_real_features_final['y'], synth_y_to_add))
                final_model_train_features = {'x': aug_x, 'y': aug_y}

            model_final, sf_final = train_p300_lda(final_model_train_features)
            accuracy_best_strategy_on_test = evaluate_p300_lda(model_final, sf_final, test_features)
            
            best_strategy_name_final = 'Synth Only' if best_mix_ratio == 0 else f"Augmented ({int(best_mix_ratio*100)}%)"
            log_message(f"BEST OVERALL STRATEGY ({best_strategy_name_final}, Val Acc: {best_strategy_details['validation_accuracy']:.2f}%) -> REAL test accuracy: {accuracy_best_strategy_on_test:.2f}%")

            # --- 9. FINAL PLOTTING & SUMMARY ---
            log_message(f"\n--- Step 9: Final Plotting and Summary ---")

            # Save the target and non-target classes of the best synthetic batch and training data
            best_overall_synth_trials_for_saving = best_strategy_details['synthetic_trials']
            best_overall_synth_labels_for_saving = best_strategy_details['synthetic_labels']

            # Save only the target class from the best synthetic batch
            target_synth_indices = np.where(best_overall_synth_labels_for_saving == 1)[0]
            target_synth_trials = best_overall_synth_trials_for_saving[target_synth_indices]
            synthetic_target_filename = os.path.join(output_dir, f"target_synthetic_data_S{subject_id_str}.mat")
            scipy.io.savemat(synthetic_target_filename, {'target_synthetic_data': target_synth_trials})
            log_message(f"Saved target class of best synthetic batch to {synthetic_target_filename}")

            # Save only the non-target class from the best synthetic batch
            nontarget_synth_indices = np.where(best_overall_synth_labels_for_saving == 2)[0]
            nontarget_synth_trials = best_overall_synth_trials_for_saving[nontarget_synth_indices]
            synthetic_nontarget_filename = os.path.join(output_dir, f"nontarget_synthetic_data_S{subject_id_str}.mat")
            scipy.io.savemat(synthetic_nontarget_filename, {'nontarget_synthetic_data': nontarget_synth_trials})
            log_message(f"Saved non-target class of best synthetic batch to {synthetic_nontarget_filename}")

            # Save only the target class from the training data
            target_train_indices = np.where(train_labels == 1)[0]
            target_train_trials = train_trials_norm[target_train_indices]
            training_target_filename = os.path.join(output_dir, f"target_training_data_S{subject_id_str}.mat")
            scipy.io.savemat(training_target_filename, {
                'target_training_data': target_train_trials,
                'train_min': train_min,
                'train_max': train_max
            })
            log_message(f"Saved target class of training data to {training_target_filename}")

            # Save only the non-target class from the training data
            nontarget_train_indices = np.where(train_labels == 2)[0]
            nontarget_train_trials = train_trials_norm[nontarget_train_indices]
            training_nontarget_filename = os.path.join(output_dir, f"nontarget_training_data_S{subject_id_str}.mat")
            scipy.io.savemat(training_nontarget_filename, {
                'nontarget_training_data': nontarget_train_trials,
                'train_min': train_min,
                'train_max': train_max
            })
            log_message(f"Saved non-target class of training data to {training_nontarget_filename}")
            
            accuracies_summary = {'Baseline': accuracy_baseline, 'Synthetic Only': accuracy_best_synth_on_test}
            if best_mix_ratio != 0:
                label = f"Augmented ({int(best_mix_ratio*100)}%)"
                accuracies_summary[label] = accuracy_best_strategy_on_test
            
            plot_accuracy_chart(accuracies_summary, subject_id_str, output_dir)

            # Prepare data for grand average plots using the synth data from the best overall strategy
            best_overall_synth_trials = best_strategy_details['synthetic_trials']
            best_overall_synth_labels = best_strategy_details['synthetic_labels']
            
            # Create 100% augmented data for plotting using the best batch
            aug_plot_trials = np.concatenate((train_trials_norm, best_overall_synth_trials), axis=0)
            aug_plot_labels = np.concatenate((train_labels, best_overall_synth_labels))

            plot_channel_idx_to_plot = 1  # Cz
            plot_grand_average_comparison(
                real_data_train_plot=train_trials_norm, real_labels_train_plot=train_labels,
                synthetic_data_plot=best_overall_synth_trials, synthetic_labels_plot=best_overall_synth_labels,
                augmented_data=aug_plot_trials, augmented_labels=aug_plot_labels,
                channel_idx_plot=plot_channel_idx_to_plot,
                output_dir_param=output_dir, subject_display_idx_param=subject_id_str,
                train_min=train_min, train_max=train_max
            )
            log_message(f"\n--- Subject {subject_id_str} processing finished successfully. ---")
        except Exception as e:
            log_message(f"\n--- CRITICAL ERROR processing subject {subject_id_str} ---\nError: {e}\n{traceback.format_exc()}\n--- Skipping to next subject. ---")

if __name__ == '__main__':
    data_folders_to_process = [f'H{i}' for i in range(1, 13)]

    for data_folder in data_folders_to_process:
        print(f"\n\n{'='*30}\nProcessing data from folder: {data_folder}\n{'='*30}")
        
        # Check for data or create dummy files for demonstration
        if not os.path.exists(data_folder):
            print(f"Folder '{data_folder}' not found. Creating for demonstration purposes.")
            os.makedirs(data_folder)

        first_gdf_file = os.path.join(data_folder, 'H01.gdf')
        if not os.path.exists(first_gdf_file):
            print(f"Data not found in '{data_folder}'. Creating dummy GDF files...")
            for i in range(1, 9):
                with open(os.path.join(data_folder, f'H0{i}.gdf'), 'w') as f:
                    pass # Create empty files
        
        try:
            main_analysis(data_folder)
        except Exception as e:
            # This top-level catch is good practice.
            print(f"\n--- SCRIPT EXECUTION FAILED FOR {data_folder} ---\nAn unexpected error occurred: {e}\n{traceback.format_exc()}")

    print("\n\n--- All data folders processed. ---")