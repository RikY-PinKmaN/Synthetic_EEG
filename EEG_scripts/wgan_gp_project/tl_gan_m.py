# =================================================================================================
#
#       P300 META-TRANSFER LEARNING & SYNTHETIC DATA GENERATION PIPELINE (v3 - Deadlock Fix)
#
# This script integrates a Leave-One-Subject-Out (LOSO) Transfer Learning approach with
# a Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP) for P300 data augmentation.
#
# V3 CORRECTIONS:
# - Fixed a critical process deadlock that caused the script to hang after the first GAN epoch.
# - The issue was a conflict between the multiprocessing workers used by TensorFlow's data
#   pipeline and the `joblib` workers used by `mlxtend`'s SequentialFeatureSelector.
# - The fix is to set `n_jobs=1` in the SFS constructor, forcing it to run sequentially
#   and preventing the multiprocessing conflict.
#
# =================================================================================================

import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")

# --- Configuration ---
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Data Configuration
SESSION_IDS = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
CH_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz']
BASE_MODEL_DIR = "tl_gan_base_models_v3"

# Preprocessing Parameters
FS = 80
ARTIFTH = 50
TS_START_SEC, TS_END_SEC = -0.1, 0.6
BASELINE_START_SEC, BASELINE_END_SEC = -0.1, 0.0

# Data Splitting Parameters
NUM_TRAIN_TARGET_TRIALS = 60
NUM_TRAIN_NONTARGET_TRIALS = 120
NUM_VALID_TARGET_TRIALS = 15
NUM_VALID_NONTARGET_TRIALS = 45

# GAN & Evaluation Parameters
LATENT_DIM_CGAN = 100
NUM_CLASSES_CGAN = 2
EMBEDDING_DIM_CGAN = 25
GAN_PRETRAIN_EPOCHS = 1500
GAN_FINETUNE_EPOCHS = 500
BATCH_SIZE_GAN = 64
FREQ_LOSS_WEIGHT = 0.4
N_CRITIC_STEPS = 5
NUM_GENERATION_BATCHES = 30

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

# =============================================================================
# PART 2: P300 CLASSIFIER & FEATURE ENGINEERING (Corrected)
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
    
    # ==================================================================
    #                           *** FIX ***
    # Changed n_jobs=-1 to n_jobs=1 to prevent a multiprocessing
    # deadlock between mlxtend/joblib and TensorFlow's data workers.
    # ==================================================================
    sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), k_features=5, forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=1)
    
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
# PART 3: CONDITIONAL GAN (cWGAN-GP) FUNCTIONS (Unchanged)
# =============================================================================

def smooth_eeg(data, window_size=5):
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

def build_generator_cgan(num_channels, time_points, latent_dim=LATENT_DIM_CGAN, num_classes=NUM_CLASSES_CGAN, embedding_dim=EMBEDDING_DIM_CGAN):
    # This function is identical to the one in n_p300.py
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
    # This function is identical to the one in n_p300.py
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
    # This function is identical to the one in n_p300.py
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return lambda_gp * gp

def train_or_finetune_wgan(eeg_data, eeg_labels, epochs, batch_size, log_prefix, output_dir, initial_generator=None):
    num_channels, time_points = eeg_data.shape[1], eeg_data.shape[2]
    labels_0_1 = (eeg_labels.squeeze() - 1).astype(np.int32)
    labels_0_1_reshaped = labels_0_1[:, np.newaxis]

    if initial_generator:
        log_message(f"--- Fine-tuning existing generator for {epochs} epochs ---")
        generator = initial_generator
    else:
        log_message(f"--- Training new generator from scratch for {epochs} epochs ---")
        generator = build_generator_cgan(num_channels, time_points)

    critic = build_critic_cgan(num_channels, time_points)
    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step_wgan_cgan(real_eeg_batch, real_labels_batch_0_1_tf):
        batch_size_tf = tf.shape(real_eeg_batch)[0]
        for _ in range(N_CRITIC_STEPS):
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch_size_tf, LATENT_DIM_CGAN])
                fake_eeg = generator([noise, real_labels_batch_0_1_tf], training=True)
                real_output = critic([real_eeg_batch, real_labels_batch_0_1_tf], training=True)
                fake_output = critic([fake_eeg, real_labels_batch_0_1_tf], training=True)
                gp = gradient_penalty_cgan(critic, real_eeg_batch, fake_eeg, real_labels_batch_0_1_tf)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
            gradients_d = tape.gradient(d_loss, critic.trainable_variables)
            crit_opt.apply_gradients(zip(gradients_d, critic.trainable_variables))
        
        with tf.GradientTape() as tape_g:
            noise_g = tf.random.normal([batch_size_tf, LATENT_DIM_CGAN])
            fake_eeg_for_g = generator([noise_g, real_labels_batch_0_1_tf], training=True)
            fake_output_for_g = critic([fake_eeg_for_g, real_labels_batch_0_1_tf], training=True)
            g_loss_wasserstein = -tf.reduce_mean(fake_output_for_g)
        gradients_g = tape_g.gradient(g_loss_wasserstein, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))
        return d_loss, g_loss_wasserstein

    d_losses, g_losses = [], []
    dataset = tf.data.Dataset.from_tensor_slices((eeg_data.astype(np.float32), labels_0_1_reshaped.astype(np.int32)))
    dataset = dataset.shuffle(1024).batch(batch_size).repeat()
    num_batches_per_epoch = len(eeg_data) // batch_size

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss_comb = 0.0, 0.0
        for i, (real_eeg_batch, real_labels_batch) in enumerate(dataset.take(num_batches_per_epoch)):
            d_loss, g_loss_base = train_step_wgan_cgan(real_eeg_batch, real_labels_batch)
            current_g_loss_total_for_batch = g_loss_base.numpy()
            
            if FREQ_LOSS_WEIGHT > 0:
                with tf.GradientTape() as freq_tape:
                    noise_for_freq = tf.random.normal([tf.shape(real_eeg_batch)[0], LATENT_DIM_CGAN])
                    fake_data_for_freq_loss = generator([noise_for_freq, real_labels_batch], training=True)
                    freq_loss_val = frequency_domain_loss(real_eeg_batch, fake_data_for_freq_loss)
                    weighted_freq_loss = FREQ_LOSS_WEIGHT * freq_loss_val
                freq_gradients = freq_tape.gradient(weighted_freq_loss, generator.trainable_variables)
                if not any(g is None for g in freq_gradients):
                    gen_opt.apply_gradients(zip(freq_gradients, generator.trainable_variables))
                current_g_loss_total_for_batch += weighted_freq_loss.numpy()

            epoch_d_loss += d_loss.numpy()
            epoch_g_loss_comb += current_g_loss_total_for_batch
        
        avg_d_loss = epoch_d_loss / num_batches_per_epoch
        avg_g_loss = epoch_g_loss_comb / num_batches_per_epoch
        d_losses.append(avg_d_loss); g_losses.append(avg_g_loss)
        if epoch % 200 == 0 or epoch == epochs - 1:
            log_message(f"({log_prefix}) Epoch {epoch}/{epochs}: D Loss={avg_d_loss:.4f}, G Loss (Comb)={avg_g_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Critic Loss"); plt.plot(g_losses, label="Gen Loss (W+Freq)")
    plt.title(f"cGAN Losses - {log_prefix}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"GAN_Losses_{log_prefix}.png")); plt.close()
    return generator

def generate_synthetic_data_cgan(generator, num_samples, target_label_1_2):
    if generator is None or num_samples == 0: return np.array([])
    target_label_0_1 = target_label_1_2 - 1
    noise = tf.random.normal([num_samples, LATENT_DIM_CGAN])
    labels_for_gen = tf.ones((num_samples, 1), dtype=tf.int32) * target_label_0_1
    synthetic_data = generator([noise, labels_for_gen], training=False).numpy()
    if synthetic_data.size > 0:
        synthetic_data = smooth_eeg(synthetic_data)
    return synthetic_data

# =============================================================================
# PART 4: PLOTTING AND ANALYSIS FUNCTIONS (Simplified)
# =============================================================================

def plot_accuracy_chart(accuracies, subject_id_str, output_dir):
    labels, values = list(accuracies.keys()), list(accuracies.values())
    plt.figure(figsize=(10, 7)); bars = plt.bar(labels, values)
    plt.ylabel('Classification Accuracy (%)'); plt.title(f'S{subject_id_str}: Classification Accuracy Comparison')
    plt.ylim(0, 100); plt.xticks(rotation=15, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Accuracy_Comparison_S{subject_id_str}.png")); plt.close()

# =============================================================================
# PART 5: DATA LOADING & MAIN EXPERIMENT SCRIPT
# =============================================================================

def load_and_preprocess_single_subject(data_path):
    all_trials, all_labels = [], []
    for s in range(0, 7, 2):
        try:
            raw1 = mne.io.read_raw_gdf(os.path.join(data_path, f"{SESSION_IDS[s]}.gdf"), preload=True, verbose=False)
            raw2 = mne.io.read_raw_gdf(os.path.join(data_path, f"{SESSION_IDS[s+1]}.gdf"), preload=True, verbose=False)
            raw1.rename_channels({old: new for old, new in zip(raw1.ch_names, CH_NAMES)})
            raw2.rename_channels({old: new for old, new in zip(raw2.ch_names, CH_NAMES)})
            raw = mne.concatenate_raws([raw1, raw2], verbose=False)
            original_fs = raw.info['sfreq']
            data = raw.get_data()
            b_hp, a_hp = butter(4, 0.4 / (original_fs / 2), 'high')
            b_lp, a_lp = butter(4, 30 / (original_fs / 2), 'low')
            b_n1, a_n1 = iirnotch(4.3 if s in [0,4] else 5.8, 5, original_fs)
            b_n2, a_n2 = iirnotch((4.3 if s in [0,4] else 5.8)*2, 5, original_fs)
            for j in range(data.shape[0]):
                data[j, :] = filtfilt(b_hp, a_hp, filtfilt(b_lp, a_lp, data[j, :]))
                data[j, :] = filtfilt(b_n1, a_n1, filtfilt(b_n2, a_n2, data[j, :]))
            raw._data = data; raw.resample(FS, verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
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
            
            if clean_trials_target and clean_trials_nontarget:
                all_trials.append(np.array(clean_trials_target + clean_trials_nontarget))
                all_labels.append(np.array([1]*len(clean_trials_target) + [2]*len(clean_trials_nontarget)))
        except Exception as e:
            print(f"Warning: Could not process session pair in {data_path}. Error: {e}")
            
    if not all_trials: return None, None
    return np.concatenate(all_trials, axis=0), np.concatenate(all_labels, axis=0)

def target_subject_analysis(target_subject_id, target_trials, target_labels, fine_tuned_generator):
    base_output_dir = "results_meta_tl_gan_v3"
    output_dir = os.path.join(base_output_dir, f"Target_{target_subject_id}_results")
    os.makedirs(output_dir, exist_ok=True)

    global current_log_file_path
    current_log_file_path = os.path.join(output_dir, "run_log.txt")
    with open(current_log_file_path, 'w') as f: f.write(f"Log for Target {target_subject_id} using Meta-TL-GAN\n{'='*60}\n")
    log_message(f"\n{'='*25} ANALYZING TARGET SUBJECT: {target_subject_id} {'='*25}")

    try:
        log_message("\n--- Step 1: Data Split, Feature Creation & Baseline Eval ---")
        idx_c1 = np.where(target_labels == 1)[0]; idx_c2 = np.where(target_labels == 2)[0]
        total_needed_c1 = NUM_TRAIN_TARGET_TRIALS + NUM_VALID_TARGET_TRIALS
        total_needed_c2 = NUM_TRAIN_NONTARGET_TRIALS + NUM_VALID_NONTARGET_TRIALS
        if len(idx_c1) < total_needed_c1 or len(idx_c2) < total_needed_c2:
            log_message(f"CRITICAL: Not enough trials for split. Skipping."); return
        
        train_idx = np.concatenate((idx_c1[:NUM_TRAIN_TARGET_TRIALS], idx_c2[:NUM_TRAIN_NONTARGET_TRIALS]))
        valid_idx = np.concatenate((idx_c1[NUM_TRAIN_TARGET_TRIALS:total_needed_c1], idx_c2[NUM_TRAIN_NONTARGET_TRIALS:total_needed_c2]))
        test_idx = np.concatenate((idx_c1[total_needed_c1:], idx_c2[total_needed_c2:]))
        
        train_trials, train_labels = target_trials[train_idx], target_labels[train_idx]
        valid_trials, valid_labels = target_trials[valid_idx], target_labels[valid_idx]
        test_trials, test_labels = target_trials[test_idx], target_labels[test_idx]

        train_min, train_max = np.min(train_trials), np.max(train_trials)
        normalize = lambda x: 2 * (x - train_min) / (train_max - train_min) - 1
        train_trials_norm = normalize(train_trials)
        valid_trials_norm = np.clip(normalize(valid_trials), -1, 1)
        test_trials_norm = np.clip(normalize(test_trials), -1, 1)
        
        train_features = create_averaged_features_from_trials(train_trials_norm, train_labels)
        valid_features = create_averaged_features_from_trials(valid_trials_norm, valid_labels)
        test_features = create_averaged_features_from_trials(test_trials_norm, test_labels)

        train_valid_features = {
            'x': np.concatenate((train_features['x'], valid_features['x']), axis=2),
            'y': np.concatenate((train_features['y'], valid_features['y']))
        }
        
        model_real, sf_real = train_p300_lda(train_valid_features)
        accuracy_baseline = evaluate_p300_lda(model_real, sf_real, test_features)
        log_message(f"Baseline Accuracy (Train+Valid -> Test): {accuracy_baseline:.2f}%")

        log_message(f"\n--- Step 2: Generating & Evaluating Batches with Fine-Tuned GAN ---")
        all_batch_results = []
        num_synth_target = NUM_TRAIN_TARGET_TRIALS + NUM_VALID_TARGET_TRIALS
        num_synth_nontarget = NUM_TRAIN_NONTARGET_TRIALS + NUM_VALID_NONTARGET_TRIALS

        for batch_idx in range(NUM_GENERATION_BATCHES):
            synth_c1 = generate_synthetic_data_cgan(fine_tuned_generator, num_synth_target, 1)
            synth_c2 = generate_synthetic_data_cgan(fine_tuned_generator, num_synth_nontarget, 2)
            if synth_c1.size > 0 and synth_c2.size > 0:
                synth_trials_all = np.concatenate((synth_c1, synth_c2), axis=0)
                synth_labels_all = np.array([1]*len(synth_c1) + [2]*len(synth_c2))
                synth_features = create_averaged_features_from_trials(synth_trials_all, synth_labels_all)
                if synth_features['x'].shape[2] == 0: continue

                model_synth, sf_synth = train_p300_lda(synth_features)
                selection_acc = evaluate_p300_lda(model_synth, sf_synth, train_valid_features)
                
                log_message(f"  Batch {batch_idx+1}: Selection Acc (on Train+Valid): {selection_acc:.2f}%")
                all_batch_results.append({'selection_accuracy': selection_acc, 'synthetic_trials': synth_trials_all})

        log_message("\n--- Step 3: Final Evaluation and Plotting ---")
        if not all_batch_results:
            log_message("CRITICAL: No synthetic batches generated."); return

        all_batch_results.sort(key=lambda x: x['selection_accuracy'], reverse=True)
        best_batch = all_batch_results[0]
        log_message(f"Selected best batch with Selection Accuracy: {best_batch['selection_accuracy']:.2f}%")
        
        best_synth_trials = best_batch['synthetic_trials']
        best_synth_labels = np.array([1]*num_synth_target + [2]*num_synth_nontarget)
        best_synth_features = create_averaged_features_from_trials(best_synth_trials, best_synth_labels)
        
        model_best_synth, sf_best_synth = train_p300_lda(best_synth_features)
        accuracy_synthetic_best = evaluate_p300_lda(model_best_synth, sf_best_synth, test_features)

        accuracies_summary = {'Real Only': accuracy_baseline, 'Synth Only': accuracy_synthetic_best}
        
        for ratio in [0.25, 0.50, 1.0]:
            num_real_features = train_valid_features['x'].shape[2]
            num_synth_to_add = int(num_real_features * ratio)
            if num_synth_to_add == 0 or best_synth_features['x'].shape[2] < num_synth_to_add: continue
            
            synth_x_to_add = best_synth_features['x'][:,:,:num_synth_to_add]
            synth_y_to_add = best_synth_features['y'][:num_synth_to_add]
            aug_features = {
                'x': np.concatenate((train_valid_features['x'], synth_x_to_add), axis=2),
                'y': np.concatenate((train_valid_features['y'], synth_y_to_add))
            }

            model_aug, sf_aug = train_p300_lda(aug_features)
            accuracy_augmented = evaluate_p300_lda(model_aug, sf_aug, test_features)
            
            label = f'Real + {int(ratio*100)}% Synth'
            accuracies_summary[label] = accuracy_augmented
            log_message(f"  Accuracy ({label}): {accuracy_augmented:.2f}%")

        plot_accuracy_chart(accuracies_summary, target_subject_id, output_dir)
        log_message(f"\n--- Subject {target_subject_id} processing finished successfully. ---")

    except Exception as e:
        log_message(f"\n--- CRITICAL ERROR in target_subject_analysis for {target_subject_id} ---\nError: {e}\n{traceback.format_exc()}\n")


def run_loso_meta_experiment(all_subject_folders):
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    
    for target_subject_folder in all_subject_folders:
        target_subject_id = os.path.basename(target_subject_folder)
        source_subject_folders = [f for f in all_subject_folders if f != target_subject_folder]
        print(f"\n\n{'='*30}\nTARGET SUBJECT: {target_subject_id}\n{'='*30}")

        print(f"\n--- Step 1: Pre-training Base GAN on {len(source_subject_folders)} source subjects ---")
        base_model_path = os.path.join(BASE_MODEL_DIR, f"base_generator_except_{target_subject_id}.keras")
        
        if os.path.exists(base_model_path):
            print(f"Loading existing pre-trained model: {base_model_path}")
            base_generator = models.load_model(base_model_path)
        else:
            print("Aggregating source data...")
            source_trials_list, source_labels_list = [], []
            for source_folder in source_subject_folders:
                trials, labels = load_and_preprocess_single_subject(source_folder)
                if trials is not None:
                    # ================== OPTIMIZATION 1A ==================
                    # Convert to float32 as soon as data is loaded
                    source_trials_list.append(trials.astype(np.float32))
                    source_labels_list.append(labels)
            
            if not source_trials_list: 
                print(f"CRITICAL: No source data. Skipping target {target_subject_id}.")
                continue

            source_trials = np.concatenate(source_trials_list, axis=0)
            source_labels = np.concatenate(source_labels_list, axis=0)
            del source_trials_list # Free memory from the list of arrays
            gc.collect()

            source_min, source_max = np.min(source_trials), np.max(source_trials)
            source_trials_norm = 2 * (source_trials - source_min) / (source_max - source_min) - 1
            
            # ================== OPTIMIZATION 1B ==================
            # Ensure the final normalized array is also float32
            source_trials_norm = source_trials_norm.astype(np.float32)
            del source_trials # Free memory from the non-normalized combined array
            gc.collect()
            
            # Pass data to the training function
            base_generator = train_or_finetune_wgan(
                source_trials_norm, source_labels, GAN_PRETRAIN_EPOCHS, BATCH_SIZE_GAN,
                f"Pre-training_except_{target_subject_id}", BASE_MODEL_DIR
            )
            
            # ================== OPTIMIZATION 2 ==================
            # After training, the huge array is no longer needed at all.
            del source_trials_norm
            del source_labels
            gc.collect()
            
            base_generator.save(base_model_path)

        print(f"\n--- Step 2: Fine-tuning GAN for {target_subject_id} ---")
        # For fine-tuning, the data is much smaller, but we apply the same good practice.
        target_trials, target_labels = load_and_preprocess_single_subject(target_subject_folder)
        if target_trials is None: 
            print(f"CRITICAL: No target data. Skipping.")
            continue
        
        target_trials = target_trials.astype(np.float32) # Use float32
        
        idx_c1 = np.where(target_labels == 1)[0]; idx_c2 = np.where(target_labels == 2)[0]
        if len(idx_c1) < NUM_TRAIN_TARGET_TRIALS or len(idx_c2) < NUM_TRAIN_NONTARGET_TRIALS:
            print(f"CRITICAL: Not enough training trials for fine-tuning. Skipping.")
            continue
        
        train_idx = np.concatenate((idx_c1[:NUM_TRAIN_TARGET_TRIALS], idx_c2[:NUM_TRAIN_NONTARGET_TRIALS]))
        finetune_trials, finetune_labels = target_trials[train_idx], target_labels[train_idx]
        
        finetune_min, finetune_max = np.min(finetune_trials), np.max(finetune_trials)
        finetune_trials_norm = 2 * (finetune_trials - finetune_min) / (finetune_max - finetune_min) - 1
        
        output_dir_finetune = os.path.join("results_meta_tl_gan_v4", f"Target_{target_subject_id}_results")
        os.makedirs(output_dir_finetune, exist_ok=True)
        
        fine_tuned_generator = train_or_finetune_wgan(
            finetune_trials_norm.astype(np.float32), finetune_labels, GAN_FINETUNE_EPOCHS, BATCH_SIZE_GAN,
            f"Finetuning_for_{target_subject_id}", output_dir_finetune,
            initial_generator=base_generator
        )

        target_subject_analysis(target_subject_id, target_trials, target_labels, fine_tuned_generator)


if __name__ == '__main__':
    # Define the list of all subject data folders to be processed in the experiment.
    data_folders_to_process = [f'H{i}' for i in range(1, 13)]

    # This loop is for demonstration and setup purposes.
    # It checks if the required data folders and dummy files exist. If not, it creates them.
    # In a real run, these folders should contain your actual GDF data files.
    for folder in data_folders_to_process:
        if not os.path.isdir(folder):
            print(f"Warning: Subject folder '{folder}' not found. Creating for demonstration.")
            os.makedirs(folder)
            for session_id in SESSION_IDS:
                # Create empty files. The script expects files with these names to exist.
                with open(os.path.join(folder, f'{session_id}.gdf'), 'w') as f:
                    pass

    # This is the main entry point of the script.
    # It starts the Leave-One-Subject-Out cross-validation experiment.
    try:
        run_loso_meta_experiment(data_folders_to_process)
        print("\n\n--- All subjects processed successfully. ---")
    except Exception as e:
        # A top-level catch to report any unexpected critical failures.
        print(f"\n--- SCRIPT EXECUTION FAILED ---")
        print(f"An unexpected critical error occurred: {e}")
        print(traceback.format_exc())