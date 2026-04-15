import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import welch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, optimizers, callbacks, utils

# Suppress TensorFlow clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Define the two groups and their paths
GROUPS = ['Young', 'Elderly']
SAMPLING_RATE = 500      
DURATION_SEC = 150       
SAMPLES_PER_SESSION = int(SAMPLING_RATE * DURATION_SEC)
CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'Oz']
N_CHANNELS = len(CHANNELS)

# --- Preprocessing ---
EPOCH_LENGTH = 2.0       
STRIDE = 2.0             
LOW_CUT = 2.0            
HIGH_CUT = 40.0          
REJECTION_THRESH = 150.0 # uV 

# --- Deep Learning ---
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001

# =============================================================================
# 2. SIGNAL PROCESSING UTILITIES
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def find_marker_column(raw_data, marker_value=5):
    n_cols = raw_data.shape[1]
    # Check backwards (timestamp/marker usually at end)
    for col_idx in range(n_cols - 1, max(n_cols - 5, 7), -1):
        if marker_value in raw_data[:, col_idx]:
            return col_idx
    # Check forwards after EEG channels
    for col_idx in range(N_CHANNELS, n_cols):
        if marker_value in raw_data[:, col_idx]:
            return col_idx
    return -1

def load_and_clean_session(filepath):
    try:
        raw = np.loadtxt(filepath)
        if raw.shape[1] <= N_CHANNELS: return None

        marker_col = find_marker_column(raw, 5)
        if marker_col == -1: return None

        eeg_signal = raw[:, :N_CHANNELS]
        marker_data = raw[:, marker_col]
        
        # Find start marker
        start_indices = np.where(marker_data == 5)[0]
        if len(start_indices) == 0: return None
        
        start_idx = start_indices[0]
        end_idx = start_idx + SAMPLES_PER_SESSION
        
        if end_idx > len(raw):
            sliced_signal = eeg_signal[start_idx:, :]
        else:
            sliced_signal = eeg_signal[start_idx:end_idx, :]

        # Fix units (nV -> uV) if needed
        if np.median(np.abs(sliced_signal)) > 500: 
            sliced_signal = sliced_signal / 1000.0
            
        return butter_bandpass_filter(sliced_signal, LOW_CUT, HIGH_CUT, SAMPLING_RATE)
    except:
        return None

def create_epochs_with_rejection(data, label, subject_id):
    fs = SAMPLING_RATE
    samples = int(EPOCH_LENGTH * fs)
    stride = int(STRIDE * fs)
    X, y, groups = [], [], []
    
    n_epochs = (data.shape[0] - samples) // stride + 1
    
    for i in range(n_epochs):
        start = i * stride
        epoch = data[start:start+samples, :]
        # Reject bad epochs
        if (np.max(epoch) - np.min(epoch)) < REJECTION_THRESH:
            X.append(epoch.T)
            y.append(label)
            groups.append(subject_id)
            
    if len(X) == 0: return np.array([]), np.array([]), np.array([])
    return np.array(X), np.array(y), np.array(groups)

# =============================================================================
# 3. DATA LOADING
# =============================================================================

def build_datasets():
    # Structure: data[group][session]...
    all_data = {}
    
    for group in GROUPS:
        print(f"\n--- Loading Group: {group} ---")
        group_data = {
            'pre': {'X': [], 'y': [], 'groups': [], 'clean_epochs': []},
            'post': {'X': [], 'y': [], 'groups': [], 'clean_epochs': []}
        }
        
        group_dir = os.path.join(os.getcwd(), group)
        if not os.path.exists(group_dir):
            print(f"Warning: Folder '{group}' not found.")
            continue
            
        subjects = sorted([d for d in glob.glob(os.path.join(group_dir, '*')) if os.path.isdir(d)])
        print(f"Found {len(subjects)} subjects in {group}/")

        for sub_idx, sub_path in enumerate(subjects):
            # Use unique subject ID across groups to avoid confusion if needed
            # (Though here we process groups separately)
            
            for f in glob.glob(os.path.join(sub_path, '*.easy')):
                fname = os.path.basename(f).lower()
                
                # Parse Filename
                sess = 'pre' if 'rs1' in fname else 'post' if 'rs2' in fname else None
                cond = 0 if ('o_' in fname or 'rs1o' in fname or 'rs2o' in fname) else 1 if ('c_' in fname or 'rs1c' in fname or 'rs2c' in fname) else None
                
                if sess and cond is not None:
                    data = load_and_clean_session(f)
                    if data is not None:
                        X_ep, y_ep, g_ep = create_epochs_with_rejection(data, cond, sub_idx)
                        
                        if len(X_ep) > 0:
                            group_data[sess]['X'].append(X_ep)
                            group_data[sess]['y'].append(y_ep)
                            group_data[sess]['groups'].append(g_ep)
                            # Store for PSD plotting
                            group_data[sess]['clean_epochs'].append({
                                'data': X_ep, 
                                'condition': cond, 
                                'subject': sub_idx
                            })

        # Concatenate arrays for this group
        for s in ['pre', 'post']:
            if len(group_data[s]['X']) > 0:
                group_data[s]['X'] = np.concatenate(group_data[s]['X'], axis=0)[..., np.newaxis]
                group_data[s]['y'] = np.concatenate(group_data[s]['y'], axis=0)
                group_data[s]['groups'] = np.concatenate(group_data[s]['groups'], axis=0)
        
        all_data[group] = group_data

    return all_data

# =============================================================================
# 4. PLOTTING
# =============================================================================

def get_group_psd(epoch_list):
    """
    Returns median PSDs for Open and Closed conditions.
    Format: (Freqs, Median_Open, IQR_Open_Low, IQR_Open_High, Median_Closed, ...)
    """
    if not epoch_list: return None
    
    psds_open, psds_closed = [], []
    
    for item in epoch_list:
        f, pxx = welch(item['data'], fs=SAMPLING_RATE, nperseg=SAMPLING_RATE, axis=2)
        # Median across epochs -> 1 PSD per subject
        avg_pxx = np.median(pxx, axis=0) 
        
        if item['condition'] == 0: psds_open.append(avg_pxx)
        else: psds_closed.append(avg_pxx)
            
    if not psds_open or not psds_closed: return None
    
    # Arrays: (Subjects, Channels, Freqs)
    arr_open = np.array(psds_open)
    arr_closed = np.array(psds_closed)
    
    # Convert to dB
    db_open = 10 * np.log10(arr_open + 1e-10)
    db_closed = 10 * np.log10(arr_closed + 1e-10)
    
    # Stats (Median & IQR) across Subjects+Channels
    stats = {
        'f': f,
        'open_med': np.median(db_open, axis=(0, 1)),
        'open_25': np.percentile(db_open, 25, axis=(0, 1)),
        'open_75': np.percentile(db_open, 75, axis=(0, 1)),
        'closed_med': np.median(db_closed, axis=(0, 1)),
        'closed_25': np.percentile(db_closed, 25, axis=(0, 1)),
        'closed_75': np.percentile(db_closed, 75, axis=(0, 1)),
    }
    return stats

def plot_psds(all_data):
    print("\nGenerating PSD Plots...")
    
    for session in ['pre', 'post']:
        
        # 1. Plot Within-Group (Open vs Closed)
        for group in GROUPS:
            stats = get_group_psd(all_data[group][session]['clean_epochs'])
            if not stats: continue
            
            f = stats['f']
            idx = np.where((f >= 2) & (f <= 40))[0]
            
            plt.figure(figsize=(10, 6))
            
            # Shading (No Label)
            plt.fill_between(f[idx], stats['open_25'][idx], stats['open_75'][idx], color='#3498db', alpha=0.2)
            plt.fill_between(f[idx], stats['closed_25'][idx], stats['closed_75'][idx], color='#e74c3c', alpha=0.2)
            
            # Lines
            plt.plot(f[idx], stats['open_med'][idx], label='Eyes Open', color='#3498db', linewidth=2)
            plt.plot(f[idx], stats['closed_med'][idx], label='Eyes Closed', color='#e74c3c', linewidth=2)
            
            plt.axvspan(8, 12, color='gray', alpha=0.1)
            plt.title(f'{group} Group ({session.upper()}) - Eyes Open vs Closed', fontsize=14)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'PSD_{group}_{session}.png')
            print(f"  Saved PSD_{group}_{session}.png")
            
        # 2. Plot Between-Group (Young vs Elderly - Eyes Closed)
        # This shows the aging effect best
        y_stats = get_group_psd(all_data['Young'][session]['clean_epochs'])
        e_stats = get_group_psd(all_data['Elderly'][session]['clean_epochs'])
        
        if y_stats and e_stats:
            f = y_stats['f']
            idx = np.where((f >= 2) & (f <= 40))[0]
            
            plt.figure(figsize=(10, 6))
            
            # Young Closed
            plt.fill_between(f[idx], y_stats['closed_25'][idx], y_stats['closed_75'][idx], color='blue', alpha=0.15)
            plt.plot(f[idx], y_stats['closed_med'][idx], label='Young (Closed)', color='blue', linewidth=2)
            
            # Elderly Closed
            plt.fill_between(f[idx], e_stats['closed_25'][idx], e_stats['closed_75'][idx], color='orange', alpha=0.15)
            plt.plot(f[idx], e_stats['closed_med'][idx], label='Elderly (Closed)', color='darkorange', linewidth=2, linestyle='--')
            
            plt.axvspan(8, 12, color='gray', alpha=0.1)
            plt.title(f'Young vs Elderly Comparison ({session.upper()}) - Eyes Closed', fontsize=14)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'PSD_Compare_Age_{session}.png')
            print(f"  Saved PSD_Compare_Age_{session}.png")

# =============================================================================
# 5. CLASSIFICATION
# =============================================================================

def EEGNet(nb_classes, Chans=64, Samples=128):
    input1 = layers.Input(shape=(Chans, Samples, 1))
    
    b1 = layers.Conv2D(8, (1, 64), padding='same', use_bias=False)(input1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=constraints.max_norm(1.))(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('elu')(b1)
    b1 = layers.AveragePooling2D((1, 4))(b1)
    b1 = layers.Dropout(0.5)(b1)
    
    b2 = layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(b1)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('elu')(b2)
    b2 = layers.AveragePooling2D((1, 8))(b2)
    b2 = layers.Dropout(0.5)(b2)
    
    flat = layers.Flatten()(b2)
    out = layers.Dense(nb_classes, activation='softmax', kernel_constraint=constraints.max_norm(0.25))(flat)
    return models.Model(inputs=input1, outputs=out)

def train_loso(X, y, groups, title):
    print(f"\n--- {title} LOSO Classification ---")
    logo = LeaveOneGroupOut()
    accs = []
    
    # Check minimum groups
    if len(np.unique(groups)) < 2:
        print("  Not enough subjects for CV.")
        return

    for train, test in logo.split(X, y, groups):
        y_tr, y_te = utils.to_categorical(y[train], 2), utils.to_categorical(y[test], 2)
        
        model = EEGNet(2, N_CHANNELS, X.shape[2])
        model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        
        es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X[train], y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(X[test], y_te), callbacks=[es])
        
        preds = np.argmax(model.predict(X[test], verbose=0), axis=1)
        acc = accuracy_score(y[test], preds)
        accs.append(acc)
        
        # print(f"  Sub {groups[test][0]}: {acc:.4f}")
        tf.keras.backend.clear_session()
        
    print(f"  MEAN ACCURACY: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    return np.mean(accs)

# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    # 1. Load All Data
    data = build_datasets()
    
    # 2. Generate Plots (Young, Elderly, Comparison)
    plot_psds(data)
    
    # 3. Run Classification per Group
    results = []
    for group in GROUPS:
        if group in data:
            for session in ['pre', 'post']:
                if len(data[group][session]['X']) > 0:
                    acc = train_loso(data[group][session]['X'], 
                                   data[group][session]['y'], 
                                   data[group][session]['groups'], 
                                   f"{group} - {session.upper()}")
                    results.append({'Group': group, 'Session': session, 'Accuracy': acc})

    # Summary
    print("\n=== FINAL SUMMARY ===")
    print(pd.DataFrame(results))