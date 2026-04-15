# =============================================================================
#
#               P300 ANALYSIS PIPELINE (Python Conversion)
#
# This script is a direct conversion of the provided MATLAB pipeline.
# It includes the main analysis script, the P300 classification function,
# and the shaded error bar plotting utility.
#
# =============================================================================

import os
import numpy as np
import mne
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import mode
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# Suppress runtime warnings from MNE about reading non-standard GDF files
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# PART 1: shadedErrorBar Function (shaded_error_bar.py)
# =============================================================================

def shaded_error_bar(x, y, err_bar, line_props=None, transparent=False, ax=None):
    """
    Python equivalent of the MATLAB shadedErrorBar function.
    Makes a 2-d line plot with a shaded error bar.

    Inputs:
    - x: vector of x values.
    - y: vector of y values or a matrix where each row is an observation.
    - err_bar: if a vector, symmetric error bars. If a 2-row matrix, asymmetric.
               Can also be a tuple of two functions (e.g., (np.mean, np.std)).
    - line_props: Dictionary of properties for the main line (e.g., {'color':'k', 'linestyle':'-'}).
    - transparent: If True, make the patch transparent.
    - ax: Matplotlib axes object to plot on. If None, a new one is created.

    Outputs:
    - H: A dictionary of handles to the plot objects.
    """
    # If no axes are specified, get the current axes.
    if ax is None:
        ax = plt.gca()

    # If y is a matrix of observations and err_bar is a function tuple,
    # calculate the main line (e.g., mean) and the error (e.g., std dev).
    if isinstance(err_bar, (list, tuple)) and callable(err_bar[0]):
        func_line = err_bar[0]
        func_err = err_bar[1]
        err_bar_val = func_err(y, axis=0)
        y = func_line(y, axis=0)
    else:
        # Otherwise, y is the main line and err_bar contains the error values.
        y = np.ravel(y)
        err_bar_val = err_bar

    x = np.ravel(x)

    # If a single vector of errors is provided, make it symmetric by duplicating it.
    if len(err_bar_val.shape) == 1:
        err_bar_val = np.vstack([err_bar_val, err_bar_val])
    
    # Set default line properties if none are provided.
    if line_props is None:
        line_props = {'color': 'k', 'linestyle': '-'}

    # Plot the main line to establish its properties, especially color.
    main_line, = ax.plot(x, y, **line_props)
    
    # Get the color of the main line for the shaded patch.
    col = main_line.get_color()
    
    # Define the patch color and transparency.
    if transparent:
        face_alpha = 0.15 # Use a light alpha for a see-through effect
        patch_color = col
    else:
        face_alpha = 1.0 # Opaque patch
        # Desaturate the color for the patch to make it lighter than the main line.
        patch_color = tuple(c + (1 - c) * 0.85 for c in plt.cm.colors.to_rgb(col))

    # Calculate the upper and lower bounds of the error bar.
    uE = y + err_bar_val[0, :]
    lE = y - err_bar_val[1, :]

    # Create the shaded region (patch) between the lower and upper bounds.
    patch = ax.fill_between(x, lE, uE, color=patch_color, alpha=face_alpha, edgecolor='none')

    # Create edge lines for the error bar bounds.
    edge_color = tuple(c + (1 - c) * 0.45 for c in plt.cm.colors.to_rgb(col))
    edge1, = ax.plot(x, lE, '-', color=edge_color)
    edge2, = ax.plot(x, uE, '-', color=edge_color)

    # Re-plot the main line on top of the patch and edges.
    main_line.remove()
    main_line, = ax.plot(x, y, **line_props)

    # Return handles to the plotted objects.
    H = {'mainLine': main_line, 'patch': patch, 'edge': (edge1, edge2)}
    return H


# =============================================================================
# PART 2: P300 Classification Function (p300_classification_stepwise.py)
# =============================================================================

def p300_classification(p300_data):
    """
    Python equivalent of the MATLAB P300_classification function.
    Performs K-fold cross-validation with stepwise LDA for feature selection.
    """
    # --- Step 1: Load and Reshape Data ---
    # Load p300 feature matrix and corresponding labels from the input dictionary.
    p300_feature = p300_data['x']
    p300_labels = p300_data['y'].ravel() # Ensure labels are a 1D array

    # Re-construct the feature matrix to be 2D (trials x features).
    # Original shape is (channels, samples, trials).
    # We want to flatten channels and samples into a single feature dimension for each trial.
    # The transpose and reshape order matches MATLAB's column-major reshaping.
    num_channels, num_samples, num_trials = p300_feature.shape
    # Reshape to (num_trials, num_channels * num_samples)
    p300_feature = p300_feature.transpose(2, 1, 0).reshape(num_trials, -1)
    print(f"Classifier Input: Reshaped feature matrix to {p300_feature.shape}")

    # --- Step 2: Cross-Validation Setup ---
    # Set the number of folds for cross-validation.
    num_folds = 9
    # Initialize a variable for storing the accuracy of each fold.
    accuracies = np.zeros(num_folds)

    # Create a KFold object. `shuffle=False` ensures the data is split sequentially,
    # mimicking MATLAB's `crossvalind` default behavior.
    kf = KFold(n_splits=num_folds, shuffle=False)

    # --- Step 3: Perform Cross-Validation ---
    # Loop through each fold.
    for fold, (train_indices, test_indices) in enumerate(kf.split(p300_feature)):
        print(f'Processing fold {fold + 1}/{num_folds}...')
        
        # Split data into training and testing sets for the current fold.
        train_data = p300_feature[train_indices, :]
        train_labels = p300_labels[train_indices]
        test_data = p300_feature[test_indices, :]
        test_labels = p300_labels[test_indices]
        print(f"  - Fold {fold+1} Split: {len(train_indices)} train samples, {len(test_indices)} test samples")
        
        # --- Step 4: Stepwise Feature Selection ---
        # This mimics MATLAB's 'stepwisefit' by selecting the 5 best features.
        # It uses a forward selection approach with an LDA classifier as the model.
        # `cv=0` means the selection is performed on the entire training set of this fold.
        lda_sfs = LinearDiscriminantAnalysis()
        sfs = SequentialFeatureSelector(
            lda_sfs,
            k_features=5,          # Select the top 5 features.
            forward=True,          # Use forward selection (add one feature at a time).
            floating=False,
            scoring='accuracy',    # Use accuracy as the selection criterion.
            cv=0,                  # Perform selection on the whole training set.
            n_jobs=-1              # Use all available CPU cores.
        )
        sfs.fit(train_data, train_labels)

        # Get the indices of the selected features.
        select_indices = list(sfs.k_feature_idx_)
        print(f"  - Fold {fold+1} Selected Features (indices): {select_indices}")
        
        # Filter the train and test data to include only the selected features.
        train_data_selected = train_data[:, select_indices]
        test_data_selected = test_data[:, select_indices]

        # --- Step 5: Classification ---
        # Initialize and train an LDA model on the selected features from the training data.
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(train_data_selected, train_labels)

        # Predict the class labels for the test data.
        predicted_labels = lda_model.predict(test_data_selected)

        # --- Step 6: Evaluate Accuracy ---
        # Calculate the accuracy for the current fold by comparing predicted and true labels.
        accuracy = np.sum(predicted_labels == test_labels) / len(test_labels)
        accuracies[fold] = accuracy
        print(f'  - Accuracy for fold {fold + 1}: {accuracy * 100:.2f}%')

    # --- Step 7: Final Result ---
    # Calculate the average accuracy across all folds.
    average_accuracy = np.mean(accuracies)
    print(f'Average accuracy over {num_folds} folds: {average_accuracy * 100:.2f}%')
    
    return average_accuracy


# =============================================================================
# PART 3: Main Analysis Script
# =============================================================================

def main_analysis():
    # --- Configuration ---
    # Set the path to the directory containing the GDF data files.
    # NOTE: Update this path to your data directory
    path = 'H3' # Assumes data is in the same folder as the script
    
    # --- Initialize storage for P300/N200 metrics ---
    num_subject_pairs = len(range(0, 7, 2)) # 4 pairs: (0,1), (2,3), (4,5), (6,7)
    P300_am = np.zeros(num_subject_pairs)
    P300_lat = np.zeros(num_subject_pairs)
    N200_am = np.zeros(num_subject_pairs)
    N200_lat = np.zeros(num_subject_pairs)
    
    # Define session IDs for loading files.
    sessionID = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08']
    
    # Define the target sampling rate (in Hz) for analysis after resampling.
    fs = 80
    
    # Define the channel to be plotted (Cz). In 0-indexed Python, this is channel 1.
    ch_to_plot = [1]
    
    # Define the artifact threshold in microvolts (uV). Epochs exceeding this will be rejected.
    artifth = 50
    
    # --- Timing Definitions ---
    # Define the time window for ERP epochs (from -100ms to +600ms relative to stimulus).
    ts_start_sec, ts_end_sec = -0.1, 0.6
    # Convert time window to sample points at the target sampling rate.
    ts = np.arange(round(ts_start_sec * fs), round(ts_end_sec * fs) + 1)
    
    # Define the time window for classification features (from +150ms to +500ms).
    ts_f_start_sec, ts_f_end_sec = 0.15, 0.5
    ts_f = np.arange(round(ts_f_start_sec * fs), round(ts_f_end_sec * fs))
    
    # Define the baseline interval in samples (from -100ms to 0ms relative to stimulus).
    BLint = np.array([-round(0.1 * fs), 0])
    
    # --- Channel Location Setup ---
    # Define channel locations for an 8-channel Standard-10-20 Cap.
    # This is used to create an MNE montage for plotting topographies (if needed) and correct channel naming.
    chan_loc_data = """
    1	       0	 0.25556	      Fz
    2	      90	       0	      Cz
    3	     180	 0.25556	      Pz
    4	     -90	 0.25556	      C3
    5	      90	 0.25556	      C4
    6	    -141	 0.33333	      P3
    7	     141	 0.33333	      P4
    8	     180	 0.51111	      Oz
    """
    chan_loc_lines = chan_loc_data.strip().split('\n')
    ch_names = []
    ch_pos = {}
    for line in chan_loc_lines:
        parts = line.split()
        label = parts[3]
        theta_deg = float(parts[1])
        radius = float(parts[2])
        # Convert polar coordinates to Cartesian for MNE.
        theta_rad = np.deg2rad(theta_deg)
        x = radius * np.sin(theta_rad)
        y = radius * np.cos(theta_rad)
        z = 0 # Assume 2D layout
        ch_pos[label] = (x, y, z)
        ch_names.append(label)
    # Create the MNE montage object.
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos)

    # ==================== MAIN SUBJECT PROCESSING LOOP ====================
    # Loop through subject pairs. s will be 0, 2, 4, 6.
    for s_idx, s in enumerate(range(0, 7, 2)):
        # --- 1. DATA LOADING AND CONCATENATION ---
        print(f"\n{'='*25} Processing subject pair: {sessionID[s]} and {sessionID[s+1]} {'='*25}")
        
        # Create a new figure for each subject pair's ERP plot.
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define the channels to be processed (all 8 channels).
        OurChans = np.arange(8)
        nchan = 8
        
        # Define paths to the two GDF files for the current pair.
        subjID1 = os.path.join(path, sessionID[s])
        subjID2 = os.path.join(path, sessionID[s+1])
        
        # Load the two GDF files into MNE Raw objects.
        raw1 = mne.io.read_raw_gdf(f"{subjID1}.gdf", preload=True, eog=None)
        raw2 = mne.io.read_raw_gdf(f"{subjID2}.gdf", preload=True, eog=None)
        
        # Rename channels from default ('EEG 1', etc.) to standard names (Fz, Cz, etc.).
        raw1.rename_channels({old: new for old, new in zip(raw1.ch_names, ch_names)})
        raw2.rename_channels({old: new for old, new in zip(raw2.ch_names, ch_names)})
        
        # Apply the channel location montage to both raw objects.
        raw1.set_montage(montage)
        raw2.set_montage(montage)
        
        # Concatenate the two raw objects into a single continuous recording.
        raw = mne.concatenate_raws([raw1, raw2])
        print(f"Step 1: Loaded and concatenated data. Total duration: {raw.times[-1]:.2f}s")
        original_fs = raw.info['sfreq']
        
        # --- 2. PRE-PROCESSING (FILTERING AND RESAMPLING) ---
        print(f"Step 2: Pre-processing data (Original Fs: {original_fs} Hz)...")
        # Extract data into a NumPy array for manual filtering.
        data = raw.get_data()
        
        # Design filters based on the *original* sampling frequency.
        b_hp, a_hp = butter(4, 0.4 / (original_fs / 2), btype='high') # High-pass at 0.4 Hz
        b_lp, a_lp = butter(4, 30 / (original_fs / 2), btype='low')   # Low-pass at 30 Hz
        # Notch filters for specific stimulation frequencies (SSVEP artifact removal).
        b_notch1, a_notch1 = iirnotch(4.3, 5, original_fs)
        b_notch2, a_notch2 = iirnotch(4.3 * 2, 5, original_fs)
        b_notch3, a_notch3 = iirnotch(5.8, 5, original_fs)
        b_notch4, a_notch4 = iirnotch(5.8 * 2, 5, original_fs)

        # Apply filters channel by channel.
        for j in OurChans:
            data[j, :] = filtfilt(b_hp, a_hp, data[j, :])
            data[j, :] = filtfilt(b_lp, a_lp, data[j, :])
            
            # Apply notch filters only for specific subjects as in the original script.
            if s == 0 or s == 4: # H01/H02, H05/H06
                data[j, :] = filtfilt(b_notch1, a_notch1, data[j, :])
                data[j, :] = filtfilt(b_notch2, a_notch2, data[j, :])
            elif s == 2 or s == 6: # H03/H04, H07/H08
                data[j, :] = filtfilt(b_notch3, a_notch3, data[j, :])
                data[j, :] = filtfilt(b_notch4, a_notch4, data[j, :])
        
        # Put the filtered data back into the MNE Raw object.
        raw._data = data
        
        # Resample the data to the target frequency (fs = 80 Hz).
        raw.resample(fs)
        print(f"   ...Filtering and resampling complete. New Fs: {raw.info['sfreq']} Hz")
        
        # --- 3. EVENT EXTRACTION AND EPOCHING ---
        print("Step 3: Extracting events and epoching data...")
        # Get events and their corresponding IDs from the raw object's annotations.
        events, event_id_dict = mne.events_from_annotations(raw)
        id_to_desc = {v: k for k, v in event_id_dict.items()} # Map ID back to description string
        
        # Extract event latencies (in samples) and event codes (as integers).
        latencies = events[:, 0]
        event_codes = np.array([int(id_to_desc[code]) for code in events[:, 2]])
        
        # Create a simplified stimulus array: 1 for Non-Target, 2 for Target, 0 otherwise.
        tot_sti = np.zeros_like(event_codes)
        tot_sti[event_codes == 33286] = 1 # Non-target flash
        tot_sti[event_codes == 33285] = 2 # Target flash
        
        # Find the indices of all flashes (both target and non-target).
        flash_indices = np.where(tot_sti != 0)[0]
        print(f"   - Found {np.sum(tot_sti==2)} potential Target and {np.sum(tot_sti==1)} potential Non-Target events.")

        # Re-reference the data to the average of all channels.
        # raw.set_eeg_reference('average', projection=False)
        # Get the re-referenced data and convert from Volts to microvolts (uV).
        EEG = raw.get_data() * 1e6
        
        # --- 4. ARTIFACT REJECTION AND ERP AVERAGING ---
        print("Step 4: Performing artifact rejection and creating clean ERP epochs...")
        # Initialize lists to store clean epochs.
        temp_erp_target = []
        temp_erp_nontarget = []
        
        # Loop through all detected flash events.
        for i in range(len(flash_indices)):
            idx = flash_indices[i]
            
            # The actual stimulus event is one sample after the trigger.
            if idx + 1 >= len(latencies): continue # Skip if it's the last event
            
            latency = latencies[idx + 1]
            
            # Define the epoch window in samples relative to the stimulus onset.
            epoch_indices = latency + ts
            if np.min(epoch_indices) < 0 or np.max(epoch_indices) >= EEG.shape[1]: continue # Ensure epoch is within data bounds

            # Extract the epoch data for all channels.
            ep = EEG[np.ix_(OurChans, epoch_indices)]
            
            # --- Baseline Correction ---
            # Define the baseline window in samples.
            bl_start, bl_end = latencies[idx] + BLint[0], latencies[idx] + BLint[1]
            if bl_start < 0 or bl_end >= EEG.shape[1]: continue # Ensure baseline is within bounds
            
            # Calculate the mean amplitude during the baseline period for each channel.
            BLamp = np.mean(EEG[:, bl_start:bl_end], axis=1, keepdims=True)
            # Subtract the baseline from the epoch.
            ep = ep - BLamp
            
            # --- Artifact Rejection ---
            # Calculate the peak-to-peak amplitude for each channel in the epoch.
            peak_to_peak = np.max(ep, axis=1) - np.min(ep, axis=1)
            # If any channel exceeds the threshold, reject the entire epoch.
            if np.any(peak_to_peak > artifth):
                continue
            
            # If the epoch is clean, add it to the appropriate list.
            if tot_sti[idx] == 2: # Target
                temp_erp_target.append(ep)
            elif tot_sti[idx] == 1: # Non-Target
                temp_erp_nontarget.append(ep)

        # Convert the lists of epochs into 3D NumPy arrays (channels x samples x trials).
        # Handle cases where no clean epochs were found.
        temp_erp_target = np.array(temp_erp_target).transpose(1, 2, 0) if temp_erp_target else np.empty((nchan, len(ts), 0))
        temp_erp_nontarget = np.array(temp_erp_nontarget).transpose(1, 2, 0) if temp_erp_nontarget else np.empty((nchan, len(ts), 0))

        # --- Report on accepted/rejected trials ---
        num_target_accepted = temp_erp_target.shape[2]
        num_nontarget_accepted = temp_erp_nontarget.shape[2]
        print(f"   - After artifact rejection ({artifth}uV threshold):")
        print(f"     - Accepted Target epochs: {num_target_accepted}. Shape: {temp_erp_target.shape}")
        print(f"     - Accepted Non-Target epochs: {num_nontarget_accepted}. Shape: {temp_erp_nontarget.shape}")

        
        # --- 5. FEATURE EXTRACTION FOR CLASSIFICATION ---
        print("Step 5: Extracting features for the LDA classifier...")
        # This section reconstructs the experiment logic to create averaged feature vectors.
        # It processes the two original sessions (a1, a2) separately.
        
        # Extract event codes for each original session.
        events1, id_dict1 = mne.events_from_annotations(raw1)
        id_to_desc1 = {v: k for k, v in id_dict1.items()}
        a1 = np.array([int(id_to_desc1[code]) for code in events1[:, 2]])

        events2, id_dict2 = mne.events_from_annotations(raw2)
        id_to_desc2 = {v: k for k, v in id_dict2.items()}
        a2 = np.array([int(id_to_desc2[code]) for code in events2[:, 2]])

        erp_data = {'a1': {}, 'a2': {}} # To store single-trial epochs for each flash
        m_erp_data = {'a1': {}, 'a2': {}} # To store averaged epochs (the features)

        # Define row/column flash codes and number of letters spelled, which differs by session.
        if s in [0, 2]: # H01-H04
            row_codes = [33025, 33026, 33027]
            col_codes = [33028, 33029, 33030]
            num_letters_a1, num_letters_a2 = 5, 4
        else: # H05-H08
            row_codes = [33025, 33026, 33027, 33028, 33029]
            col_codes = [33030, 33031, 33032, 33033, 33034]
            num_letters_a1, num_letters_a2 = 5, 4
        all_rc_codes = row_codes + col_codes
        
        # Loop over the two sessions (a1 and a2).
        for prefix, event_array, num_letters in [('a1', a1, num_letters_a1), ('a2', a2, num_letters_a2)]:
            # Find the indices of all row/column flash events.
            rc_indices = {code: np.where(event_array == code)[0] for code in all_rc_codes}
            
            # Loop through each letter that was spelled.
            for letter_idx in range(num_letters):
                erp_data[prefix][letter_idx] = {}
                m_erp_data[prefix][letter_idx] = {}
                
                # Each letter is defined by 15 flashes of each row/column.
                for rc_code in all_rc_codes:
                    start = 15 * letter_idx
                    end = start + 15
                    if end > len(rc_indices[rc_code]): continue
                    letter_flashes_indices = rc_indices[rc_code][start:end]
                    
                    # Extract the 15 epochs for this row/col flash for this letter.
                    epochs = []
                    for flash_idx in letter_flashes_indices:
                        # Find the global index in the concatenated data.
                        event_offset = 0 if prefix == 'a1' else len(a1)
                        global_idx = flash_idx + event_offset
                        
                        if global_idx + 1 >= len(latencies): continue
                        latency = latencies[global_idx + 1]
                        
                        # Extract epoch using the classification time window `ts_f`.
                        epoch_indices = latency + ts_f
                        if np.min(epoch_indices) < 0 or np.max(epoch_indices) >= EEG.shape[1]: continue

                        ep = EEG[np.ix_(OurChans, epoch_indices)]
                        
                        # Perform baseline correction.
                        bl_start, bl_end = latencies[global_idx] + BLint[0], latencies[global_idx] + BLint[1]
                        if bl_start < 0 or bl_end >= EEG.shape[1]: continue
                        BLamp = np.mean(EEG[:, bl_start:bl_end], axis=1, keepdims=True)
                        ep = ep - BLamp
                        epochs.append(ep)
                        
                    # If epochs were successfully extracted, average them to create one feature vector.
                    if epochs:
                        erp_data[prefix][letter_idx][rc_code] = np.array(epochs).transpose(1, 2, 0)
                        # The feature is the mean ERP for those 15 flashes.
                        m_erp_data[prefix][letter_idx][rc_code] = np.mean(erp_data[prefix][letter_idx][rc_code], axis=2)

        # --- Assemble Features and Labels for the Classifier ---
        feature_list = []
        label_list = []
        
        # Loop through the extracted averaged ERPs (`m_erp_data`) and assign labels.
        for prefix, event_array, num_letters in [('a1', a1, num_letters_a1), ('a2', a2, num_letters_a2)]:
            for letter_idx in range(num_letters):
                for rc_code_list in [row_codes, col_codes]:
                    for rc_code in rc_code_list:
                        # Check if a feature was successfully created.
                        if m_erp_data[prefix][letter_idx].get(rc_code) is not None:
                            feature_list.append(m_erp_data[prefix][letter_idx][rc_code])
                            # The label (Target/Non-Target) is determined by the event *preceding* the flash sequence.
                            start, end = 15 * letter_idx, 15 * (letter_idx + 1)
                            flash_indices_local = np.where(event_array == rc_code)[0]
                            letter_flash_indices = flash_indices_local[start:end]
                            if len(letter_flash_indices) > 0:
                                preceding_codes = event_array[letter_flash_indices - 1]
                                # The label is the mode of the preceding event codes (should all be the same).
                                label = mode(preceding_codes, keepdims=False)[0]
                                label_list.append(label)

        # If no features were extracted, skip the rest of the analysis for this pair.
        if not feature_list:
            print("No features could be extracted. Skipping classification and plotting for this pair.")
            continue

        # Convert feature and label lists into NumPy arrays.
        p300_feature = np.array(feature_list).transpose(1, 2, 0)
        p300_labels = np.array(label_list)

        # Convert MNE event codes to simple 1 (Target) / 2 (Non-Target) labels.
        # NOTE: This is opposite to the epoching step labels. Here, Target=1, Non-Target=2.
        p300_labels[p300_labels == 33286] = 2 # Non-target
        p300_labels[p300_labels == 33285] = 1 # Target
        
        # --- Report on classifier features ---
        num_class_target = np.sum(p300_labels == 1)
        num_class_nontarget = np.sum(p300_labels == 2)
        print(f"   - Assembled data for classifier:")
        print(f"     - Total feature vectors: {p300_feature.shape[2]}")
        print(f"     - Target features: {num_class_target}, Non-Target features: {num_class_nontarget}")
        print(f"     - Feature matrix shape: {p300_feature.shape} (Channels x Samples x Trials)")

        # --- 6. P300 CLASSIFICATION ---
        p300_data = {'x': p300_feature, 'y': p300_labels}
        print(f"\n--- Step 6: Starting P300 Classification for trial {s_idx + 1} ---")
        p300_accuracy = p300_classification(p300_data)

        # --- 7. ERP PLOTTING & PEAK ANALYSIS ---
        print(f"\n--- Step 7: Calculating and plotting grand average ERPs ---")
        # Design a low-pass filter for plotting to make the ERPs smoother.
        b_plot, a_plot = butter(4, 16 / fs * 2, btype='low')

        # --- Target ERP ---
        if temp_erp_target.shape[2] > 0:
            # Calculate the grand average ERP by averaging across the trials dimension (axis=2).
            cpp_target = np.mean(temp_erp_target[ch_to_plot, :, :], axis=2).squeeze()
            
            # --- P300 and N200 Peak Analysis ---
            # Define analysis windows in sample points, relative to the `ts` vector.
            p300_start_idx = np.where(ts == round(0.2 * fs))[0][0] # 200ms
            p300_end_idx = np.where(ts == round(0.4 * fs))[0][0]   # 400ms
            n200_end_idx = np.where(ts == round(0.3 * fs))[0][0]   # 300ms
            zero_time_idx = np.where(ts == 0)[0][0]

            # P300 analysis: find the max peak between 200-400ms.
            p300_window = cpp_target[p300_start_idx:p300_end_idx]
            if p300_window.size > 0:
                p_rel = np.argmax(p300_window) # Peak index relative to window start
                p_abs = p300_start_idx + p_rel # Peak index relative to epoch start
                
                # Average amplitude in a 50ms window around the peak.
                peak_window_half = round(0.025 * fs)
                peak_ran = np.arange(p_abs - peak_window_half, p_abs + peak_window_half + 1)
                peak_ran = np.clip(peak_ran, 0, len(cpp_target) - 1)
                p300_amp = np.mean(cpp_target[peak_ran])
                p300_lat_sec = (p_abs - zero_time_idx) / fs # Latency in seconds
                
                # Store results.
                P300_am[s_idx] = p300_amp
                P300_lat[s_idx] = p300_lat_sec
                print(f"P300 Peak Analysis -> Amplitude: {p300_amp:.2f} uV, Latency: {p300_lat_sec:.3f} s")

            # N200 analysis: find the min peak between 200-300ms.
            n200_window = cpp_target[p300_start_idx:n200_end_idx]
            if n200_window.size > 0:
                p2_rel = np.argmin(n200_window) # Peak index relative to window start
                p2_abs = p300_start_idx + p2_rel # Peak index relative to epoch start
                
                n200_amp = cpp_target[p2_abs] # Amplitude is the value at the peak
                n200_lat_sec = (p2_abs - zero_time_idx) / fs # Latency in seconds
                
                # Store results.
                N200_am[s_idx] = n200_amp
                N200_lat[s_idx] = n200_lat_sec
                print(f"N200 Peak Analysis -> Amplitude: {n200_amp:.2f} uV, Latency: {n200_lat_sec:.3f} s")
            
            # --- Plotting Target ERP ---
            # Calculate the Standard Error of the Mean (SEM) for the error bars.
            squeezed_target_data = temp_erp_target[ch_to_plot, :, :].squeeze()
            cpp_error_target = np.std(squeezed_target_data, axis=1, ddof=1) / np.sqrt(temp_erp_target.shape[2])
            
            # Filter the averaged ERP for plotting.
            cpp_target_filt = filtfilt(b_plot, a_plot, cpp_target)
            # Plot the line with a shaded error bar.
            shaded_error_bar(
                x=ts / fs, y=cpp_target_filt, err_bar=cpp_error_target,
                line_props={'color': 'b', 'linewidth': 2.5, 'label': f'Target ({num_target_accepted} trials)'},
                transparent=True, ax=ax
            )
        
        # --- Non-Target ERP ---
        if temp_erp_nontarget.shape[2] > 0:
            # Calculate the grand average ERP.
            cpp_nontarget = np.mean(temp_erp_nontarget[ch_to_plot, :, :], axis=2).squeeze()

            # Calculate the SEM for the error bars.
            squeezed_nontarget_data = temp_erp_nontarget[ch_to_plot, :, :].squeeze()
            cpp_error_nontarget = np.std(squeezed_nontarget_data, axis=1, ddof=1) / np.sqrt(temp_erp_nontarget.shape[2])

            # Filter the averaged ERP for plotting.
            cpp_nontarget_filt = filtfilt(b_plot, a_plot, cpp_nontarget)
            # Plot the line with a shaded error bar.
            shaded_error_bar(
                x=ts / fs, y=cpp_nontarget_filt, err_bar=cpp_error_nontarget,
                line_props={'color': 'r', 'linewidth': 2.5, 'label': f'Non-Target ({num_nontarget_accepted} trials)'},
                transparent=True, ax=ax
            )
        
        # --- Finalize and Show Plot ---
        ax.set_xlim([-0.1, 0.6])
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--') # Add horizontal line at 0
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--') # Add vertical line at stimulus onset
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Amplitude (μV)', fontsize=16)
        ax.set_title(f'ERP for Subject Pair {s+1}-{s+2} at Channel {ch_names[ch_to_plot[0]]}')
        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    # --- DUMMY FILE CREATION (for demonstration) ---
    # In a real scenario, you would have your actual GDF data files in the specified path.
    # This block creates empty placeholder files if they don't exist, so the script can be run
    # without immediately failing due to missing files. The `mne.io.read_raw_gdf` call will
    # still fail if the files are not valid GDFs.
    if not os.path.exists(os.path.join('.', 'H01.gdf')):
        print("Creating dummy GDF files for demonstration...")
        if not os.path.exists('.'):
            os.makedirs('.')
        for i in range(1, 9):
            # Create empty files as placeholders.
            with open(os.path.join('.', f'H0{i}.gdf'), 'w') as f:
                pass

    try:
        main_analysis()
    except Exception as e:
        print("\n--- SCRIPT EXECUTION FAILED ---")
        print(f"Error: {e}")
        print("\nThis likely happened because the dummy GDF files are empty or your real GDF files are not in the correct path.")
        print("Please place your actual GDF data files in the same directory as this script (or update the 'path' variable).")