% =========================================================================
% SCRIPT TO CALCULATE AND COMPARE CSP PATTERNS FROM REAL AND SYNTHETIC DATA
% =========================================================================
clc;
clear;
close all; % Close any existing figures

% --- 1. Load Data ---
fprintf('Loading data...\n');
load('training_8');             % Load real training data
load('best_synthetic_batch_S8');% Load generated synthetic data
load('Emap.mat');               % Load electrode map for plotting

% --- 2. Prepare Real Training Data Structure ---
data1.x = training_set_x;
data1.y = training_set_y;

% --- 3. Prepare Synthetic Data Structure ---
% Concatenate the generated data for class 0 and class 1
synth_data_combined = cat(1, class0_data_gan_fmt, class1_data_gan_fmt);

% --- CRITICAL FIX: Create correct labels (1 and 2) for the CSP function ---
num_class0_samples = size(class0_data_gan_fmt, 1);
num_class1_samples = size(class1_data_gan_fmt, 1);
labels_for_class1 = ones(num_class0_samples, 1);   % Assign label 1
labels_for_class2 = 2 * ones(num_class1_samples, 1); % Assign label 2
synthetic_labels = cat(1, labels_for_class1, labels_for_class2);

% Prepare the final synthetic data structure
data2.x = permute(synth_data_combined, [3 2 1]);
data2.y = synthetic_labels;

% --- 4. Regularize Synthetic Data (Robustness Fix) ---
% This prevents potential numerical issues in the csp function
fprintf('Cleaning and regularizing synthetic data...\n');
data2.x(isnan(data2.x) | isinf(data2.x)) = 0; % Precautionary cleaning
noise_factor = 1e-6; % A very small number for regularization
noise = noise_factor * std(data2.x(:)) * randn(size(data2.x));
data2.x = data2.x + noise;

% --- 5. Run Common Spatial Patterns (CSP) ---
fprintf('Running CSP on real and synthetic data...\n');

% Calculate CSP on real data
W1 = csp(data1);
selectedw1 = [W1(1,:); W1(end,:)]; % Get first (best for class 1) and last (best for class 2) patterns

% Calculate CSP on synthetic data
W2 = csp(data2);

% --- 6. (Optional but Recommended) Align Signs for Visual Consistency ---
% This checks if the synthetic patterns are inverted compared to the real ones
% and flips them if necessary. This does not affect classification performance
% but makes the plots much easier to compare visually.
if sign(W1(1,1)) ~= sign(W2(1,1))
    fprintf('Flipping synthetic patterns for better visual alignment.\n');
    W2 = -W2;
end
selectedw2 = [W2(1,:); W2(end,:)];

% --- 7. Generate a Single Figure with All Four Plots ---
fprintf('Generating comparison plots...\n');

% Create a new, large figure window
figure('Name', 'CSP Pattern Comparison: Real vs. Synthetic', 'NumberTitle', 'off', 'Position', [100, 100, 1100, 850]);

% Plot 1: Real Data, Right-Hand Imagery
subplot(2, 2, 1);
plotElecPotentialsSingle(Emap, selectedw1(1,:));
title('Real Data: Right-Hand Imagery', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Real Data', 'FontSize', 14, 'FontWeight', 'bold');

% Plot 2: Real Data, Left-Hand Imagery
subplot(2, 2, 2);
plotElecPotentialsSingle(Emap, selectedw1(2,:));
title('Real Data: Left-Hand Imagery', 'FontSize', 12, 'FontWeight', 'bold');

% Plot 3: Synthetic Data, Right-Hand Imagery
subplot(2, 2, 3);
plotElecPotentialsSingle(Emap, selectedw2(1,:));
title('Synthetic Data: Right-Hand Imagery', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Synthetic Data', 'FontSize', 14, 'FontWeight', 'bold');

% Plot 4: Synthetic Data, Left-Hand Imagery
subplot(2, 2, 4);
plotElecPotentialsSingle(Emap, selectedw2(2,:));
title('Synthetic Data: Left-Hand Imagery', 'FontSize', 12, 'FontWeight', 'bold');

% Add a main title for the entire figure
sgtitle('Comparison of CSP Patterns: Real vs. Synthetic Data', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('Done.\n');