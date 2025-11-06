% =========================================================================
% SCRIPT FOR P300 CLASSIFICATION USING A SUPPORT VECTOR MACHINE (SVM)
% =========================================================================
% clear; clc; close all;

%% Step 1: Load Your Data
% -------------------------------------------------------------------------
% This script assumes your .mat files contain variables with specific names.
% Please check your files and update the variable names here if needed.
disp('Loading data...');

% Load features (the P300 data)
% loaded_data = load('P300_feature.mat');
Data = p300_feature;

% Load labels
% loaded_labels = load('P300_label.mat');
labels = p300_labels;

disp('Data and labels loaded successfully.');

%% Step 2: Inspect and Preprocess the Data
% -------------------------------------------------------------------------
% Get dimensions from the data. Shape should be [Channels x Timepoints x Trials]
[nChannels, nTimepoints, nTrials] = size(Data);

fprintf('Data dimensions:\n');
fprintf('- Channels: %d\n- Timepoints: %d\n- Trials: %d\n', nChannels, nTimepoints, nTrials);

% --- Reshape Data for SVM ---
% Classic machine learning models like SVM require a 2D matrix where:
% Rows = Observations (Trials)
% Columns = Features
% We will flatten the [Channels x Timepoints] for each trial into a single feature vector.
nFeatures = nChannels * nTimepoints;
% Reshape and transpose to get [Trials x Features]
data_reshaped = reshape(Data, nFeatures, nTrials)'; % The transpose (') is crucial

fprintf('Data reshaped for SVM to: [%d Trials x %d Features]\n', nTrials, nFeatures);

% --- Prepare Labels ---
% Per your request, the original numeric labels are used.
% Label '1' is for Target trials (P300).
% Label '2' is for Non-Target trials (Non-P300).

%% Step 3: Split Data into Training and Testing Sets
% -------------------------------------------------------------------------
% Per your request, we will randomly select 30 trials for training and
% use the remaining trials for testing.
disp('Splitting data into 30 training trials and a testing set...');

% Generate a random permutation of all trial indices
rand_indices = randperm(nTrials);

% Assign the first 30 random indices to the training set
idxTrain = rand_indices(1:30);
% Assign the rest of the indices to the testing set
idxTest = rand_indices(31:end);

% Create the training and testing sets
XTrain = data_reshaped(idxTrain, :);
YTrain = labels(idxTrain); % Using original numeric labels

XTest = data_reshaped(idxTest, :);
YTest = labels(idxTest); % Using original numeric labels

fprintf('Training set size: %d trials\n', size(XTrain, 1));
fprintf('Testing set size: %d trials\n', size(XTest, 1));

%% Step 4: Train the SVM Classifier
% -------------------------------------------------------------------------
disp('Training the SVM model...');
tic; % Start timer

% fitcsvm is the function to train a binary SVM
% 'KernelFunction', 'linear': A good, fast starting point for EEG data.
% 'Standardize', true: VERY IMPORTANT. This scales the features to have
%                      zero mean and unit variance. SVMs are sensitive to
%                      feature scaling, so this usually improves performance.
svm_model = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Predict on the test data
testPred = predict(svm_model, XTest);

% Calculate and store the accuracy for this subject
% NOTE: The original accuracy calculation was incorrect and has been fixed.
% We now correctly sum the number of true predictions.
Accuracy = (sum(testPred' == YTest)/ numel(YTest)) * 100;
disp(Accuracy)