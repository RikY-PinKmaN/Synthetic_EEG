"""
Configuration file containing all hyperparameters
"""
# Signal processing parameters
LOWCUT = 8
HIGHCUT = 35
SAMPLING_RATE = 250
FILTER_RIPPLE = 0.5
FILTER_ATTENUATION = 30

# Data splitting parameters
TRAIN_TRIALS = 20
VAL_TRIALS = 10

# CSP+LDA parameters
CSP_COMPONENTS = [3]  # Multiple CSP component settings to try
CSP_EVAL_COMPONENTS = [3]  # Multiple evaluation component settings to try
CSP_REGULARIZATION = [None, 0.1, 0.3]  # Added regularization values to try
CSP_LOG = True
CSP_NORM_TRACE = False

# Model architecture parameters
NOISE_DIM = 100
NUM_SAMPLES= 50
GENERATOR_INIT_CHANNELS = 64
CRITIC_INIT_FILTERS = 64

# GAN training parameters
EPOCHS = 3000
BATCH_SIZE = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10  # Gradient penalty weight
LAMBDA_CLS = 1.5  # Classification loss weight
LEARNING_RATE = 1e-4
BETA_1 = 0.5
BETA_2 = 0.9

# Early stopping parameters
PATIENCE = 500

# Training iteration parameters
NUM_TRAINING_ITERATIONS = 1  # Reduced to 1 as requested
NUM_SYNTHETIC_BATCHES_PER_TRAINING = 20
MIX_RATIOS = [25, 50]  # Percentage of synthetic data to mix

# Visualization parameters
CHANNEL_INDEX_L = 11  # Channel for left hand visualization
CHANNEL_INDEX_R = 8   # Channel for right hand visualization
