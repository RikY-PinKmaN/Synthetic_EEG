import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import src.config as cfg

def build_generator():
    noise_input = layers.Input(shape=(cfg.NOISE_DIM,))  # Noise vector (latent space)

    # Fully connected layer to project noise into a larger space
    x = layers.Dense(cfg.GENERATOR_INIT_CHANNELS * 63)(noise_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((63, cfg.GENERATOR_INIT_CHANNELS))(x)  # Shape: (63, 64)

    # Upsampling layers with Conv1DTranspose
    x = layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (126, 128)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (252, 64)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(22, kernel_size=7, strides=2, padding="same", activation="tanh")(x)  # Shape: (504, 22)

    # Crop the extra time points to match (22, 500)
    x = layers.Cropping1D(cropping=(2, 2))(x)  # Shape: (500, 22)

    # Permute to match critic's expected input shape
    x = layers.Permute((2, 1))(x)  # Shape: (22, 500)

    return tf.keras.Model(noise_input, x)

def build_critic():
    data_input = layers.Input(shape=(22, 750))  # Updated to handle 750 time points as per your previous request
    x = layers.Permute((2, 1))(data_input)  # Permute to (time_points, channels)

    x = layers.Conv1D(cfg.CRITIC_INIT_FILTERS, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(cfg.CRITIC_INIT_FILTERS*2, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(cfg.CRITIC_INIT_FILTERS*4, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(cfg.CRITIC_INIT_FILTERS*8, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    
    # Adversarial score output
    critic_score = layers.Dense(1)(x)  # For WGAN-GP
    
    # Classification output (2 classes: left hand = 1, right hand = 2)
    class_logits = layers.Dense(2)(x)  # Output logits for 2 classes

    return tf.keras.Model(inputs=data_input, outputs=[critic_score, class_logits])  # Return both
