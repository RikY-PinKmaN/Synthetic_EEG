import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import src.config as cfg

class CSPLDAClassifier:
    def __init__(self, n_components=cfg.CSP_COMPONENTS[0], reg=None):
        self.csp = CSP(n_components=n_components, 
                       reg=reg, 
                       log=cfg.CSP_LOG, 
                       norm_trace=cfg.CSP_NORM_TRACE)
        self.lda = LinearDiscriminantAnalysis()
        self.fitted = False  # Flag to check if the model is fitted

    def fit(self, X, y):
        # Convert labels to 1/2 format if they're in 0/1 format
        if np.min(y) == 0:
            y = y + 1  # Ensure labels are 1 or 2

        # Convert to float64 for numerical stability
        X = X.astype(np.float64)

        # Fit CSP
        self.csp.fit(X, y.ravel())

        # Transform data and fit LDA
        X_csp = self.csp.transform(X)
        self.lda.fit(X_csp, y.ravel())

        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X = X.astype(np.float64)  # Ensure float64
        X_csp = self.csp.transform(X)
        return self.lda.predict(X_csp)

    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X = X.astype(np.float64)
        X_csp = self.csp.transform(X)
        return self.lda.predict_proba(X_csp)

    def evaluate(self, eval_data, eval_labels):
        """
        Evaluate the classifier on evaluation data.
        
        Parameters:
        -----------
        eval_data : array
            Evaluation data (shape: (n_trials, n_channels, n_time_points), e.g., (n, 22, 750))
        eval_labels : array
            Evaluation labels
        
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        if not self.fitted:
            raise ValueError("Classifier must be fitted before evaluation")
        
        eval_data_64 = eval_data.astype(np.float64)
        predictions = self.predict(eval_data_64)
        accuracy = accuracy_score(eval_labels, predictions)
        
        return accuracy  # You can extend this to return more metrics if needed

# The rest of the functions like generate_synthetic_data remain unchanged
def generate_synthetic_data(left_generator, right_generator, num_samples=cfg.NUM_SAMPLES):
    # (Unchanged from previous version)
    if num_samples % 2 != 0:
        num_samples += 1

    synthetic_left_hand_data = left_generator(tf.random.normal([num_samples//2, cfg.NOISE_DIM]), training=False)
    synthetic_right_hand_data = right_generator(tf.random.normal([num_samples//2, cfg.NOISE_DIM]), training=False)

    synthetic_data = np.concatenate([synthetic_left_hand_data.numpy(), synthetic_right_hand_data.numpy()], axis=0)
    synthetic_labels = np.concatenate([
        np.ones(synthetic_left_hand_data.shape[0]),
        np.ones(synthetic_right_hand_data.shape[0]) * 2
    ], axis=0)

    return synthetic_data, synthetic_labels
