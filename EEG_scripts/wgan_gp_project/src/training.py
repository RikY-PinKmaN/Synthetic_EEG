import numpy as np
import tensorflow as tf
import src.config as config

# # Function to evaluate classification performance inside TF graph
# def classification_reward(fake_data, target_class, csp_lda_classifier):
#     # This function handles evaluation of CSP+LDA model inside TF graph
#     def _evaluate_classification(data, target):
#         # Convert tensor to numpy
#         data_np = data.numpy()
#         target_np = target.numpy()[0]  # Get scalar value

#         # Get class probabilities from CSP+LDA
#         pred_probs = csp_lda_classifier.predict_proba(data_np)

#         # For target class 1 (left hand), we want low probability for class 2
#         # For target class 2 (right hand), we want high probability for class 2
#         if target_np == 1:
#             # For left hand, we want high prob of class 1 (low prob of class 2)
#             class_prob = 1.0 - pred_probs[:, 1]
#         else:
#             # For right hand, we want high prob of class 2
#             class_prob = pred_probs[:, 1]

#         # Calculate mean classification performance (higher = better)
#         classification_score = np.mean(class_prob)

#         # Convert to a loss (lower = better)
#         cls_loss = 1.0 - classification_score

#         # Debug info - print occasionally to avoid flooding output
#         if np.random.random() < 0.01:  # 1% chance to print
#             print(f"Target class: {target_np}, Mean probability: {classification_score:.4f}, Loss: {cls_loss:.4f}")

#         return np.array(cls_loss, dtype=np.float32)

#     # Use tf.py_function to wrap our NumPy-based function
#     return tf.py_function(
#         _evaluate_classification,
#         [fake_data, tf.constant([target_class])],
#         Tout=tf.float32
#     )

# Gradient penalty
def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=config.LAMBDA_GP):
    alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        interpolated_predictions = critic(interpolated, training=True)

    gradients = gp_tape.gradient(interpolated_predictions, [interpolated])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return lambda_gp * gradient_penalty
