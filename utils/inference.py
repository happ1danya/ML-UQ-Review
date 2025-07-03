# inference.py
import numpy as np
from tqdm import trange
import keras.utils
import tensorflow as tf

def run_mc_dropout(model, X, num_obs, batch_size=None):
    """
    Returns array shape (num_obs, N, C)
    """
    observations = []
    N = X.shape[0]
    for i in trange(num_obs, desc="MC Dropout Progress"):
        keras.utils.set_random_seed(i)
        if batch_size is None:
            preds = model(X, training=True).numpy()
        else:
            batches = []
            for batch in tf.data.Dataset.from_tensor_slices(X).batch(batch_size):
                batches.append(model(batch, training=True).numpy())
            preds = np.concatenate(batches, axis=0)
        observations.append(preds)
    return np.stack(observations, axis=0)

def compute_truth_flags(y, preds):
    true_labels = y.argmax(axis=1)
    flags = ['T' if p==t else 'F' for p,t in zip(preds, true_labels)]
    return flags