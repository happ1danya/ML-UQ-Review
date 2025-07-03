# data_io.py
import numpy as np
from tensorflow.keras.models import load_model

def load_data(model_path, X_path, y_path):
    model = load_model(model_path)
    X = np.load(X_path)
    y = np.load(y_path)
    return model, X, y
