import numpy as np

def DAT_normalize(X):
    X = X - np.expand_dims(np.mean(X,1),1)
    X = X / np.expand_dims(np.expand_dims(np.abs(X).max(1).max(1), 1), 1)
    return X
