import numpy as np

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def L1(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def L2(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def mbe(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def SVMLoss(y, y_pred):
    return np.mean(np.maximum(0, y_pred - y + 1))

def CrossEntropy(y, y_pred):
    return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

def CategoicalCrossEntropy(y, y_pred):
    return np.mean(-np.sum(y * np.log(y_pred), axis=1))