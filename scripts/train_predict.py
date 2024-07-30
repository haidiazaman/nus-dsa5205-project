import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_it

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression


def train_val_test_split(X, y, train_size, val_size, test_size):
    """
    split X,y according to sizes
    """
    train_size, val_size, test_size = int(len(X) * train_size), int(len(X) * val_size), int(len(X) * test_size)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    return X_train,y_train,X_val,y_val,X_test,y_test

{'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (8, 64), 'learning_rate': 'constant', 'solver': 'adam'}