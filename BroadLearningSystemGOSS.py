# -*- coding: utf-8 -*-
"""
Broad Learning System with Array Compliance
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time

# Helper Functions (Updated for array compliance)
def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    return np.mean(label_1 == predlabel)

def pinv(A, reg):
    """Replaced matrix inverse with array-based implementation"""
    return np.linalg.inv(reg*np.eye(A.shape[1]) + A.T.dot(A)).dot(A.T)

def sparse_bls(A, b):
    """Array-based implementation without matrix operations"""
    lam = 0.001
    iters = 50
    x = np.zeros((A.shape[1], b.shape[1]))
    for _ in range(iters):
        x = shrinkage(A.T.dot(b) + x - A.T.dot(A.dot(x)), lam)
    return x

def shrinkage(a, b):
    return np.maximum(a - b, 0) - np.maximum(-a - b, 0)

# Main BLS Implementation (Array-compliant)
def BLS(train_x, train_y, test_x, test_y, s=0.5, c=1e-4, N1=10, N2=5, N3=50):
    # Convert all inputs to arrays explicitly
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    
    # Training Phase
    train_start = time.time()
    train_x = preprocessing.scale(train_x)
    train_samples = train_x.shape[0]
    train_x_bias = np.hstack([train_x, 0.1*np.ones((train_samples, 1))])
    
    # Feature Mapping with Array conversions
    feature_maps = np.zeros((train_samples, N2*N1))
    betas = []
    scalers = []
    
    for i in range(N2):
        W = 2 * random.randn(train_x_bias.shape[1], N1) - 1
        features = train_x_bias.dot(W)
        beta = sparse_bls(features, train_x_bias)
        betas.append(beta)
        
        mapped_features = train_x_bias.dot(beta)
        scaler = preprocessing.MinMaxScaler().fit(mapped_features)
        scalers.append(scaler)
        feature_maps[:, i*N1:(i+1)*N1] = scaler.transform(mapped_features)
    
    # Enhancement Layer with Array ops
    enh_input = np.hstack([feature_maps, 0.1*np.ones((train_samples, 1))])
    W_enh = LA.orth(2*random.randn(enh_input.shape[1], N3) - 1)
    enh_output = tansig(enh_input.dot(W_enh) * (s/np.max(enh_input.dot(W_enh))))
    
    # Final Output Weights
    final_input = np.hstack([feature_maps, enh_output])
    W_out = pinv(final_input, c).dot(train_y)
    
    # Training metrics
    train_pred = final_input.dot(W_out)
    train_acc = show_accuracy(train_pred, train_y)
    train_time = time.time() - train_start
    
    # Testing Phase
    test_start = time.time()
    test_x = preprocessing.scale(test_x)
    test_x_bias = np.hstack([test_x, 0.1*np.ones((test_x.shape[0], 1))])
    test_features = np.zeros((test_x.shape[0], N2*N1))
    
    for i in range(N2):
        test_features[:, i*N1:(i+1)*N1] = scalers[i].transform(test_x_bias.dot(betas[i]))
    
    test_enh = tansig(np.hstack([test_features, 0.1*np.ones((test_x.shape[0], 1))]).dot(W_enh))
    test_final = np.hstack([test_features, test_enh]).dot(W_out)
    test_acc = show_accuracy(test_final, test_y)
    test_time = time.time() - test_start
    
    return train_acc, test_acc, train_time, test_time

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X_train = random.randn(1000, 10)
    y_train = random.randint(0, 2, 1000)
    X_test = random.randn(200, 10)
    y_test = random.randint(0, 2, 200)
    
    # Convert to one-hot encoding using array operations
    y_train = np.eye(2)[y_train]
    y_test = np.eye(2)[y_test]
    
    # Run BLS with array-compliant implementation
    train_acc, test_acc, train_time, test_time = BLS(
        X_train, y_train, X_test, y_test,
        N1=20, N2=10, N3=100
    )
    
    print(f"Training Accuracy: {train_acc*100:.2f}% | Time: {train_time:.2f}s")
    print(f"Testing Accuracy: {test_acc*100:.2f}% | Time: {test_time:.2f}s")