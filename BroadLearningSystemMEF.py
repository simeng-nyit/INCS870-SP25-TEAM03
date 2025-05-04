# -*- coding: utf-8 -*-
"""Broad Learning System with Exclusive Feature Bundling (Fixed for NumPy 2.0+)"""
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time

# ---------------------- Helper Functions ----------------------
def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    return round(np.sum(label_1 == predlabel) / len(Label), 5)

def tansig(x):
    return 2/(1+np.exp(-2*x)) - 1

def pinv(A, reg):
    return np.asmatrix(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)  # Fixed line

def shrinkage(a, b):
    return np.maximum(a - b, 0) - np.maximum(-a - b, 0)

def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)  
    m, n = A.shape[1], b.shape[1]
    x1 = np.zeros((m, n))
    wk, ok, uk = x1.copy(), x1.copy(), x1.copy()
    L1 = np.linalg.inv(AA + np.eye(m))
    L2 = L1.dot(A.T.dot(b))
    for _ in range(itrs):
        ck = L2 + L1.dot(ok - uk)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
    return ok

# ------------------- Feature Merging Algorithm -------------------
def merge_exclusive_features(X, bundles):
    """LightGBM-style exclusive feature bundling"""
    merged = []
    for bundle in bundles:
        bundle_features = X[:, bundle]
        offset = 0
        merged_feature = np.zeros(X.shape[0])
        for fid in range(bundle_features.shape[1]):
            merged_feature += bundle_features[:, fid] + offset
            offset += int(np.max(bundle_features[:, fid])) + 1
        merged.append(merged_feature)
    return np.column_stack(merged)

# ---------------------- Main BLS Implementation ----------------------
def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, feature_bundles=None):
    # Feature merging
    if feature_bundles is not None:
        train_x = merge_exclusive_features(train_x, feature_bundles)
        test_x = merge_exclusive_features(test_x, feature_bundles)
    
    # Preprocessing
    train_x = preprocessing.scale(train_x, axis=1)
    test_x = preprocessing.scale(test_x, axis=1)
    
    # Store results
    train_acc, test_acc = 0, 0
    
    try:
        # Feature mapping nodes
        FeatureOfInputDataWithBias = np.hstack([train_x, 0.1*np.ones((train_x.shape[0],1))])
        OutputOfFeatureMappingLayer = np.zeros((train_x.shape[0], N2*N1))
        Beta1OfEachWindow = []

        # Generate feature nodes
        for i in range(N2):
            random.seed(i)
            weight = 2*random.randn(train_x.shape[1]+1,N1)-1
            feature = FeatureOfInputDataWithBias.dot(weight)
            scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(feature)
            feature_scaled = scaler.transform(feature)
            beta = sparse_bls(feature_scaled, FeatureOfInputDataWithBias).T
            Beta1OfEachWindow.append(beta)
            OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = FeatureOfInputDataWithBias.dot(beta)

        # Generate enhancement nodes
        InputOfEnhanceLayer = np.hstack([OutputOfFeatureMappingLayer, 0.1*np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
        if N1*N2 >= N3:
            random.seed(67797325)
            weights = LA.orth(2*random.randn(N2*N1+1,N3)-1)
        else:
            random.seed(67797325)
            weights = LA.orth(2*random.randn(N2*N1+1,N3).T-1).T
        
        temp = InputOfEnhanceLayer.dot(weights)
        parameter = s/np.max(temp)
        OutputOfEnhanceLayer = tansig(temp * parameter)

        # Final training
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
        pinv_input = pinv(InputOfOutputLayer, c)
        OutputWeight = pinv_input.dot(train_y)
        
        # Testing
        FeatureOfInputTest = np.hstack([test_x, 0.1*np.ones((test_x.shape[0],1))])
        OutputOfFeatureTest = np.zeros((test_x.shape[0], N2*N1))
        
        for i in range(N2):
            OutputOfFeatureTest[:,N1*i:N1*(i+1)] = FeatureOfInputTest.dot(Beta1OfEachWindow[i])
        
        InputOfEnhanceTest = np.hstack([OutputOfFeatureTest, 0.1*np.ones((OutputOfFeatureTest.shape[0],1))])
        temp_test = InputOfEnhanceTest.dot(weights)
        OutputOfEnhanceTest = tansig(temp_test * parameter)
        
        # Calculate results
        InputOfOutputTest = np.hstack([OutputOfFeatureTest, OutputOfEnhanceTest])
        predict_train = InputOfOutputLayer.dot(OutputWeight)
        predict_test = InputOfOutputTest.dot(OutputWeight)
        
        train_acc = show_accuracy(predict_train, train_y)
        test_acc = show_accuracy(predict_test, test_y)
        
    except Exception as e:
        print(f"Error in BLS computation: {str(e)}")
        return 0, 0

    return train_acc, test_acc