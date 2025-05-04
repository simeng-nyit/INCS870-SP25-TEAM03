# -*- coding: utf-8 -*-
""" 
BLS with GOSS (Gradient-based One-Side Sampling) Integration
Modified based on LightGBM's GOSS algorithm from "GOSS: Gradient-based One-Side Sampling"
"""
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time

# GOSS ADDITION: New GOSS functions
def compute_gradients(y_true, y_pred):
    return y_pred - y_true  # Cross-entropy gradient for softmax

def apply_goss(gradients, top_ratio=0.2, other_ratio=0.5):
    sorted_indices = np.argsort(-np.abs(gradients).mean(axis=1))
    top_n = int(len(gradients) * top_ratio)
    other_n = int(len(gradients) * other_ratio)
    
    top_indices = sorted_indices[:top_n]
    other_indices = np.random.choice(
        sorted_indices[top_n:], 
        size=other_n, 
        replace=False
    )
    
    selected_indices = np.concatenate([top_indices, other_indices])
    weights = np.ones_like(selected_indices, dtype=np.float32)
    weights[top_n:] *= (1 - top_ratio)/other_ratio  # Weight compensation
    return selected_indices, weights

def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def pinv(A, reg):
    return np.linalg.pinv(reg*np.eye(A.shape[1]) + A.T @ A) @ A.T

def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z

def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.linalg.inv(AA + np.eye(m))
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

# GOSS ADDITION: Modified BLS function
def BLS_GOSS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, top_ratio=0.2, other_ratio=0.5):
    # Initialize timers
    train_start = time.time()
    
    # Initial processing remains the same
    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    
    # Phase 1: Initial Training without GOSS
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    
    if N1*N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    
    # GOSS ADDITION: First phase training
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)
    
    # GOSS ADDITION: Calculate gradients
    goss_start = time.time()
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    gradients = compute_gradients(train_y, OutputOfTrain)
    
    # Apply GOSS sampling
    selected_indices, sample_weights = apply_goss(gradients, top_ratio, other_ratio)
    
    # GOSS ADDITION: Second phase training with selected samples
    WeightedInput = InputOfOutputLayer[selected_indices] * sample_weights[:, np.newaxis]
    WeightedOutput = train_y[selected_indices] * sample_weights[:, np.newaxis]
    
    pinvOfInput_goss = pinv(WeightedInput, c)
    OutputWeight = np.dot(pinvOfInput_goss, WeightedOutput)
    goss_duration = round(time.time() - goss_start, 2)

    total_train_time = round(time.time() - train_start, 2)
    
    # Testing remains the same
    test_start = time.time()
    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2*N1])
    
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1*i:N1*(i+1)] = (outputOfEachWindowTest - minOfEachWindow[i])/distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    test_duration = round(time.time() - test_start, 2)
    
    # Return metrics
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    testAcc = show_accuracy(OutputOfTest, test_y)
    
    print(f'Training Accuracy: {trainAcc*100:.2f}%')
    print(f'Testing Accuracy: {testAcc*100:.2f}%')
    
    return (testAcc,          # [0] Test accuracy
            total_train_time, # [1] Total training time
            trainAcc,         # [2] Training accuracy
            test_duration)    # [3] Testing time