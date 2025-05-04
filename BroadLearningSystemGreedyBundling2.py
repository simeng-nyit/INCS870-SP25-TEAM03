# -*- coding: utf-8 -*-
"""
Modified Broad Learning System with Greedy Feature Bundling
Incorporated Greedy Bundling Algorithm from LightGBM paper
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
from itertools import combinations

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
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
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
    L1 = np.linalg.inv(AA + np.eye(m))  # Using direct inversion
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def greedy_feature_bundling(F, K):
    """Greedy Feature Bundling Algorithm from LightGBM paper"""
    n_features = F.shape[1]
    conflict = np.zeros((n_features, n_features), dtype=int)
    
    # Calculate feature conflicts using absolute correlation
    for i, j in combinations(range(n_features), 2):
        corr = abs(np.corrcoef(F[:,i], F[:,j])[0,1])
        conflict[i][j] = corr > 0.5  # Threshold for conflict
        conflict[j][i] = conflict[i][j]
    
    # Sort features by conflict degree (descending)
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(-degrees)
    
    bundles = []
    for feature in order:
        placed = False
        for bundle in bundles:
            # Check if feature can be added to existing bundle
            if sum(conflict[feature][bundle]) <= K:
                bundle.append(feature)
                placed = True
                break
        if not placed:
            bundles.append([feature])
    
    return bundles

def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, K=None):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    original_features = train_x.shape[1]
    
    # Always use original features for initial mapping
    bundles = [range(original_features)] * N2

    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    
    time_start=time.time()
    
    # Build initial feature mapping layer
    for i in range(N2):
        bundle_features = bundles[i]
        FeatureSubset = np.hstack([train_x[:,bundle_features], 
                                 0.1 * np.ones((train_x.shape[0],1))])
        
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(len(bundle_features)+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureSubset, weightOfEachWindow)
        
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureSubset).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        
        outputOfEachWindow = np.dot(FeatureSubset, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow

    # Apply bundling to feature mapping layer output
    feature_mapping_bundles = None
    if K is not None:
        feature_mapping_bundles = greedy_feature_bundling(OutputOfFeatureMappingLayer, K)
        # Aggregate bundled features
        bundled_output = np.zeros((OutputOfFeatureMappingLayer.shape[0], len(feature_mapping_bundles)))
        for i, bundle in enumerate(feature_mapping_bundles):
            bundled_output[:, i] = OutputOfFeatureMappingLayer[:, bundle].sum(axis=1)
        OutputOfFeatureMappingLayer = bundled_output
        print(f"\nTraining feature matrix dimension after bundling: {OutputOfFeatureMappingLayer.shape}")

    # Prepare enhance layer input
    M = OutputOfFeatureMappingLayer.shape[1]
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 
                                           0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if M >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(M+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(M+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    # Generate final input
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = np.dot(pinvOfInput,train_y) 
    time_end=time.time() 
    trainTime = time_end - time_start
    
    # Training metrics
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accuracy is',trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    
    # Testing phase
    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        bundle_features = bundles[i]
        FeatureSubsetTest = np.hstack([test_x[:,bundle_features], 
                                     0.1 * np.ones((test_x.shape[0],1))])
        outputOfEachWindowTest = np.dot(FeatureSubsetTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]

    # Apply same bundling to test data
    if K is not None and feature_mapping_bundles is not None:
        bundled_output_test = np.zeros((OutputOfFeatureMappingLayerTest.shape[0], len(feature_mapping_bundles)))
        for i, bundle in enumerate(feature_mapping_bundles):
            bundled_output_test[:, i] = OutputOfFeatureMappingLayerTest[:, bundle].sum(axis=1)
        OutputOfFeatureMappingLayerTest = bundled_output_test

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 
                                               0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accuracy is',testAcc * 100,'%')
    print('Testing time is ',testTime,'s')

    return testAcc, testTime, trainAcc, trainTime

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    y = np.eye(10)[y]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # BLS with Feature Mapping Layer Bundling
    print("\nBLS with Feature Mapping Layer Bundling:")
    BLS(X_train, y_train, X_test, y_test, s=0.5, c=1e-5, N1=10, N2=10, N3=50, K=3)