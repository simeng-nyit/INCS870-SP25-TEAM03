# -*- coding: utf-8 -*-
"""
Modified Broad Learning System with Feature Bundling Diagnostics
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
from itertools import combinations

def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in range(Label.shape[0]):
        if label_1[j] == predlabel[j]:
            count += 1
    return round(count/len(Label), 5)

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def pinv(A, reg):
    return np.linalg.pinv(reg*np.eye(A.shape[1]) + A.T @ A) @ A.T

def shrinkage(a, b):
    return np.maximum(a - b, 0) - np.maximum(-a - b, 0)

def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    L1 = np.linalg.inv(AA + np.eye(m))
    L2 = (L1.dot(A.T)).dot(b)
    wk, ok, uk = x1.copy(), x1.copy(), x1.copy()
    for _ in range(itrs):
        ck = L2 + L1.dot(ok - uk)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
    return ok

def greedy_feature_bundling(F, K):
    n_features = F.shape[1]
    conflict = np.zeros((n_features, n_features), dtype=int)
    
    for i, j in combinations(range(n_features), 2):
        corr = abs(np.corrcoef(F[:,i], F[:,j])[0,1])
        conflict[i,j] = conflict[j,i] = corr > 0.5
    
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(-degrees)
    
    bundles = []
    for feature in order:
        placed = False
        for bundle in bundles:
            if sum(conflict[feature, bundle]) <= K:
                bundle.append(feature)
                placed = True
                break
        if not placed:
            bundles.append([feature])
    return bundles

def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, K=None):
    # Training Phase
    train_start = time.time()
    train_x = preprocessing.scale(train_x, axis=1)
    original_features = train_x.shape[1]
    bundles = [range(original_features)] * N2

    # Feature Mapping Layer
    OutputOfFeatureMapping = np.zeros((train_x.shape[0], N2*N1))
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []

    for i in range(N2):
        bundle_features = bundles[i]
        FeatureSubset = np.hstack([train_x[:,bundle_features], 
                                 0.1*np.ones((train_x.shape[0],1))])
        
        random.seed(i)
        weight = 2*random.randn(len(bundle_features)+1, N1)-1
        mapped_features = np.dot(FeatureSubset, weight)
        
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        normalized_features = scaler.fit_transform(mapped_features)
        beta = sparse_bls(normalized_features, FeatureSubset).T
        
        output = np.dot(FeatureSubset, beta)
        dist = np.max(output, axis=0) - np.min(output, axis=0)
        min_val = np.min(output, axis=0)
        
        Beta1OfEachWindow.append(beta)
        distOfMaxAndMin.append(dist)
        minOfEachWindow.append(min_val)
        OutputOfFeatureMapping[:, N1*i:N1*(i+1)] = (output - min_val)/dist

    # Feature Bundling Diagnostics
    train_bundles = None
    if K is not None:
        train_bundles = greedy_feature_bundling(OutputOfFeatureMapping, K)
        
        # Print bundle information
        print("\n" + "="*60)
        print("Feature Bundle Diagnostics:")
        print(f"Original input features: {train_x.shape[1]}")
        print(f"Feature mapping dimension: {N2} groups x {N1} nodes = {N2*N1}")
        print(f"Reduced to {len(train_bundles)} bundles with max conflict {K}")
        
        print("\nBundle Details (feature mapping columns grouped):")
        for idx, bundle in enumerate(train_bundles):
            print(f"Bundle {idx+1}: {sorted(bundle)} ({len(bundle)} features)")
            
        print("\nDimension Comparison:")
        print(f"{'Stage':<20} | {'Features':<10} | {'Reduction %'}")
        print(f"{'Original Input':<20} | {train_x.shape[1]:<10} | -")
        print(f"{'Feature Mapping':<20} | {N2*N1:<10} | -")
        print(f"{'After Bundling':<20} | {len(train_bundles):<10} | "
              f"{(1 - len(train_bundles)/(N2*N1)):.1%}")
        print("="*60 + "\n")

        # Apply bundling
        bundled_train = np.zeros((OutputOfFeatureMapping.shape[0], len(train_bundles)))
        for idx, bundle in enumerate(train_bundles):
            bundled_train[:, idx] = OutputOfFeatureMapping[:, bundle].sum(axis=1)
        OutputOfFeatureMapping = bundled_train

    # Enhancement Layer
    M = OutputOfFeatureMapping.shape[1]
    InputWithBias = np.hstack([OutputOfFeatureMapping, 
                             0.1*np.ones((train_x.shape[0],1))])
    
    if M >= N3:
        random.seed(67797325)
        enhance_weights = LA.orth(2*random.randn(M+1, N3)-1)
    else:
        random.seed(67797325)
        enhance_weights = LA.orth(2*random.randn(M+1, N3).T-1).T
    
    enhanced = tansig(np.dot(InputWithBias, enhance_weights)*(s/np.max(InputWithBias)))
    final_input = np.hstack([OutputOfFeatureMapping, enhanced])
    
    # Train output weights
    OutputWeight = np.dot(pinv(final_input, c), train_y)
    train_end = time.time()

    # Testing Phase
    test_start = time.time()
    test_x = preprocessing.scale(test_x, axis=1)
    OutputOfFeatureMappingTest = np.zeros((test_x.shape[0], N2*N1))

    for i in range(N2):
        bundle_features = bundles[i]
        FeatureSubsetTest = np.hstack([test_x[:,bundle_features],
                                     0.1*np.ones((test_x.shape[0],1))])
        
        outputTest = np.dot(FeatureSubsetTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingTest[:, N1*i:N1*(i+1)] = \
            (outputTest - minOfEachWindow[i])/distOfMaxAndMin[i]

    # Apply bundling to test data
    if K is not None:
        print("Applying to test data:")
        print(f"Original test features: {OutputOfFeatureMappingTest.shape[1]}")
        print(f"Using {len(train_bundles)} bundles from training")
        
        bundled_test = np.zeros((OutputOfFeatureMappingTest.shape[0], len(train_bundles)))
        for idx, bundle in enumerate(train_bundles):
            bundled_test[:, idx] = OutputOfFeatureMappingTest[:, bundle].sum(axis=1)
        OutputOfFeatureMappingTest = bundled_test
        print(f"Bundled test features: {OutputOfFeatureMappingTest.shape[1]}\n")

    # Test enhancements
    TestInputWithBias = np.hstack([OutputOfFeatureMappingTest,
                                 0.1*np.ones((test_x.shape[0],1))])
    
    enhanced_test = tansig(np.dot(TestInputWithBias, enhance_weights)*(s/np.max(TestInputWithBias)))
    final_test_input = np.hstack([OutputOfFeatureMappingTest, enhanced_test])
    test_end = time.time()

    # Results
    train_pred = np.dot(final_input, OutputWeight)
    test_pred = np.dot(final_test_input, OutputWeight)
    
    print(f"{'Training Accuracy:':<20} {show_accuracy(train_pred, train_y)*100:.2f}%")
    print(f"{'Training Time:':<20} {train_end - train_start:.2f}s")
    print(f"{'Testing Accuracy:':<20} {show_accuracy(test_pred, test_y)*100:.2f}%")
    print(f"{'Testing Time:':<20} {test_end - test_start:.2f}s")
    
    return (show_accuracy(test_pred, test_y),
            test_end - test_start,
            show_accuracy(train_pred, train_y),
            train_end - train_start)