# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
import gc

# ---------------------- Utility Functions ----------------------
def show_accuracy(predictLabel, Label):
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    return round(np.sum(label_1 == predlabel) / len(Label), 5)

def pinv(A, reg):
    return np.asmatrix(reg*np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

def shrinkage(a, b):
    return np.maximum(a - b, 0) - np.maximum(-a - b, 0)

def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    
    AA = A.T.dot(A).astype(np.float32)
    m = A.shape[1]
    n = b.shape[1]
    
    L1 = np.asmatrix(AA + np.eye(m, dtype=np.float32)).I.astype(np.float32)
    L2 = (L1.dot(A.T)).dot(b).astype(np.float32)
    
    wk = ok = uk = np.zeros((m, n), dtype=np.float32)
    
    for _ in range(itrs):
        ck = L2 + L1.dot(ok - uk)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
        
    return wk.astype(np.float32)

# ---------------------- BLS Core Implementation ----------------------
def merge_exclusive_features(X, max_conflict):
    X_nonzero = (X != 0).astype(np.int8)
    n_samples, n_features = X.shape
    bundles = []
    current_bundle = []
    
    for f_idx in range(n_features):
        feature = X_nonzero[:, f_idx]
        conflict_count = 0
        
        for b in current_bundle:
            conflict_count += np.dot(feature, X_nonzero[:, b])
            if conflict_count > max_conflict:
                break
        
        if conflict_count <= max_conflict:
            current_bundle.append(f_idx)
        else:
            bundles.append(current_bundle)
            current_bundle = [f_idx]
            if len(bundles) % 100 == 0:
                gc.collect()
    
    if current_bundle:
        bundles.append(current_bundle)
    
    merged_features = np.zeros((n_samples, len(bundles)), dtype=np.float32)
    for i, bundle in enumerate(bundles):
        merged_features[:, i] = X[:, bundle].sum(axis=1).astype(np.float32)
    
    return merged_features, bundles

def apply_feature_bundles(X, bundles):
    n_samples = X.shape[0]
    merged = np.zeros((n_samples, len(bundles)), dtype=np.float32)
    
    for i, bundle in enumerate(bundles):
        merged[:, i] = X[:, bundle].sum(axis=1).astype(np.float32)
    
    return merged


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, K=None):
    # Initialize timing and memory tracking
    start_time = time.time()
    gc.collect()
    
    # Convert inputs to float32 for memory efficiency
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)

    # Feature bundling stage
    if K is not None:
        train_x, bundles = merge_exclusive_features(train_x, K)
        test_x = apply_feature_bundles(test_x, bundles)
        N2 = len(bundles)  # Update number of feature windows
        del bundles
        gc.collect()

    # Feature Mapping Layer
    # Add bias term
    FeatureOfInputDataWithBias = np.hstack([
        train_x, 
        np.full((train_x.shape[0], 1), 0.1, dtype=np.float32)
    ]).astype(np.float32)
    
    OutputOfFeatureMappingLayer = np.zeros(
        (train_x.shape[0], N2*N1), 
        dtype=np.float32
    )
    
    Beta1OfEachWindow = []
    minOfEachWindow = []
    distOfMaxAndMin = []
    
    # Batch processing parameters
    batch_size = 5000  # Adjust based on available memory
    num_samples = train_x.shape[0]
    
    for i in range(N2):
        # Generate random weights for this window
        random.seed(i)
        weightOfEachWindow = (2 * random.randn(
            train_x.shape[1] + 1,  # Account for bias term
            N1
        ) - 1).astype(np.float32)
        
        # Process data in batches
        for b_start in range(0, num_samples, batch_size):
            b_end = min(b_start + batch_size, num_samples)
            batch_data = FeatureOfInputDataWithBias[b_start:b_end]
            
            # Calculate feature window output
            FeatureOfEachWindow = np.dot(
                batch_data, 
                weightOfEachWindow
            ).astype(np.float32)
            
            # Scale to [0,1]
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            FeatureOfEachWindow = scaler.fit_transform(FeatureOfEachWindow)
            
            # Calculate sparse beta
            beta = sparse_bls(
                FeatureOfEachWindow.astype(np.float32),
                batch_data.astype(np.float32)
            ).T.astype(np.float32)
            
            # Store beta for test phase
            if b_start == 0:  # Only need to store once per window
                Beta1OfEachWindow.append(beta)
            
            # Calculate output
            output = np.dot(batch_data, beta).astype(np.float32)
            
            # Store normalization parameters
            if b_start == 0:  # Calculate once per window
                window_min = output.min(axis=0)
                window_dist = output.max(axis=0) - window_min
                minOfEachWindow.append(window_min)
                distOfMaxAndMin.append(window_dist)
            
            # Normalize and store
            normalized_output = (output - minOfEachWindow[i]) / distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[
                b_start:b_end, 
                N1*i:N1*(i+1)
            ] = normalized_output
            
            # Clean up batch variables
            del FeatureOfEachWindow, output, normalized_output
            gc.collect()
        
        # Clean up window variables
        del weightOfEachWindow, beta
        gc.collect()

    # Enhancement Layer
    InputOfEnhanceLayer = np.hstack([
        OutputOfFeatureMappingLayer,
        np.full((OutputOfFeatureMappingLayer.shape[0], 1), 0.1, dtype=np.float32)
    ]).astype(np.float32)
    
    # Generate enhancement weights
    if N1*N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(
            2 * random.randn(N1*N2 + 1, N3) - 1
        ).astype(np.float32)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(
            2 * random.randn(N1*N2 + 1, N3).T - 1
        ).T.astype(np.float32)
    
    # Calculate enhancement features
    tempOfOutputOfEnhanceLayer = np.dot(
        InputOfEnhanceLayer, 
        weightOfEnhanceLayer
    ).astype(np.float32)
    
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = np.tanh(
        tempOfOutputOfEnhanceLayer * parameterOfShrink
    ).astype(np.float32)
    
    # Final Output Layer
    InputOfOutputLayer = np.hstack([
        OutputOfFeatureMappingLayer,
        OutputOfEnhanceLayer
    ]).astype(np.float32)
    
    # Calculate pseudoinverse
    pinvOfInput = pinv(InputOfOutputLayer, c).astype(np.float32)
    OutputWeight = np.dot(pinvOfInput, train_y).astype(np.float32)
    
    # Training results
    training_time = time.time() - start_time
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    train_acc = show_accuracy(OutputOfTrain, train_y)
    
    # Testing Phase
    test_start_time = time.time()
    
    # Prepare test data with bias
    FeatureOfInputDataWithBiasTest = np.hstack([
        test_x,
        np.full((test_x.shape[0], 1), 0.1, dtype=np.float32)
    ]).astype(np.float32)
    
    OutputOfFeatureMappingLayerTest = np.zeros(
        (test_x.shape[0], N2*N1),
        dtype=np.float32
    )
    
    # Process test data through feature mapping
    for i in range(N2):
        output = np.dot(
            FeatureOfInputDataWithBiasTest,
            Beta1OfEachWindow[i]
        ).astype(np.float32)
        
        OutputOfFeatureMappingLayerTest[:, N1*i:N1*(i+1)] = \
            (output - minOfEachWindow[i]) / distOfMaxAndMin[i]
    
    # Process enhancement layer for test data
    InputOfEnhanceLayerTest = np.hstack([
        OutputOfFeatureMappingLayerTest,
        np.full((test_x.shape[0], 1), 0.1, dtype=np.float32)
    ]).astype(np.float32)
    
    tempOfOutputOfEnhanceLayerTest = np.dot(
        InputOfEnhanceLayerTest,
        weightOfEnhanceLayer
    ).astype(np.float32)
    
    OutputOfEnhanceLayerTest = np.tanh(
        tempOfOutputOfEnhanceLayerTest * parameterOfShrink
    ).astype(np.float32)
    
    # Final test output
    InputOfOutputLayerTest = np.hstack([
        OutputOfFeatureMappingLayerTest,
        OutputOfEnhanceLayerTest
    ]).astype(np.float32)
    
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    testing_time = time.time() - test_start_time
    test_acc = show_accuracy(OutputOfTest, test_y)
    
    # Clean up large variables
    del (FeatureOfInputDataWithBias, OutputOfFeatureMappingLayer,
         InputOfEnhanceLayer, weightOfEnhanceLayer, InputOfOutputLayer,
         pinvOfInput, OutputWeight)
    gc.collect()
    
    return (
        np.array([test_acc]),
        np.array([testing_time]),
        np.array([train_acc]),
        np.array([training_time])
    )

#%%%%%%%%%%%%%%%%%%%%%%%%    
'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''


def BLS_AddEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M):
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append( np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis =0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis =0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = pinvOfInput.dot(train_y) 
    time_end=time.time() 
    trainTime = time_end - time_start
    
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    
    test_x = preprocessing.scale(test_x, axis=1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
 
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start=time.time()
        if N1*N2>= M : 
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M)-1)
        else :
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
        
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.asmatrix(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time
        test_acc[0][e+1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )
        
    return test_acc,test_time,train_acc,train_time


'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''
#%%%%%%%%%%%%%%%%
def BLS_AddFeatureEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M1,M2,M3):
    
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis = 0) - np.min(outputOfEachWindow,axis = 0))
        minOfEachWindow.append(np.mean(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain,c)
    OutputWeight =pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayerTrain,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x,axis = 1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i] 

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
  
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() 
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增加Mapping 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e+N2+u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1,M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)

        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append( np.max(TempOfFeatureOutput,axis = 0) - np.min(TempOfFeatureOutput,axis = 0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput,axis = 0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]

        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer,outputOfNewWindow])

        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0],1))])

        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2]).T-1).T  
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)
        
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)

        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2*N1+e*M1>=M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)

        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)
        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow,outputOfNewFeatureEhanceNodes,OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain,OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        
        if C.all() == 0:
            w = D.shape[1]
            B = np.asmatrix(np.eye(w) - D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeight = pinvOfInput.dot(train_y)        
        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        TrainingAccuracy = show_accuracy(predictLabel,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        
        # 测试过程
        #先生成新映射窗口输出
        time_start = time.time() 
        WeightOfNewMapping =  Beta1OfEachWindow[N2+e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping )
        
        outputOfNewWindowTest = (outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e] 
        
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,outputOfNewWindowTest])
        
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest,0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0],1])])
        
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest,0.1*np.ones([outputOfNewWindowTest.shape[0],1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        
        OutputOfRelateEnhanceNodes = tansig(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)
        
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest,outputOfNewWindowTest,OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
    
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        TestingAccuracy = show_accuracy(predictLabel,test_y)
        time_end = time.time()
        Testing_time= time_end - time_start
        test_time[0][e+1] = Testing_time
        test_acc[0][e+1]=TestingAccuracy
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' )

    return test_acc,test_time,train_acc,train_time