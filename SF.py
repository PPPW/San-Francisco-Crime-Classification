#! /usr/bin/python

""" San Francisco Crime Classification. 

"""

__author__ = 'Pei Wang'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import time
import sys
import os
import multiprocessing
import gzip

from sklearn import preprocessing, cross_validation, linear_model
from sklearn.learning_curve import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier,\
                             AdaBoostClassifier, \
                             BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
#import NeuralNetwork as nn

#import theano
#import theano.tensor as T
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh
#from theano.tensor.nnet import sigmoid
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

# ----------------------------------------------------------------------


def loadData():
    """ Load the data, prepocess it and save it for future use. 

    This function only needs to be run for one time, after this, one can load the
    picled data. 
 
    """
    data = pd.read_csv('train.csv', header = 0)
    data = data.drop(['Descript','Resolution'],1)
    #data = dataAll.ix[:500,:]

    dataSub = pd.read_csv('test.csv', header = 0)

    data = preprocess(data)
    le = preprocessing.LabelEncoder()
    le.fit(data.Category)
    data.Categoy = le.transform(data.Category)
    y_ = data['Category'].astype('category')
    data = data.drop('Category', 1)

    dataSub = dataSub.drop('Id', 1)
    dataSub = preprocess(dataSub)
 
    # standarize
    ss = preprocessing.StandardScaler()
    ss.fit(data)
    data[data.columns] = ss.transform(data)
    ss.fit(dataSub)
    dataSub[dataSub.columns] = ss.transform(dataSub)

    print data.columns
    print dataSub.columns

    # after deleting outliers, index is not continous
    #data.index = range(data.shape[0])
    #y_.index = range(len(y_))
    #dataSub.index = range(dataSub.shape[0])
    #print type(data), type(dataSub), type(y_)

    pickle.dump((data, y_), open('data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(le.classes_, open('category.pkl','wb'),pickle.HIGHEST_PROTOCOL)
    pickle.dump(dataSub, open('dataSub.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

def preprocess(data):
    """ Preprocess the raw data. 

    Args:
        Raw data in Pandas DataFrame format.
    Returns: 
        Preprocessed data in Pandas DataFrame format.

    """

    # outliers
    data = data[abs(data["X"]) < 121.0]    
    data.index = range(data.shape[0])

    # add features from address
    data['atCross']=data['Address'].apply(lambda x: 1 if "/" in x else 0)
    data['inBlock']=data['Address'].apply(lambda x: 1 if 'block' in x.lower() else 0)

    # categorical variables
    categoricals = []
    for name in ['DayOfWeek','PdDistrict']:
        dummy = pd.get_dummies(data[name], prefix=name)
        categoricals.append(dummy)
        data = data.drop(name, 1)
    categoricals.append(data)
    data = pd.concat(categoricals, 1)

    # Address needs more preprocessing
    data = data.drop('Address', 1)

    # Dates
    le = preprocessing.LabelEncoder()
    year = data.Dates.map(lambda x:pd.to_datetime(x).year)
    le.fit(year)
    data['year'] = le.transform(year)
    #print le.classes_
    month = data.Dates.map(lambda x:pd.to_datetime(x).month) 
    le.fit(month)
    data['month'] = le.transform(month)
    #print le.classes_
    day = data.Dates.map(lambda x:pd.to_datetime(x).day)
    le.fit(day)
    data['day'] = le.transform(day)
    #print le.classes_
    hour = data.Dates.map(lambda x:pd.to_datetime(x).hour)
    le.fit(hour)
    data['hour'] = le.transform(hour)
    #print le.classes_

    data = data.drop('Dates', 1)

    # add more features

    data['awake'] = data['hour'].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    #print data['awake'].unique()
    data['season'] = data['month'].apply(lambda x : ((x-3)/3)%4)
    #print data['season'].unique()
    #data['isDup'] = pd.Series(data.duplicated()|data.duplicated(take_last=True)).apply(int)
    #print data['isDup'].unique()

    return data

def shuffle(X, y):
    """ Shuffle the data. 

    Args:
        param1: data except for the labels
        param2: the labels
    Returns: 
        Randomly shuffled data. 

    """

    np.random.seed(2)
    index = np.arange(len(y))
    np.random.shuffle(index)
    X = X.iloc[index]
    y = y[index]
    return X, y

def pca(X_train, X_test):
    """ Apply PCA 

    Args:
        param1: training data
        param2: testing data
    Returns: 
        training and testing data after PCA.

    """

    pcaModel = PCA(n_components=50)
    pcaModel.fit(X_train)
    #print pcaModel.explained_variance_ratio_
    return pcaModel.transform(X_train), pcaModel.transform(X_test)

'''
# log loss
def log_loss(y, P, eps=1e-15):
    P = np.array(P)
    #P = P / np.sum(P, axis=1)[:,None]
    p = P[range(y.size),y]
    #p = [P[i][y[i]] for i in range(y.size)]
    p = np.clip(p, eps, 1-eps)
    return -np.mean(np.log(p))
'''
'''
# cross validation, singel process version
def cv(Xall, y, clf):
    startTime = time.clock()
    X = Xall.iloc[:len(y),:]
    Xsub = Xall.iloc[len(y):, :]
    #print X.columns
    #print Xsub.columns
    
    # if 'final', do prediction on all the data; otherwise, only on train.csv
    # if 'save', save result into submit.csv
    if sys.argv[2] == 'final':
        clf.fit(X, y)
        finalProbs = pd.DataFrame(clf.predict_proba(Xsub))
        if sys.argv[3] == 'save':
            categories = pickle.load(open('category.pkl'))
            finalProbs.columns = categories            
            #finalProbs['Id'] = range(Xsub.shape[0])
            finalProbs.to_csv("submit.csv", header=True, index_label='Id')

    #kf = cross_validation.KFold(len(y), n_folds=4, indices=None, shuffle=False, random_state=5)
    kf = cross_validation.StratifiedKFold(y, n_folds=4, indices=None, shuffle=False, random_state=1)

    probs = []
    stackIndex = []    
    for train_index, test_index in kf:        
        stackIndex.append(test_index)
        print "Model %s, cross validation on %d" % (sys.argv[1], test_index[0])
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        assert len(y_train.unique()) == len(y.unique())
        assert len(y_test.unique()) == len(y.unique())        
        #print X_train.info()
        #print y_train[y_train.isnull()]
        clf.fit(X_train, y_train)
        
        testProb = clf.predict_proba(X_test)
        print log_loss(y_train, clf.predict_proba(X_train)), log_loss(y_test, testProb)
        probs.append(pd.DataFrame(testProb, index=test_index))    

    # save predicts for combining models
    probsAll = pd.concat(probs, 0)
    probsAll = probsAll.iloc[np.hstack(stackIndex)]
    if sys.argv[2] == 'final':
        probsAll = pd.concat([probsAll, finalProbs], 0)

    pickle.dump(probsAll, open('%s.pkl' % sys.argv[1], 'wb'), 
                pickle.HIGHEST_PROTOCOL)    
    print time.clock()-starTime

'''
# ----------------------------------------------------------------------

# for multiprocessing
def fitting(clf, train_index, test_index, X, y):
    startTime = time.clock()
    #print clf
    #print X.info()
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    assert len(y_train.unique()) == len(y.unique())
    assert len(y_test.unique()) == len(y.unique())        

    # apply PCA
    #X_train, X_test = pca(X_train, X_test)
    
    clf.fit(X_train, y_train)
    
    #    features = X_train.columns
    #pickle.dump(pd.DataFrame(clf.feature_importances_, index=features), 
    #open('rfimportance.pkl', 'wb'),
    #pickle.HIGHEST_PROTOCOL) 
    #pickle.dump(pd.DataFrame(clf.coef_, columns=features), 
    #            open('lrcoef.pkl', 'wb'),
    #           pickle.HIGHEST_PROTOCOL) 
    #print intercept_
    
    testProb = clf.predict_proba(X_test)
    trainScore = log_loss(y_train, clf.predict_proba(X_train))
    testScore = log_loss(y_test, testProb)

    print trainScore, testScore
    print time.clock()-startTime
    return pd.DataFrame(testProb, index=test_index), trainScore, testScore       
                
# cross validation, multiprocess version 
def cv(Xall, y, clf):
    X = Xall.iloc[:len(y)]
    Xsub = Xall.iloc[len(y):]
    
    # if 'final', do prediction on all the data; otherwise, only on train.csv
    # if 'save', save result into submit.csv
    if sys.argv[2] == 'final':
        # apply PCA
        #X, Xsub = pca(X, Xsub)
        
        clf.fit(X, y)
        finalProbs = pd.DataFrame(clf.predict_proba(Xsub))
        if sys.argv[3] == 'save':
            categories = pickle.load(open('category.pkl'))
            finalProbs.columns = categories            
            with gzip.GzipFile('submit.csv.gz', mode='w') as gzfile:
                finalProbs.to_csv(gzfile, header=True, index_label='Id')
        exit()

    # use k-fold and save probs on X for combining models later
    #kf = cross_validation.KFold(len(y), n_folds=4, indices=None, shuffle=False, random_state=1)
    kf = cross_validation.StratifiedKFold(y, n_folds=4, indices=None, shuffle=False, random_state=1)
    
    f= open('log', 'a')
    stackIndex = []
    rInPool = []
    pool = multiprocessing.Pool(4)
    for train_index, test_index in kf:
        stackIndex.append(test_index)
        print "Model %s, cross validation on %d" % (sys.argv[1], test_index[0])
        r = pool.apply_async(fitting, (clf, train_index, test_index, X, y))
        rInPool.append(r)
    probs = []
    scores = []
    for r in rInPool:
        probs.append(r.get()[0])
        scores.append(r.get()[1:])

    pool.close()
    pool.join()    
    
    f.write('Model %s\n' % sys.argv[1])
    for s in scores:
        f.write('%f %f\n' % (s[0], s[1]))
    f.close()

    # save predicts for combining models
    probsAll = pd.concat(probs, 0)
    probsAll = probsAll.iloc[np.hstack(stackIndex)]
    if sys.argv[2] == 'final':
        probsAll = pd.concat([probsAll, finalProbs], 0)

    pickle.dump(probsAll, open('%s.pkl' % sys.argv[1], 'wb'), 
                pickle.HIGHEST_PROTOCOL)    

# ----------------------------------------------------------------------

#loadData()
#exit()

X, y = pickle.load(open('data.pkl'))
Xsub = pickle.load(open('dataSub.pkl'))

#X.drop(['logodds33', 'logodds22'], 1)
#X.drop('IsDup', 1)

#X, y = pickle.load(open('elio.pkl'))
#X = pd.DataFrame(X)
#Xsub = X.iloc[0,:]

if sys.argv[1] == 'lr':
    clf = linear_model.LogisticRegression(C=1,random_state=33)
 
if sys.argv[1] == 'softmax':
    clf = linear_model.LogisticRegression(C=1,multi_class='multinomial',solver='lbfgs')
    
if sys.argv[1] == 'knn':
    clf = KNeighborsClassifier(n_neighbors = 40)
 
if sys.argv[1] == 'sgd':
    clf = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, n_iter=53, epsilon=0.1, n_jobs=8,learning_rate='optimal')

if sys.argv[1] == 'rf':
    clf = RandomForestClassifier(n_estimators=160,criterion='entropy',n_jobs=1,max_depth=12,max_features=0.06)

if sys.argv[1] == 'lasagne': 
    clf = NeuralNet(layers=
                    [('input', InputLayer),
                     ('dense0', DenseLayer),
                     ('dropout', DropoutLayer),
                     ('dense1', DenseLayer),
                     ('output', DenseLayer)],
                    input_shape=(None, 70),
                    dense0_num_units=200,
                    dense0_nonlinearity=sigmoid,
                    dropout_p=0.1,
                    dense1_num_units=200,
                    dense1_nonlinearity=sigmoid,
                    output_num_units=39,
                    output_nonlinearity=softmax,                 
                    update=nesterov_momentum,
                    update_learning_rate=0.3,
                    update_momentum=0.8,
                    objective_loss_function = categorical_crossentropy,
                    #eval_size=0.25,
                    train_split=TrainSplit(0.25),
                    verbose=1,
                    max_epochs=20)
    
    X, y = shuffle(X, y)
    X = X.values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y).astype(np.int32)
    categories = le.classes_
    clf.fit(X,y)    
    #clf.save_weights_to('lsn_weights')
    print log_loss(y, clf.predict_proba(X))
    #exit()

    #clf.initialize_layers()
    #clf.load_weights_from('lsn_weights')
    finalProbs = pd.DataFrame(clf.predict_proba(Xsub))
    finalProbs.columns = categories
    with gzip.GzipFile('submit.csv.gz', mode='w') as gzfile:
        finalProbs.to_csv(gzfile, header=True, index_label='Id')
    exit()

# combining models
models = [
    'lr',
    #'softmax', 
    #'knn',
    'rf',
    ]

# stacking via logistic regression
if sys.argv[1] == 'stackLog':    
    combined = []
    for m in models:
        probs = pickle.load(open('%s.pkl' % m))
        combined.append(probs)
    combined = pd.concat(combined, 1)

    clf = linear_model.LogisticRegression()
    #clf = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
        
    cv(combined, y, clf)    

'''
# stacking via neural network
# file 'NeuralNetwork.py' is missing
elif sys.argv[1] == 'stacknn':
    
    combined = []
    for m in models:
        probs = pickle.load(open('%s.pkl' % m))
        combined.append(probs)
    combined = pd.concat(combined, 1)

    layers = [combined.shape[1], 20, 39]
    np.random.seed(1)
    clf = nn.NeuralNetwork(lambda_=1.0, layers=layers, callback=None, maxfun=310)
    
    cv(combined, y, clf)

    '''    
    layers = [X.shape[1], 18, 39]
    np.random.seed(13)
    clf = nn.NeuralNetwork(lambda_=1, layers=layers, callback=None, maxfun=310)  
    cv(X, y, clf)    
    '''
# for all individual models
else:
    cv(pd.concat([X, Xsub], 0), y, clf)
'''
print 'Model %s is done!\n' % sys.argv[1]
