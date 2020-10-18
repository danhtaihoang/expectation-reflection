#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,roc_curve,auc

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from expectation_reflection import ExpectationReflection

import expectation_reflection_cv as ERCV
import expectation_reflection_cvmc as ERCVMC
from function import split_train_test,make_data_balance 
#-------------------------------------------------------------------------------------------------

np.random.seed(1)
data_id = sys.argv[1]

data_name_list = ['kidney','breastcancer','diabetes','diabetic_retinopathy',
    'orthopedics', 'breasttissue','protein_expression','gene','cardiotocography']

#data_id = 0

data_id = int(data_id)
data_name = data_name_list[data_id]
print(data_name)

# load data
X = np.loadtxt('../%s_X.txt'%data_name)
y = np.loadtxt('../%s_y.txt'%data_name)
#print('ini:',np.unique(y,return_counts=True))

X,y = make_data_balance(X,y)
#print('balance:',np.unique(y,return_counts=True))

X, y = shuffle(X, y)
X = MinMaxScaler().fit_transform(X)

#-------------------------------------------------------------------------------------------------
def inference(X_train,y_train,X_test,y_test):     
    n = X_train.shape[1]
    m = len(np.unique(y_train))

    l2 = np.logspace(-5,1,20,base=10.0)
    #l2 = [0.,0.00001,0.0001,0.001,0.01,0.1,1.]
    #l2 = [0.1]
    nl2 = len(l2)
              
    kf = 3   
    kfold = KFold(n_splits=kf,shuffle=False,random_state=1)

    # predict with ER
    #if m == 2:
    h01 = np.zeros(kf)
    w1 = np.zeros((kf,n))
    acc1 = np.zeros(kf)
    
    h0 = np.zeros(nl2)
    w = np.zeros((nl2,n))
    acc = np.zeros(nl2)            
    for il2 in range(nl2):            
        for i,(train_index,val_index) in enumerate(kfold.split(y_train)):
            X_train1, X_val = X_train[train_index], X_train[val_index]
            y_train1, y_val = y_train[train_index], y_train[val_index]
            h01[i],w1[i,:] = ERCV.fit(X_train1,y_train1,X_val,y_val,niter_max=1000,l2=l2[il2])
            
            y_val_pred,p_val_pred = ERCV.predict(X_val,h01[i],w1[i])
            #acc1[i] = accuracy_score(y_val,y_val_pred)
            acc1[i] = ((p_val_pred - y_val)**2).mean()
                                        
        h0[il2] = h01.mean(axis=0)
        w[il2,:] = w1.mean(axis=0)
        acc[il2] = acc1.mean()
    
    il2_select = np.argmin(acc)
    #print('l2:',l2[il2_select])
    y_pred,p_pred = ERCV.predict(X_test,h0[il2_select],w[il2_select,:])

    # entire training set:
    #model = ExpectationReflection(niter_max=1000,l2=l2[il2_select])
    #model.fit(X_train,y_train)
    #y_pred = model.predict(X_test)
                                        
    accuracy = accuracy_score(y_test,y_pred)
    cost = ((p_pred - y_test)**2).mean()
  
    # Compute ROC curve and ROC area for each class
    fpr, tpr,thresholds = roc_curve(y_test, p_pred, drop_intermediate=False)
    roc_auc= auc(fpr, tpr)

    print(len(p_pred),len(y_test),len(fpr),len(tpr))                       

    return accuracy,cost,roc_auc
#-------------------------------------------------------------------------------------------------
def compare_inference(X,y,train_size):
    npred = 5
    acc = np.zeros(npred)
    cost = np.zeros(npred)
    roc_auc = np.zeros(npred)
    for ipred in range(npred):
        X_train,X_test,y_train,y_test = split_train_test(X,y,train_size,test_size=0.2)

        #X_train = MinMaxScaler().fit_transform(X_train)
        #X_test = MinMaxScaler().fit_transform(X_test)
        
        # 2019.07.15
        acc[ipred],cost[ipred],roc_auc[ipred] = inference(X_train,y_train,X_test,y_test)
            
    return acc.mean(axis=0),acc.std(),cost.mean(),cost.std(),roc_auc.mean(),roc_auc.std()
#-------------------------------------------------------------------------------------------------
list_train_size = [0.8]
n_size = len(list_train_size)
acc = np.zeros((n_size,6)) # 0: acc, 1: acc_std, 2: cost, 3: cost_std
for i,train_size in enumerate(list_train_size):
    acc[i,:] = compare_inference(X,y,train_size)
    print(train_size,acc[i,:])

np.savetxt('%s_acc.dat'%data_name,acc,fmt='%f')
