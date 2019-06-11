##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler

""" --------------------------------------------------------------------------------------
2019.06.11: Synthesize data according to the kinetic ising model
Input: data length l, number of variable n, std of interactions g
Output: w[n], X[l,n],y[l]
"""
def synthesize_data(l,n,g,data_type='discrete'):        
    if data_type == 'binary':
        X = np.sign(np.random.rand(l,n)-0.5)
        w = np.random.normal(0.,g,size=n)
        
    if data_type == 'continuous':
        X = 2*np.random.rand(l,n)-1
        w = np.random.normal(0.,g,size=n)
        
    if data_type == 'categorical':        
        from sklearn.preprocessing import OneHotEncoder
        m = 5 # categorical number for each variables
        # initial s (categorical variables)
        s = np.random.randint(0,m,size=(l,n)) # integer values
        onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
        X = onehot_encoder.fit_transform(s)
        w = np.random.normal(0.,g,size=n*m)
        
    h = X.dot(w)
    p = 1/(1+np.exp(-2*h)) # kinetic
    y = np.sign(p - np.random.rand(l))

    # Scaler X
    X = MinMaxScaler().fit_transform(X)
    return X,y



