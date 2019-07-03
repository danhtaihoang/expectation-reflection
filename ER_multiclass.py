##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
#=========================================================================================

""" --------------------------------------------------------------------------------------
2019.06.14: fit h0 and w based on Expectation Reflection
input: features x[l,n], target: y[l,m] (y = +/-1)
 output: h0[m], w[n,m]
"""
def fit(x,y,niter_max=100,regu=0.0):
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))

    y_onehot = 2*y_onehot-1

    #print(niter_max)        
    n = x.shape[1]
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c += regu*np.identity(n)
    c_inv = linalg.pinvh(c)

    H0 = np.zeros(m)
    W = np.zeros((n,m))

    for i in range(m):
        y = y_onehot[:,i]
        # initial values
        h0 = 0.
        w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
        
        cost = np.full(niter_max,100.)
        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            y_model = np.tanh(h)    

            # stopping criterion
            cost[iloop] = ((y[:]-y_model[:])**2).mean()
            if iloop>0 and cost[iloop] >= cost[iloop-1]: break

            # update local field
            t = h!=0    
            h[t] *= y[t]/y_model[t]
            h[~t] = y[~t]

            # find w from h    
            h_av = h.mean()
            dh = h - h_av 
            dhdx = dh[:,np.newaxis]*dx[:,:]

            dhdx_av = dhdx.mean(axis=0)
            w = c_inv.dot(dhdx_av)
            h0 = h_av - x_av.dot(w)

        H0[i] = h0
        W[:,i] = w
    return H0,W

""" --------------------------------------------------------------------------------------
2019.06.12: calculate probability p based on x,h0, and w
input: x[l,n], w[n,my], h0
output: p[l]
"""
def predict(x,h0,w):
    h = h0[np.newaxis,:] + x.dot(w)
    p = np.exp(h)
        
    # normalize
    p_sum = p.sum(axis=1)       
    p /= p_sum[:,np.newaxis]  

    return np.argmax(p,axis=1)
