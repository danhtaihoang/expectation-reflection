##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

def fit(x,y,x_test,y_test,niter_max,l2):       
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))  # y[t,:] = {0,1,...,0}

    #print(niter_max)        
    n = x.shape[1]
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    # 2019.07.16:  l2 = lamda/(2L)
    c += l2*np.identity(n) / (2*len(y))
    c_inv = linalg.pinvh(c)

    H0 = np.zeros(m)
    W = np.zeros((n,m))

    for i in range(m):
        y = y_onehot[:,i]  # y = {0,1}
        y1 = 2*y - 1       # y1 = {-1,1}
        # initial values
        h0 = 0.
        w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
        
        cost = np.full(niter_max,100.)
        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    

            # stopping criterion
            p = 1/(1+np.exp(-h))

            cost[iloop] = ((p-y)**2).mean()

            # 2019.07.12: lost function
            #cost[iloop] = ((y1[:]-y1_model[:])**2).mean()
            #cost[iloop] = (-y[:]*np.log(p) - (1-y)*np.log(1-p)).mean()

            h_test = h0 + x_test.dot(w)
            p_test = 1/(1+np.exp(-h_test))
            cost[iloop] = ((p_test-y_test)**2).mean()

            if iloop > 0 and cost[iloop] >= cost[iloop-1] : break
                        
            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]

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

#=========================================================================================    
def predict(x,h0,w):
    h = h0[np.newaxis,:] + x.dot(w)
    p = 1/(1+np.exp(-h))
    p /= p.sum(axis=1)[:,np.newaxis]
    y = np.argmax(p,axis=1)
                      
    return y    
