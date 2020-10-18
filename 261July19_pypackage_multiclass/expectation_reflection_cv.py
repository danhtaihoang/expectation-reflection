##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

def fit(x,y,x_test,y_test,niter_max,l2):      
    # convert 0,1 to -1, 1
    y1 = 2*y - 1.
   
    #print(niter_max)    
    n = x.shape[1]
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    # 2019.07.16:  
    c += l2*np.identity(n) / (2*len(y))
    c_inv = linalg.pinvh(c)

    # initial values
    h0 = 0.
    w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
    
    cost = np.full(niter_max,100.)
    for iloop in range(niter_max):
        h = h0 + x.dot(w)
        y1_model = np.tanh(h/2.)    

        # stopping criterion
        #p = 1/(1+np.exp(-h))                
        #cost[iloop] = ((p-y)**2).mean()

        h_test = h0 + x_test.dot(w)
        p_test = 1/(1+np.exp(-h_test))
        cost[iloop] = ((p_test-y_test)**2).mean()

        if iloop>0 and cost[iloop] >= cost[iloop-1]: break

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

    return h0,w
    
def predict(x,h0,w):
    """ --------------------------------------------------------------------------
    calculate probability p based on x,h0, and w
    input: x[l,n], w[n], h0
    output: p[l]
    """
    h = h0 + x.dot(w)
    p = 1/(1+np.exp(-h))        
    y = np.sign(p-0.5) # -1, 1
    y = (y+1)/2        # 0, 1
                      
    return y,p    
