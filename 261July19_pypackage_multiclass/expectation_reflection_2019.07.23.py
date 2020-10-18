##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

class ExpectationReflection(object):

    def __init__(self, niter_max, l2):
        self.niter_max = niter_max
        self.l2 = l2

    """ ----------------------------------------------------------------------------------
    fit h0 and w based on Expectation Reflection
    input: features x[l,n], target: y[l] (y = {0,1})
     output: h0, w[n]
    """
    def fit(self,x,y):
        niter_max = self.niter_max
        l2 = self.l2
        
        n_unique_y = len(np.unique(y))
        if n_unique_y == 1:    
            print('The training data set is USELESS because it contains only 1 class')
            
        elif n_unique_y == 2:  # binary        
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
                p = 1/(1+np.exp(-h))                
                cost[iloop] = ((p-y)**2).mean()

                #cost[iloop] = ((y1[:]-y1_model[:])**2).mean()
                # 2019.07.12: lost function                
                #cost[iloop] = (-y[:]*np.log(p) - (1-y)*np.log(1-p)).mean()

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
            
            self.h0 = h0
            self.w = w
            self.classtype = 'binary'

        else:  # multiple classes
            """ -----------------------------------------------------------------------
            2019.06.14: fit h0 and w based on Expectation Reflection
            input: features x[l,n], target: y[l,m] (y = +/-1)
             output: h0[m], w[n,m]
            """
            #def fit_multi(self,x,y,niter_max=500,l2=0.001):        
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
            
            self.h0 = H0
            self.w = W
            self.classtype = 'multi'    
            #return H0,W
    #=====================================================================================                   
    def predict(self,x):
        classtype = self.classtype        
        h0 = self.h0
        w = self.w
        
        if classtype == 'binary':
            """ --------------------------------------------------------------------------
            calculate probability p based on x,h0, and w
            input: x[l,n], w[n], h0
            output: p[l]
            """
            h = h0 + x.dot(w)
            p = 1/(1+np.exp(-h))        
            y = np.sign(p-0.5) # -1, 1
            y = (y+1)/2        # 0, 1
            
        elif classtype == 'multi': 
            """ --------------------------------------------------------------------------
            2019.06.12: calculate probability p based on x,h0, and w
            input: x[l,n], w[n,my], h0
            output: p[l]
            """
            h = h0[np.newaxis,:] + x.dot(w)
            p = 1/(1+np.exp(-h))
            p /= p.sum(axis=1)[:,np.newaxis]
            y = np.argmax(p,axis=1)
        else:
            print('Cannot define the classtype, not binary nor multi')
                          
        return y
