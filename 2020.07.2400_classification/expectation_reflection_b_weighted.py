##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from scipy.special import erf as sperf
from sklearn.linear_model import ElasticNet

#=============================================================================================================
def infer_LAD(x, y, regu=0.1,tol=1e-8, max_iter=5000):
## 2019.12.26: Jungmin's code    
    weights_limit = sperf(1e-10)*1e10
    
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    
    mu = np.zeros(x.shape[1])

    w_sol = 0.0*(np.random.rand(s_pred,s_target) - 0.5)
    b_sol = np.random.rand(1,s_target) - 0.5

#     print(weights.shape)
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones(s_sample)

        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights)
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]

#         print(cov.shape, cov_xx.shape, cov_xy.shape)
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
#             old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
#             print(w_sol[:,index].shape, npl.solve(cov_xx, cov_xy).reshape(s_pred).shape)

            # 2019.12.26: Tai - added regularization
            sigma_w = np.std(w_sol[:,index])
                
            w_eq_0 = np.abs(w_sol[:,index]) < 1e-10
            mu[w_eq_0] = 2./np.sqrt(np.pi)
        
            mu[~w_eq_0] = sigma_w*sperf(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
            
            # 2020.08.08: Vipul's suggestion                                            
            b_sol[0,index] = np.mean((y[:,index]-x.dot(w_sol[:,index]))*weights)/np.mean(weights)
                                                        
            w_sol[:,index] = np.linalg.solve(cov_xx + regu*np.diag(mu),cov_xy).reshape(s_pred)
        
            #b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            # 2020.07.25: Vipul's suggestion
            #b_sol[0,index] = np.mean((y[:,index]-x.dot(w_sol[:,index]))*weights[:,0])/np.mean(weights)
            #b_sol[0,index] = np.mean((y[:,index]-x.dot(w_sol[:,index]))*weights)/np.mean(weights)

            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(weights))
#             error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            
            #weights = sigma*sperf(weights/sigma)/weights            
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                         ddof=0, aweights=weights)
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#             print(old_error,error)
    #return b_sol,w_sol 
    return b_sol[0][0],w_sol[:,0] # for only one target case

#=============================================================================================================
# 2020.01.15: find w and h0 by ElasticNet
def fit_ElasticNet(x,y,niter_max,alpha,l1_ratio):      
    # convert 0, 1 to -1, 1
    y1 = 2*y - 1.
   
    #print(niter_max)    
    n = x.shape[1]
    
    #x_av = np.mean(x,axis=0)
    #dx = x - x_av
    #c = np.cov(dx,rowvar=False,bias=True)

    # 2019.07.16:  
    #c += l2*np.identity(n) / (2*len(y))
    #c_inv = linalg.pinvh(c)

    # initial values
    h0 = 0.
    w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
    
    cost = np.full(niter_max,100.)

    model = ElasticNet(random_state=0,alpha=alpha,l1_ratio=l1_ratio)

    for iloop in range(niter_max):
        h = h0 + x.dot(w)
        y1_model = np.tanh(h/2.)    

        # stopping criterion
        p = 1/(1+np.exp(-h))                
        cost[iloop] = ((p-y)**2).mean()

        #h_test = h0 + x_test.dot(w)
        #p_test = 1/(1+np.exp(-h_test))
        #cost[iloop] = ((p_test-y_test)**2).mean()

        if iloop>0 and cost[iloop] >= cost[iloop-1]: break

        # update local field
        t = h!=0    
        h[t] *= y1[t]/y1_model[t]
        h[~t] = 2*y1[~t]

        # find w from h    
        #h_av = h.mean()
        #dh = h - h_av 
        #dhdx = dh[:,np.newaxis]*dx[:,:]
        #dhdx_av = dhdx.mean(axis=0)
        #w = c_inv.dot(dhdx_av)
        #h0 = h_av - x_av.dot(w)

        # 2019.12.26: 
        #h0,w = infer_LAD(x,h[:,np.newaxis],l2)

        # 2020.01.15: find w and h0 by ElasticNet
        model.fit(x,h)
        w = model.coef_
        h0 = model.intercept_
        
    return h0,w
#=============================================================================================================
def fit_LAD(x,y,niter_max,l2):      
    # convert 0, 1 to -1, 1
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

        #h_test = h0 + x_test.dot(w)
        #p_test = 1/(1+np.exp(-h_test))
        #cost[iloop] = ((p_test-y_test)**2).mean()

        if iloop>0 and cost[iloop] >= cost[iloop-1]: break

        # update local field
        t = h!=0    
        h[t] *= y1[t]/y1_model[t]
        h[~t] = 2*y1[~t]

        # find w from h    
        #h_av = h.mean()
        #dh = h - h_av 
        #dhdx = dh[:,np.newaxis]*dx[:,:]
        #dhdx_av = dhdx.mean(axis=0)
        #w = c_inv.dot(dhdx_av)
        #h0 = h_av - x_av.dot(w)

        # 2019.12.26: 
        h0,w = infer_LAD(x,h[:,np.newaxis],l2)

    return h0,w

#=============================================================================================================
def fit(x,y,niter_max,l2):      
    # convert 0, 1 to -1, 1
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

        #h_test = h0 + x_test.dot(w)
        #p_test = 1/(1+np.exp(-h_test))
        #cost[iloop] = ((p_test-y_test)**2).mean()

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

#=========================================================================================    
def predict(x,h0,w):
    """ --------------------------------------------------------------------------
    calculate probability p based on x,h0, and w
    input: x[l,n], w[n], h0
    output: p[l]
    """
    h = h0 + x.dot(w)
    p = 1./(1. + np.exp(-h))        
    y = np.sign(p-0.5) # -1, 1
    y = (y+1)/2        # 0, 1
                      
    return y,p    
