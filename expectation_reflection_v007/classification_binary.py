import numpy as np
from scipy import linalg
from scipy.special import erf as sperf
from sklearn.preprocessing import LabelBinarizer
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.linear_model import ElasticNet


class model(object):
    def __init__(self,max_iter=100,regu=0.01,random_state=1):

        self.max_iter = max_iter
        self.regu = regu
        self.random_state = random_state

        #print('ini max_iter, regu, random_state:',max_iter,regu,random_state)

    ##======================================================================
    def fit(self,x,y):
        max_iter = self.max_iter
        regu = self.regu
        random_state = self.random_state

        #print('fit max_iter, regu, random_state:',max_iter,regu,random_state)
        np.random.seed(random_state)

        # 2020.10.30:
        label_binarizer = LabelBinarizer().fit(y)
        y = label_binarizer.transform(y).reshape(-1,) # y = [1,0,1,...,0] 

        # convert y{0, 1} to y1{-1, 1}
        y1 = 2*y - 1.
        #print(niter_max)
        n_features = x.shape[1]

        b = 0.
        w = np.random.normal(0.0,1./np.sqrt(n_features),size=(n_features))
        cost = np.full(max_iter,100.)
        for iloop in range(max_iter):
            h = b + x.dot(w)
            y1_model = np.tanh(h/2.)    
            # stopping criterion
            p = 1/(1+np.exp(-h))
            cost[iloop] = ((p-y)**2).mean()
            if iloop>0 and cost[iloop] >= cost[iloop-1]: break
            # update local field
            t = h!=0
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]
            # 2019.12.26:
            b,w = infer_LAD(x,h[:,np.newaxis],regu)


        self.b = b
        self.w = w
        # just for output in the main script
        self.intercept_ = b
        self.coef_ = w

        #self.onehot_encoder = onehot_encoder
        #print('b:',b)
        self.label_binarizer = label_binarizer
            
        return self

    ##=====================================================================
    def predict(self,x):
        """calculate probability p based on x,b, and w
        input: x[l,n], w[n], b
        output: p[l]
        """
        b = self.b
        w = self.w

        h = b + x.dot(w)
        p = 1./(1. + np.exp(-h))
        y = np.sign(p-0.5) # {-1, 1}
        y = (y+1)/2        # {0, 1}

        # 2020.10.30: y_inv
        label_binarizer = self.label_binarizer
        y_inv = label_binarizer.inverse_transform(y)

        return y_inv

    ##=====================================================================
    def predict_proba(self,x):
        """
        calculate probability p based on x,b, and w
        input: x[l,n], w[n], b
        output: p[l]
        """
        b = self.b
        w = self.w
        
        h = b + x.dot(w)
        p = 1./(1. + np.exp(-h))        

        return p

    ##======================================================================
    ## 2020.09.28    
    def get_params(self,deep = True):
        return {"max_iter":self.max_iter, "regu":self.regu,\
         "random_state":self.random_state}

    def set_params(self,**parameters):
        for parameter,value in parameters.items():
            setattr(self,parameter,value)
        return self

    def score(self,X,y):
        # 2020.10.30:
        label_binarizer = self.label_binarizer
        y = label_binarizer.transform(y).reshape(-1,) # y = [1,0,1,...,0] 

        p_pred = self.predict_proba(X)
        cost = ((p_pred - y)**2).mean()
        return (1-cost) # larger is better

##===========================================================================
def infer_LAD(x,y,regu,tol=1e-8,max_iter=5000):   
    weights_limit = sperf(1e-10)*1e10
    
    n_sample, n_var = x.shape
    n_target = y.shape[1]
    
    mu = np.zeros(x.shape[1])

    w_sol = 0.0*(np.random.rand(n_var,n_target) - 0.5)
    b_sol = np.random.rand(1,n_target) - 0.5

    for index in range(n_target):
        error, old_error = np.inf, 0
        weights = np.ones(n_sample)

        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights)
        cov_xx, cov_xy = cov[:n_var,:n_var],cov[:n_var,n_var:(n_var+1)]

        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))

            # 2019.12.26: Tai - added regularization
            sigma_w = np.std(w_sol[:,index])
                
            w_eq_0 = np.abs(w_sol[:,index]) < 1e-10
            mu[w_eq_0] = 2./np.sqrt(np.pi)
        
            mu[~w_eq_0] = sigma_w*sperf(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
                                                        
            w_sol[:,index] = np.linalg.solve(cov_xx + regu*np.diag(mu),cov_xy).reshape(n_var)
        
            b_sol[0,index] = np.mean((y[:,index]-x.dot(w_sol[:,index]))*weights)/np.mean(weights)

            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(weights))

            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
                      
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                         ddof=0, aweights=weights)
            cov_xx, cov_xy = cov[:n_var,:n_var],cov[:n_var,n_var:(n_var+1)]
#             print(old_error,error)

    return b_sol[0][0],w_sol[:,0] # for only one target case

