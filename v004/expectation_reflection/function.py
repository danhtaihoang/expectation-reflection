##========================================================================================
import numpy as np
from sklearn.utils import shuffle

#=========================================================================================
def make_data_balance(X,y):
# make data balance and shuffle    
    y_unique,ny = np.unique(y,return_counts=True)
    i0 = np.argmin(ny)
    l0 = ny[i0]

    t0 = y == y_unique[i0]
    X_balance,y_balance = X[t0],y[t0]

    for i in range(len(y_unique)):
        if i != i0:
            ti = y == y_unique[i]
            Xi,yi = X[ti],y[ti]

            ti = np.random.choice(len(yi),size=l0,replace=False)
            Xi,yi = Xi[ti],yi[ti]

            X_balance = np.vstack([X_balance,Xi])
            y_balance = np.hstack([y_balance,yi])
      
    return shuffle(X_balance, y_balance)

#=========================================================================================
def split_train_test(X,y,train_size,test_size):
    ## split train, test with the same fraction in each class (from y).
    X, y = shuffle(X, y)
    
    y_unique,ny = np.unique(y,return_counts=True)

    for i in range(len(y_unique)):
        t = y == y_unique[i]

        Xi,yi = X[t],y[t]

        # test
        t1 = np.random.choice(len(yi),size=int(test_size*len(yi)),replace=False)
        # train
        t2 = np.random.choice(np.delete(np.arange(len(yi)),t1),size=int(train_size*len(yi)),replace=False)

        X1,y1 = Xi[t1],yi[t1]
        X2,y2 = Xi[t2],yi[t2]

        if i == 0:
            X_test,y_test = X1,y1
            X_train,y_train = X2,y2
        else:
            X_test = np.vstack([X_test,X1])
            y_test = np.hstack([y_test,y1])
            X_train = np.vstack([X_train,X2])
            y_train = np.hstack([y_train,y2])

    X_train, y_train = shuffle(X_train, y_train)            
    X_test, y_test = shuffle(X_test, y_test)

    return X_train,X_test,y_train,y_test
