import numpy as np
import pandas as pd
import causallearn.search.HiddenCausal.GIN.FisherTest as FisherTest
import causallearn.search.HiddenCausal.GIN.independence as ID

#GIN by 
#X=['X1','X2']
#Z=['X3']
#data.type=Pandas.DataFrame
def GIN(X, Z, data, test_function, alpha=0.05):
    omega = getomega(data,X,Z)
    tdata= data[:, X]
    #print(tdata.T)
    result = np.dot(tdata, omega)[:,None]
    for i in Z:
        temp = np.array(data[:, [i]])
        pval =test_function(result, temp)
        if pval > alpha:
            flag = True
        else:
            flag = False

        if not flag:#not false == ture  ---> if false
            #print(X,Z,flag)
            return False

    return True



#GIN by Fisher's method
def FisherGIN(X,Z,data,test_function,alpha=0.01):
    omega = getomega(data,X,Z)
    tdata= data[:,X]
    result = np.dot(tdata, omega)[:,None]
    pvals=[]

    for i in Z:
        temp = np.array(data[:, [i]])
        pval=test_function(result, temp)
        pvals.append(pval)
    flag,fisher_pval=FisherTest.FisherTest(pvals,alpha)

    return flag






#mthod 1: estimating mutual information by k nearest neighbors (density estimation)
#mthod 2: estimating mutual information by sklearn package
def GIN_MI(X,Z,data,method='1'):
    omega = getomega(data,X,Z)
    tdata= data[:, X]
    result = np.dot(tdata, omega)
    MIS=0
    for i in Z:

        temp = np.array(data[:, i])
        if method =='1':
            mi=ID.independent(result,temp)
        else:
            mi=ID.independent11(result,temp)
        MIS+=mi
    MIS = MIS/len(Z)

    return MIS





def getomega(data,X,Z):
    cov_m =np.cov(data,rowvar=False)
    # print(f'{cov_m.shape = }')
    col = list(range(data.shape[1]))
    Xlist = []
    Zlist = []
    for i in X:
        t = col.index(i)
        Xlist.append(t)
    for i in Z:
        t = col.index(i)
        Zlist.append(t)
    B = cov_m[Xlist]
    B = B[:,Zlist]
    A = B.T
    u,s,v = np.linalg.svd(A)
    lens = len(X)
    omega =v.T[:,lens-1]

    return omega

