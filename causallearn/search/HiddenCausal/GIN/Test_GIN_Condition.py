import numpy as np
import pandas as pd
import causallearn.search.HiddenCausal.GIN.FisherTest as FisherTest
import causallearn.search.HiddenCausal.GIN.Mutual_Information as ID



def GIN(X, Z, data, test_function, alpha = 0.05):
    '''
    Generalized Independent Noise Condition Test
    Parameters
    ----------
    data : numpy ndarray
           data set
    X, Z : list

    test_function : str, default='kci'
        the name of the independence test being used
    alpha : float, default=0.05
        desired significance level of independence tests (p_value) in (0,1)
    Returns
    -------
    Boolean : True or False
    '''
    omega = getomega(data, X, Z)
    tdata = data[:,X]
    result = np.dot(tdata, omega)[:,None]
    for i in Z:
        temp = np.array(data[:, [i]])
        pval = test_function(result, temp)
        if pval > alpha:
            flag = True
        else:
            flag = False

        if not flag:
            return False

    return True


def GIN_MI(X, Z, data):
    """Method : Calculate mutual information by k nearest neighbors (density estimation) """
    omega = getomega(data, X, Z)
    tdata = data[:, X]
    result = np.dot(tdata, omega)
    MIS = 0
    for i in Z:
        temp = np.array(data[:, i])
        mi = ID.MI_1(result, temp)
        MIS += mi
    MIS = MIS/len(Z)

    return MIS


def FisherGIN(X, Z, data, test_function, alpha = 0.01):
    """Test GIN Condition by fisher method """
    omega = getomega(data, X, Z)
    tdata = data[:, X]
    result = np.dot(tdata, omega)[:,None]
    pvals = []

    for i in Z:
        temp = np.array(data[:, [i]])
        pval = test_function(result, temp)
        pvals.append(pval)

    flag, fisher_pval = FisherTest.FisherTest(pvals, alpha)

    return flag

def getomega(data, X, Z):
    cov = np.cov(data, rowvar=False)
    cov_m = cov[np.ix_(Z, X)]
    _, _, v = np.linalg.svd(cov_m)
    omega = v.T[:, -1]
    return omega



