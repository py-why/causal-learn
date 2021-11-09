import os, sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LassoCV


class Granger(object):
    '''
    Python implementation of granger causality, including 1) regression+hypothesis test and 2) lass regression
    '''
    def __init__(self, maxlag=2, test='ssr_ftest', cv=5):
        '''
        Construct the Granger model.

        Parameters
        ----------
        maxlag: maximum time lag
        test: statistical test (details can be found in https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
        cv: cross validation for lasso regression
        '''
        self.maxlag = maxlag
        self.test = test
        self.cv = cv

    def granger_test_2d(self, data):
        '''
        Granger causality test for two-dimensional time series

        Parameters:
        -----------
        data - input data (nxd)

        Returns:
        ----------
        p_value_matrix: p values for x1->x2 and x2->x1
        '''
        n, dim = data.shape
        assert dim==2, "Data have more than two dimensions"
        p_value_matrix = np.zeros((dim, dim))
        for c in range(dim):
            for r in range(dim):
                test_result = grangercausalitytests(data[:, [r, c]], maxlag=self.maxlag, verbose=False)
                p_values = [round(test_result[i + 1][0][self.test][1], 4) for i in range(self.maxlag)]
                min_p_value = np.min(p_values)
                p_value_matrix[r, c] = min_p_value
        return p_value_matrix

    def granger_lasso(self, data):
        '''
        Granger causality test for multi-dimensional time series

        Parameters:
        -----------
        data - input data (nxd)

        Returns:
        ----------
        coeff: coefficient matrix
        '''
        n, dim = data.shape
        # stack data to form one-vs-all regression
        Y = data[self.maxlag:]
        X = np.hstack([data[self.maxlag-k:-k] for k in range(1, self.maxlag+1)])

        lasso_cv = LassoCV(cv=self.cv)
        coeff = np.zeros((dim, dim*self.maxlag))
        # Consider one variable after the other as target
        for i in range(dim):
            lasso_cv.fit(X, Y[:,i])
            coeff[i] = lasso_cv.coef_
        return coeff