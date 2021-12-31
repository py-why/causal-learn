import numpy as np
from sklearn.linear_model import LassoCV
from statsmodels.tsa.stattools import grangercausalitytests


class Granger(object):
    '''
    Python implementation of granger causality, including 1) regression+hypothesis test and 2) lasso regression
    '''

    def __init__(self, maxlag=2, test='ssr_ftest', significance_level=0.01, cv=5):
        '''
        Construct the Granger model.

        Parameters
        ----------
        maxlag: maximum time lag
        test: statistical test (details can be found in https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
        cv: number of cross validation folds for lasso regression
        '''
        self.maxlag = maxlag
        self.test = test
        self.significance_level = significance_level
        self.cv = cv

    def granger_test_2d(self, data):
        '''
        Granger causality test for two-dimensional time series

        Parameters:
        -----------
        data - input data (nxd)

        Returns:
        ----------
        p_value_matrix: [P_1,P_2,...], where P_k is the p value matrix for the k-th time lag. The ij-th element in P_k
        is the pvalue of influence from variable j to variable i. Note: a small p value means significant relation.

        adj_matrix: [A_1,A_2,...], where A_k is the adjacency matrix for the k-th time lag. The ij-th element in A_k
        is the influence from variable j to variable i.
        '''
        n, dim = data.shape
        assert dim == 2, "Data have more than two dimensions"
        p_value_matrix = np.zeros((dim, dim*self.maxlag))
        for c in range(dim):
            for r in range(dim):
                if r != c:
                    test_result = grangercausalitytests(data[:, [r, c]], maxlag=self.maxlag, verbose=False)
                    for i in range(self.maxlag):
                        p_value_matrix[r, c+dim*i] = round(test_result[i + 1][0][self.test][1], 4)
        adj_matrix = p_value_matrix < self.significance_level
        return p_value_matrix, adj_matrix*1

    def granger_lasso(self, data):
        '''
        Granger causality test for multi-dimensional time series

        Parameters:
        -----------
        data - input data (nxd)

        Returns:
        ----------
        coeff: coefficient matrix [A_1, A_2, ..], where A_k is the dxd causal matrix for the k-th time lag. The ij-th entry
        in A_k represents the causal influence from j-th variable to the i-th variable.
        '''
        n, dim = data.shape
        # stack data to form one-vs-all regression
        Y = data[self.maxlag:]
        X = np.hstack([data[self.maxlag - k:-k] for k in range(1, self.maxlag + 1)])

        lasso_cv = LassoCV(cv=self.cv)
        coeff = np.zeros((dim, dim * self.maxlag))
        # Consider one variable after the other as target
        for i in range(dim):
            lasso_cv.fit(X, Y[:, i])
            coeff[i] = lasso_cv.coef_
        return coeff
