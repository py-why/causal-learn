import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from causallearn.utils.KCI.KCI import KCI_UInd


class ANM(object):
    '''
    Python implementation of additive noise model-based causal discovery.
    References
    ----------
    [1] Hoyer, Patrik O., et al. "Nonlinear causal discovery with additive noise models." NIPS. Vol. 21. 2008.
    '''

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        '''
        Construct the ANM model.

        Parameters:
        ----------
        kernelX: kernel function for hypothetical cause
        kernelY: kernel function for estimated noise
        '''
        self.kernelX = kernelX
        self.kernelY = kernelY

    def fit_gp(self, X, y):
        '''
        Fit a Gaussian process regression model

        Parameters
        ----------
        X: input data (nx1)
        y: output data (nx1)

        Returns
        --------
        pred_y: predicted output (nx1)
        '''
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
        gpr = GaussianProcessRegressor(kernel=kernel)

        # fit Gaussian process, including hyperparameter optimization
        gpr.fit(X, y)
        pred_y = gpr.predict(X)
        return pred_y

    def cause_or_effect(self, data_x, data_y):
        '''
        Fit a GP model in two directions and test the independence between the input and estimated noise

        Parameters
        ---------
        data_x: input data (nx1)
        data_y: output data (nx1)

        Returns
        ---------
        pval_forward: p value in the x->y direction
        pval_backward: p value in the y->x direction
        '''

        # set up unconditional test
        kci = KCI_UInd(self.kernelX, self.kernelY)

        # test x->y
        pred_y = self.fit_gp(data_x, data_y)
        res_y = data_y - pred_y
        pval_foward, _ = kci.compute_pvalue(data_x, res_y)

        # test y->x
        pred_x = self.fit_gp(data_y, data_x)
        res_x = data_x - pred_x
        pval_backward, _ = kci.compute_pvalue(data_y, res_x)

        return pval_foward, pval_backward
