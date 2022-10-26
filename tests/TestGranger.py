import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import sys
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.Granger.Granger import Granger

######################################### Test Notes ###########################################
# All the benchmark results (p_value_matrix_truth, adj_matrix_truth, coeff_truth)              #
# are obtained from the code of causal-learn as of commit                                      #
# https://github.com/cmu-phil/causal-learn/commit/b49980d046607baaaa66ff8dc0ceb98452ab8616     #
# (b49980d).                                                                                   #
#                                                                                              #
# We are not sure if the results are completely "correct" (reflect ground truth graph) or not. #
# So if you find your tests failed, it means that your modified code is logically inconsistent #
# with the code as of b49980d, but not necessarily means that your code is "wrong".            #
# If you are sure that your modification is "correct" (e.g. fixed some bugs in b49980d),       #
# please report it to us. We will then modify these benchmark results accordingly. Thanks :)   #
######################################### Test Notes ###########################################


class TestGranger(unittest.TestCase):
    # simulate data from a VAR model
    def syn_data_3d(self):
        # generate transition matrix, time lag 2
        np.random.seed(0)
        A = 0.2 * np.random.rand(3,6)
        # generate time series
        T = 1000
        data = np.random.rand(3, T)
        data[:,2:] = 0
        for i in range(2,T):
            data[:,i] = A[:,0:3].dot(data[:,i-1]) + A[:,3:6].dot(data[:,i-2]) + 0.1 * np.random.randn(3)

        return data.T

    def syn_data_2d(self):
        # generate transition matrix, time lag 2
        np.random.seed(3)
        A = 0.5*np.random.rand(2,4)
        A[0,1] = 0
        A[0,3] = 0
        # generate time series
        T = 100
        data = np.random.rand(2, T)
        data[:,2:] = 0
        for i in range(2,T):
            data[:,i] = A[:,0:2].dot(data[:,i-1]) + A[:,2:4].dot(data[:,i-2]) + 0.1 * np.random.randn(2)

        return data.T

    # example1
    # for data with two dimensions, granger test.
    def test_granger_test(self):
        # df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
        # df['month'] = df.date.dt.month
        # dataset = df[['value', 'month']].to_numpy()
        dataset = self.syn_data_2d()
        G = Granger()
        p_value_matrix, adj_matrix = G.granger_test_2d(data=dataset)
        p_value_matrix_truth = np.array([[0, 0.5989, 0, 0.5397], [0.0006, 0, 0.0014, 0]])
        adj_matrix_truth = np.array([[1, 0, 1, 0], [1, 1, 1, 1]])
        self.assertEqual((np.round(p_value_matrix, 4) - p_value_matrix_truth).all(), 0)
        self.assertEqual((adj_matrix - adj_matrix_truth).all(), 0)

    # example2
    # for data with multi-dimensional variables, granger lasso regression.
    def test_granger_lasso(self):
        # df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
        # df['month'] = df.date.dt.month
        # dataset = df[['value', 'month']].to_numpy()
        dataset = self.syn_data_3d()
        G = Granger()
        coeff = G.granger_lasso(data=dataset)
        coeff_truth = np.array([[0.09, 0.1101, 0.1527, 0.1127, 0.0226, 0.1538],
                                [0.1004, 0.15, 0.1757, 0.1037, 0.1612, 0.0987],
                                [0.1155, 0.1485, 0, 0.039, -0., 0.1085]])
        self.assertEqual((np.round(coeff, 4) - coeff_truth).all(), 0)

