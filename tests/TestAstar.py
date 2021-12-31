import sys

from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

sys.path.append("")
import unittest
from pickle import load

import numpy as np


class TestAstar(unittest.TestCase):

    # example1
    # for data with single-variate dimensions, astar.
    def test_single_astar(self):
        with open("example_data1.pk", 'rb') as example_data1:
            # example_data1 = load(open("example_data1.pk", 'rb'))
            example_data1 = load(example_data1)
            X = example_data1['X']
            X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
            X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
            X = X[:50, :]
            dag_est, search_stats = bic_exact_search(X, search_method='astar')
            print(dag_est)
            print(search_stats)

    # example2
    # for data with multi-variate dimensions, astar.
    def test_multi_astar(self):
        with open("example_data2.pk", 'rb') as example_data:
            # example_data = load(open("example_data2.pk", 'rb'))
            example_data = load(example_data)
            Data_save = example_data['Data_save']
            trial = 0
            X = Data_save[trial]
            X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
            X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
            X = X[:50, :]
            dag_est, search_stats = bic_exact_search(X, search_method='astar')
            print(dag_est)
            print(search_stats)
