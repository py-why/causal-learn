import sys, os
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
from causallearn.search.Granger.Granger import Granger
import sys
import pandas as pd
import unittest
from pickle import load
import numpy as np


class TestGranger(unittest.TestCase):

    # example1
    # for data with two dimensions, granger test.
    def test_granger_test(self):
        df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
        df['month'] = df.date.dt.month
        dataset = df[['value', 'month']].to_numpy()
        G = Granger()
        p_value_matrix = G.granger_test_2d(data=dataset)
        print(p_value_matrix)

    # example2
    # for data with multi-dimensional variables, granger lasso regression.
    def test_granger_lasso(self):
        df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
        df['month'] = df.date.dt.month
        dataset = df[['value', 'month']].to_numpy()
        G = Granger()
        coeff = G.granger_lasso(data=dataset)
        print(coeff)


if __name__ == '__main__':
    test = TestGranger()
    test.test_granger_test()
    test.test_granger_lasso()