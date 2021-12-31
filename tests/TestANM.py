import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import sys
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased.ANM.ANM import ANM


class TestANM(unittest.TestCase):
    def syn_data(self, b, n):
        # x = np.abs(np.random.randn(n, 1) * np.sign(np.random.randn(n, 1)))
        x = np.random.randn(n, 1)
        x = x / np.std(x)
        # e = np.abs(np.random.randn(n, 1) * np.sign(np.random.randn(n, 1)))
        e = np.random.randn(n, 1)
        e = e / np.std(e)
        y = x + b * x**3 + e
        return x, y

    # example1
    # simulated data y = x + bx^3 + e
    def test_anm_simul(self):
        anm = ANM()
        x, y = self.syn_data(1, 300)
        p_value_foward, p_value_backward = anm.cause_or_effect(x, y)
        print('pvalue for x->y is {:.4f}'.format(p_value_foward))
        print('pvalue for y->x is {:.4f}'.format(p_value_backward))

    # example2
    # data pair from the Tuebingen cause-effect pair dataset.
    def test_anm_pair(self):
        df = pd.read_csv('TestData/pair0001.txt', sep=' ', header=None)
        dataset = df.to_numpy()
        anm = ANM()
        n = dataset.shape[0]
        p_value_foward, p_value_backward = anm.cause_or_effect(dataset[:, 0].reshape(n, 1), dataset[:, 1].reshape(n, 1))
        print('pvalue for x->y is {:.4f}'.format(p_value_foward))
        print('pvalue for y->x is {:.4f}'.format(p_value_backward))


if __name__ == '__main__':
    test = TestANM()
<<<<<<< HEAD
    test.test_anm_simul()
    test.test_anm_pair()
=======
    test.test_anm_pair()
>>>>>>> 8443c497f18ad0894ca9e86790cc630631680cb0
