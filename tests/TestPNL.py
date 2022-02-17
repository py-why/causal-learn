import os
import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
# # sys.path.append(BASE_DIR)
from causallearn.search.FCMBased.PNL.PNL import PNL


class TestPNL(unittest.TestCase):
    def syn_data(self, b, n):
        # x = np.abs(np.random.randn(n, 1) * np.sign(np.random.randn(n, 1)))
        x = np.random.randn(n, 1)
        x = x / np.std(x)
        # e = np.abs(np.random.randn(n, 1) * np.sign(np.random.randn(n, 1)))
        e = np.random.randn(n, 1)
        e = e / np.std(e)
        y = x + 0 * x ** 3 + e
        plt.plot(x, y, '.')
        plt.show()
        return x, y

    # example1
    # simulated data y = x + bx^3 + e
    def test_pnl_simul(self):
        pnl = PNL()
        x, y = self.syn_data(1, 1000)
        p_value_foward, p_value_backward = pnl.cause_or_effect(x, y)
        print('pvalue for x->y is {:.4f}'.format(p_value_foward))
        print('pvalue for y->x is {:.4f}'.format(p_value_backward))

    # example2
    # data pair from the Tuebingen cause-effect pair dataset.
    def test_pnl_pair(self):
        df = pd.read_csv('TestData/pair0001.txt', sep=' ', header=None)
        dataset = df.to_numpy()
        pnl = PNL()
        n = dataset.shape[0]
        p_value_foward, p_value_backward = pnl.cause_or_effect(dataset[:, 0].reshape(n, 1), dataset[:, 1].reshape(n, 1))
        print('pvalue for x->y is {:.4f}'.format(p_value_foward))
        print('pvalue for y->x is {:.4f}'.format(p_value_backward))


if __name__ == '__main__':
    test = TestPNL()
    test.test_pnl_simul()
    # test.test_pnl_pair()
    test.test_pnl_pair()
