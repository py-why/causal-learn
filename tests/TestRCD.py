import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam


class TestRCD(unittest.TestCase):

    def test_RCD(self):
        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

        model = lingam.RCD()
        model.fit(X)

        print(model.adjacency_matrix_)
