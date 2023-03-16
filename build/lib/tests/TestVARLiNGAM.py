import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam


class TestVARLiNGAM(unittest.TestCase):

    def test_DirectLiNGAM(self):
        X = pd.read_csv('sample_data_var_lingam.csv')
        model = lingam.VARLiNGAM()
        model.fit(X)

        print(model.causal_order_)
        print(model.adjacency_matrices_[0])
        print(model.adjacency_matrices_[1])
        print(model.residuals_)
