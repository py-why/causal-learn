import unittest

import numpy as np

from causallearn.utils.cit import CIT


class TestCIT(unittest.TestCase):
    def test_fisherz_singularity_problem(self):
        X1 = X2 = np.random.normal(size=1000)
        X = np.array([X1, X2]).T

        cit = CIT(data=X, method='fisherz')

        with self.assertRaises(ValueError) as context:
            cit.fisherz(0, 1, tuple())

