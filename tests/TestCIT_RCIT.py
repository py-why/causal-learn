import unittest

import numpy as np

import causallearn.utils.cit as cit


class TestCIT_RCIT(unittest.TestCase):
    def test_Gaussian_dist(self):
        np.random.seed(10)
        X = np.random.randn(300, 1)
        X_prime = np.random.randn(300, 1)
        Y = X + 0.5 * np.random.randn(300, 1)
        Z = Y + 0.5 * np.random.randn(300, 1)
        data = np.hstack((X, X_prime, Y, Z))

        pvalue01 = []
        pvalue03 = []
        pvalue032 = []
        for approx in ["lpd4", "hbe", "gamma", "chi2", "perm"]:
            for num_f in [50, 100]:
                for num_f2 in [5, 10]:
                    for rcit in [True, False]:
                        cit_CIT = cit.CIT(data, 'rcit', approx=approx, num_f=num_f,
                                          num_f2=num_f2, rcit=rcit)
                        pvalue01.append(round(cit_CIT(0, 1), 4))
                        pvalue03.append(round(cit_CIT(0, 3), 4))
                        pvalue032.append(round(cit_CIT(0, 3, {2}), 4))

        pvalue01 = np.array(pvalue01)
        pvalue03 = np.array(pvalue03)
        pvalue032 = np.array(pvalue032)
        self.assertTrue(np.all((0.0 <= pvalue01) & (pvalue01 <= 1.0)),
                        "pvalue01 contains invalid values")
        self.assertTrue(np.all((0.0 <= pvalue03) & (pvalue03 <= 1.0)),
                        "pvalue03 contains invalid values")
        self.assertTrue(np.all((0.0 <= pvalue032) & (pvalue032 <= 1.0)),
                        "pvalue032 contains invalid values")
