import unittest

import numpy as np

import causallearn.utils.cit as cit

class TestCIT_KCI(unittest.TestCase):
    def test_Gaussian(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        cit_CIT = cit.CIT(data=np.hstack((X,X1,Y,Z)), method='kci')

        pvalue = cit_CIT.kci(0, 1, {})
        self.assertTrue(pvalue > 0.01) # X and X1 are independent

        pvalue = cit_CIT.kci(0, 3, {})
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        pvalue = cit_CIT.kci(0, 3, {2})
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y

    def test_Polynomial(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        cit_CIT = cit.CIT(data=np.hstack((X,X1,Y,Z)), method='kci', kernelX='Polynomial', kernelY='Polynomial', kernelZ='Polynomial')

        pvalue = cit_CIT.kci(0, 1, {})
        self.assertTrue(pvalue > 0.01) # X and X1 are independent

        pvalue = cit_CIT.kci(0, 3, {})
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        pvalue = cit_CIT.kci(0, 3, {2})
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y

    def test_Linear(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        cit_CIT = cit.CIT(data=np.hstack((X,X1,Y,Z)), method='kci', kernelX='Linear', kernelY='Linear', kernelZ='Linear')

        pvalue = cit_CIT.kci(0, 1, {})
        self.assertTrue(pvalue > 0.01) # X and X1 are independent

        pvalue = cit_CIT.kci(0, 3, {})
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        pvalue = cit_CIT.kci(0, 3, {2})
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y

if __name__ == '__main__':
    unittest.main()
