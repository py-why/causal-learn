import os
import sys
# BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
# sys.path.append(BASE_DIR)
import unittest

import numpy as np

from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd

# TODO (@aoqi): Design good test for the hypothesis testing.
# TODO (@aoqi): Design more comprehensive test cases, including:
#  every possible combinations of arguments values; design dataset of corner cases.
class TestKCI(unittest.TestCase):
    def test_Gaussian(self):
        X = np.random.randn(300, 1)
        X_prime = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd()
        pvalue, _ = kci_uind.compute_pvalue(X, X_prime)
        self.assertTrue(pvalue > 0.01) # X and X_prime are independent

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        kci_cind = KCI_CInd()
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y

    def test_Polynomial(self):
        X = np.random.randn(300, 1)
        X_prime = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd(kernelX='Polynomial', kernelY='Polynomial')
        pvalue, _ = kci_uind.compute_pvalue(X, X_prime)
        self.assertTrue(pvalue > 0.01) # X and X_prime are independent

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        kci_cind = KCI_CInd(kernelX='Polynomial', kernelY='Polynomial', kernelZ='Polynomial')
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y

    def test_Linear(self):
        X = np.random.randn(300, 1)
        X_prime = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd(kernelX='Linear', kernelY='Linear')
        pvalue, _ = kci_uind.compute_pvalue(X, X_prime)
        self.assertTrue(pvalue > 0.01) # X and X_prime are independent

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        self.assertTrue(pvalue <= 0.01) # X and Z are dependent

        kci_cind = KCI_CInd(kernelX='Linear', kernelY='Linear', kernelZ='Linear')
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        self.assertTrue(pvalue > 0.01) # X and Z are independent conditional on Y
