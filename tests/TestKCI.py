import os
import sys
# BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
# sys.path.append(BASE_DIR)
import unittest

import numpy as np

import causallearn.utils.cit as cit
from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd


class TestKCI(unittest.TestCase):
    def test_Gaussian(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd()
        pvalue, _ = kci_uind.compute_pvalue(X, X1)
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        kci_cind = KCI_CInd()
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))

    def test_Polynomial(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd(kernelX='Polynomial', kernelY='Polynomial')
        pvalue, _ = kci_uind.compute_pvalue(X, X1)
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        kci_cind = KCI_CInd(kernelX='Polynomial', kernelY='Polynomial', kernelZ='Polynomial')
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))

    def test_Linear(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        kci_uind = KCI_UInd(kernelX='Linear', kernelY='Linear')
        pvalue, _ = kci_uind.compute_pvalue(X, X1)
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue, _ = kci_uind.compute_pvalue(X, Z)
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        kci_cind = KCI_CInd(kernelX='Linear', kernelY='Linear', kernelZ='Linear')
        pvalue, _ = kci_cind.compute_pvalue(X, Z, Y)
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))


class TestCIT_KCI(unittest.TestCase):
    def test_Gaussian(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        pvalue = cit.kci_ui(X, X1)
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ui(X, Z)
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ci(X, Z, Y)
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))

    def test_Polynomial(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        pvalue = cit.kci_ui(X, X1, kernelX='Polynomial', kernelY='Polynomial')
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ui(X, Z, kernelX='Polynomial', kernelY='Polynomial')
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ci(X, Z, Y, kernelX='Polynomial', kernelY='Polynomial', kernelZ='Polynomial')
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))

    def test_Linear(self):
        X = np.random.randn(300, 1)
        X1 = np.random.randn(300, 1)
        Y = np.concatenate((X, X), axis=1) + 0.5 * np.random.randn(300, 2)
        Z = Y + 0.5 * np.random.randn(300, 2)

        pvalue = cit.kci_ui(X, X1, kernelX='Linear', kernelY='Linear')
        print('X and X1 are independent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ui(X, Z, kernelX='Linear', kernelY='Linear')
        print('X and Z are dependent, pvalue is {:.2f}'.format(pvalue))

        pvalue = cit.kci_ci(X, Z, Y, kernelX='Linear', kernelY='Linear', kernelZ='Linear')
        print('X and Z are independent conditional on Y, pvalue is {:.2f}'.format(pvalue))


if __name__ == '__main__':
    test = TestKCI()
    print('------------------------------')
    print('Test KCI with Gaussian kernel')
    test.test_Gaussian()
    print('------------------------------')
    print('Test KCI with Polynomial kernel')
    test.test_Polynomial()
    print('------------------------------')
    print('Test KCI with Linear kernel')
    test.test_Linear()

    test = TestCIT_KCI()
    print('------------------------------')
    print('Test CIT_KCI with Gaussian kernel')
    test.test_Gaussian()
    print('------------------------------')
    print('Test CIT_KCI with Polynomial kernel')
    test.test_Polynomial()
    print('------------------------------')
    print('Test CIT_KCI with Linear kernel')
    test.test_Linear()
