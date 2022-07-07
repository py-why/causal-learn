import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import sys
import unittest
from pickle import load

import numpy as np

from causallearn.search.FCMBased.ANM.ANM import ANM


class TestANM(unittest.TestCase):

    # Test ANM by some simulated data
    def test_anm_simulation(self):

        anm = ANM()

        # generate cause and noises
        num_samples = 1000
        x = np.random.randn(num_samples, 1)
        x = x / np.std(x)
        e = np.random.randn(num_samples, 1)
        e = e / np.std(e)

        # simulated data y = x + 2x^3 + e
        y_1 = x + 2 * x**3 + e
        p_value_1, _ = anm.cause_or_effect(x, y_1)
        self.assertTrue(p_value_1 > 0.05)

        # simulated data y = 3*exp(x) + e
        y_2 = 3 * np.exp(x) + e
        p_value_2, _ = anm.cause_or_effect(x, y_2)
        self.assertTrue(p_value_2 > 0.05)

        # simulated data y = 2 * Sigmoid(x) + e
        y_3= 2/(1+np.exp(-x)) + e
        p_value_3, _ = anm.cause_or_effect(x, y_3)
        self.assertTrue(p_value_3 > 0.05)

    # data pair from the Tuebingen cause-effect pair dataset.
    def test_anm_pair(self):

        dataset = np.loadtxt('TestData/pair0001.txt')
        anm = ANM()
        p_value_forward, _ = anm.cause_or_effect(dataset[:, 0].reshape(-1, 1), dataset[:, 1].reshape(-1, 1))
        self.assertTrue(p_value_forward > 0.05)


if __name__ == '__main__':
    test = TestANM()
    test.test_anm_simulation()
    test.test_anm_pair()
