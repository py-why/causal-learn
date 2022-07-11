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
    p_value_th = 0.1

    def test_anm_simulation(self):

        anm = ANM()

        # generate cause and noises
        num_samples = 3000
        x = np.random.randn(num_samples, 1)
        x = x / np.std(x)
        e = np.random.randn(num_samples, 1)
        e = e / np.std(e)

        # simulated data y = x + 2x^3 + e
        y_1 = x + 2 * x**3 + e
        p_value_forward_1, p_value_backward_1 = anm.cause_or_effect(x, y_1)
        self.assertTrue(p_value_forward_1 > self.p_value_th)
        self.assertTrue(p_value_backward_1 < self.p_value_th)

        # simulated data y = 3*exp(x) + e
        y_2 = 5 * np.exp(x) + e
        p_value_forward_2, p_value_backward_2 = anm.cause_or_effect(x, y_2)
        self.assertTrue(p_value_forward_2 > self.p_value_th)
        self.assertTrue(p_value_backward_2 < self.p_value_th)

        # simulated data y = 2 * Sigmoid(x) + e
        y_3 = 5 / (1 + np.exp(-x)) + e
        p_value_forward_3, p_value_backward_3 = anm.cause_or_effect(x, y_3)
        self.assertTrue(p_value_forward_3 > self.p_value_th)
        self.assertTrue(p_value_backward_3 < self.p_value_th)

    # data pair from the Tuebingen cause-effect pair dataset.
    def test_anm_pair(self):

        dataset = np.loadtxt('TestData/pair0001.txt')
        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(dataset[:, 0].reshape(-1, 1), dataset[:, 1].reshape(-1, 1))
        self.assertTrue(p_value_forward > self.p_value_th)
        self.assertTrue(p_value_backward < self.p_value_th)
