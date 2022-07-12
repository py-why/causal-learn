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

    # Set the threshold for independence test
    p_value_threshold = 0.1
    anm = ANM()

    # Test ANM by some simulated data
    def test_anm_simulation_1(self):
        # simulated data y = x + 2x^3 + e
        simulated_dataset_1 = np.loadtxt('TestData/anm_simulation_1.txt', delimiter=',')
        x_1 = simulated_dataset_1[:, 0].reshape(-1, 1)
        y_1 = simulated_dataset_1[:, 1].reshape(-1, 1)
        p_value_forward_1, p_value_backward_1 = self.anm.cause_or_effect(x_1, y_1)
        self.assertTrue(p_value_forward_1 > self.p_value_threshold)
        self.assertTrue(p_value_backward_1 < self.p_value_threshold)
        print('ANM passed the first simulated case!')

    def test_anm_simulation_2(self):
        # simulated data y = 5 * exp(x) + e
        simulated_dataset_2 = np.loadtxt('TestData/anm_simulation_2.txt', delimiter=',')
        x_2 = simulated_dataset_2[:, 0].reshape(-1, 1)
        y_2 = simulated_dataset_2[:, 1].reshape(-1, 1)
        p_value_forward_2, p_value_backward_2 = self.anm.cause_or_effect(x_2, y_2)
        self.assertTrue(p_value_forward_2 > self.p_value_threshold)
        self.assertTrue(p_value_backward_2 < self.p_value_threshold)
        print('ANM passed the second simulated case!')

    def test_anm_simulation_3(self):
        # simulated data y = 3^x + e
        simulated_dataset_3 = np.loadtxt('TestData/anm_simulation_3.txt', delimiter=',')
        x_3 = simulated_dataset_3[:, 0].reshape(-1, 1)
        y_3 = simulated_dataset_3[:, 1].reshape(-1, 1)
        p_value_forward_3, p_value_backward_3 = self.anm.cause_or_effect(x_3, y_3)
        self.assertTrue(p_value_forward_3 > self.p_value_threshold)
        self.assertTrue(p_value_backward_3 < self.p_value_threshold)
        print('ANM passed the third simulated case!')

    # data pair from the Tuebingen cause-effect pair dataset.
    def test_anm_pair(self):

        dataset = np.loadtxt('TestData/pair0001.txt')
        p_value_forward, p_value_backward = self.anm.cause_or_effect(dataset[:, 0].reshape(-1, 1), dataset[:, 1].reshape(-1, 1))
        self.assertTrue(p_value_forward > self.p_value_threshold)
        self.assertTrue(p_value_backward < self.p_value_threshold)
        print('ANM passed the real data case!')
