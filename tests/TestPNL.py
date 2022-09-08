import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import sys
import unittest
import numpy as np
from causallearn.search.FCMBased.PNL.PNL import PNL


class TestPNL(unittest.TestCase):

    # # Data simulation
    # np.random.seed(0)
    # x = np.random.randn(1000, 1)
    # x = x / np.std(x)
    # e = np.random.randn(1000, 1)
    # e = e / np.std(e)
    # y_1 = (x + x**3 + e)**2
    # y_2 = np.exp(x**2 + e)
    # np.savetxt(r"TestData/pnl_simulation_1.txt", np.hstack([x, y_1]), delimiter=',')
    # np.savetxt(r"TestData/pnl_simulation_2.txt", np.hstack([x, y_2]), delimiter=',')

    # Set the threshold for independence test
    p_value_threshold = 0.1 # useless now but left
    pnl = PNL()

    # Test PNL by some simulated data
    def test_pnl_simulation_1(self):
        # simulated data y = (x + x^3 + e)^2
        simulated_dataset_1 = np.loadtxt('TestData/pnl_simulation_1.txt', delimiter=',')
        simulated_dataset_1_p_value_forward, simulated_dataset_1_p_value_backward = 0.396, 0.0  # round(value, 3) results
        x_1 = simulated_dataset_1[:, 0].reshape(-1, 1)
        y_1 = simulated_dataset_1[:, 1].reshape(-1, 1)
        p_value_forward_1, p_value_backward_1 = self.pnl.cause_or_effect(x_1, y_1)
        self.assertTrue(p_value_forward_1 == simulated_dataset_1_p_value_forward)
        self.assertTrue(p_value_backward_1 == simulated_dataset_1_p_value_backward)
        self.assertTrue(p_value_forward_1 > self.p_value_threshold)
        self.assertTrue(p_value_backward_1 < self.p_value_threshold)
        print('PNL passed the first simulated case!')

    def test_pnl_simulation_2(self):
        # simulated data y = exp(x^2 + e)
        simulated_dataset_2 = np.loadtxt('TestData/pnl_simulation_2.txt', delimiter=',')
        simulated_dataset_2_p_value_forward, simulated_dataset_2_p_value_backward = 0.369, 0.0  # round(value, 3) results
        x_2 = simulated_dataset_2[:, 0].reshape(-1, 1)
        y_2 = simulated_dataset_2[:, 1].reshape(-1, 1)
        p_value_forward_2, p_value_backward_2 = self.pnl.cause_or_effect(x_2, y_2)
        self.assertTrue(p_value_forward_2 == simulated_dataset_2_p_value_forward)
        self.assertTrue(p_value_backward_2 == simulated_dataset_2_p_value_backward)
        self.assertTrue(p_value_forward_2 > self.p_value_threshold)
        self.assertTrue(p_value_backward_2 < self.p_value_threshold)
        print('PNL passed the second simulated case!')

