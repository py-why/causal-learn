import unittest
import numpy as np
from causallearn.search.FCMBased.ANM.ANM import ANM


class TestANM(unittest.TestCase):

    # # Data simulation
    # x = np.random.randn(3000, 1)
    # x = x / np.std(x)
    # e = np.random.randn(3000, 1)
    # e = e / np.std(e)
    # y_1 = x + 2 * x**3 + e
    # y_2 = 5 * np.exp(x) + e
    # y_3 = 3**x + e
    # np.savetxt(r"TestData/anm_simulation_1.txt", np.hstack([x, y_1]), delimiter=',')
    # np.savetxt(r"TestData/anm_simulation_2.txt", np.hstack([x, y_2]), delimiter=',')
    # np.savetxt(r"TestData/anm_simulation_3.txt", np.hstack([x, y_3]), delimiter=',')

    # Set the threshold for independence test
    p_value_threshold = 0.1 # useless now but left
    anm = ANM()

    # Test ANM by some simulated data
    def test_anm_simulation_1(self):
        # simulated data y = x + 2x^3 + e
        simulated_dataset_1 = np.loadtxt('tests/TestData/anm_simulation_1.txt', delimiter=',')
        simulated_dataset_1_p_value_forward, simulated_dataset_1_p_value_backward = 0.99541, 0.0  # round(value, 5) results
        x_1 = simulated_dataset_1[:, 0].reshape(-1, 1)
        y_1 = simulated_dataset_1[:, 1].reshape(-1, 1)
        p_value_forward_1, p_value_backward_1 = self.anm.cause_or_effect(x_1, y_1)
        self.assertTrue(round(p_value_forward_1, 5) == simulated_dataset_1_p_value_forward)
        self.assertTrue(round(p_value_backward_1, 5) == simulated_dataset_1_p_value_backward)
        self.assertTrue(p_value_forward_1 > self.p_value_threshold)
        self.assertTrue(p_value_backward_1 < self.p_value_threshold)
        print('ANM passed the first simulated case!')

    def test_anm_simulation_2(self):
        # simulated data y = 5 * exp(x) + e
        simulated_dataset_2 = np.loadtxt('tests/TestData/anm_simulation_2.txt', delimiter=',')
        simulated_dataset_2_p_value_forward, simulated_dataset_2_p_value_backward = 0.99348, 0.0  # round(value, 5) results
        x_2 = simulated_dataset_2[:, 0].reshape(-1, 1)
        y_2 = simulated_dataset_2[:, 1].reshape(-1, 1)
        p_value_forward_2, p_value_backward_2 = self.anm.cause_or_effect(x_2, y_2)
        self.assertTrue(round(p_value_forward_2, 5) == simulated_dataset_2_p_value_forward)
        self.assertTrue(round(p_value_backward_2, 5) == simulated_dataset_2_p_value_backward)
        self.assertTrue(p_value_forward_2 > self.p_value_threshold)
        self.assertTrue(p_value_backward_2 < self.p_value_threshold)
        print('ANM passed the second simulated case!')

    def test_anm_simulation_3(self):
        # simulated data y = 3^x + e
        simulated_dataset_3 = np.loadtxt('tests/TestData/anm_simulation_3.txt', delimiter=',')
        simulated_dataset_3_p_value_forward, simulated_dataset_3_p_value_backward = 0.65933, 0.0  # round(value, 5) results
        x_3 = simulated_dataset_3[:, 0].reshape(-1, 1)
        y_3 = simulated_dataset_3[:, 1].reshape(-1, 1)
        p_value_forward_3, p_value_backward_3 = self.anm.cause_or_effect(x_3, y_3)
        self.assertTrue(round(p_value_forward_3, 5) == simulated_dataset_3_p_value_forward)
        self.assertTrue(round(p_value_backward_3, 5) == simulated_dataset_3_p_value_backward)
        self.assertTrue(p_value_forward_3 > self.p_value_threshold)
        self.assertTrue(p_value_backward_3 < self.p_value_threshold)
        print('ANM passed the third simulated case!')

    # data pair from the Tuebingen cause-effect pair dataset.
    def test_anm_pair(self):

        dataset = np.loadtxt('tests/TestData/pair0001.txt')
        dataset_p_value_forward, dataset_p_value_backward = 0.14773, 0.0  # round(value, 5) results
        p_value_forward, p_value_backward = self.anm.cause_or_effect(dataset[:, 0].reshape(-1, 1), dataset[:, 1].reshape(-1, 1))
        self.assertTrue(round(p_value_forward, 5) == dataset_p_value_forward)
        self.assertTrue(round(p_value_backward, 5) == dataset_p_value_backward)
        self.assertTrue(p_value_forward > self.p_value_threshold)
        self.assertTrue(p_value_backward < self.p_value_threshold)
        print('ANM passed the real data case!')
