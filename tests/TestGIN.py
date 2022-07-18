import random
import unittest

import numpy as np

from causallearn.search.HiddenCausal.GIN.GIN import GIN


class TestGIN(unittest.TestCase):
    def test_case1(self):
        sample_size = 500
        random.seed(42)
        np.random.seed(42)
        L1 = np.random.uniform(-1, 1, size=sample_size)
        L2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
        X1 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X2 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X3 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X4 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        data = np.array([X1, X2, X3, X4]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        _, k = GIN(data)
        k = [sorted(i) for i in k]
        ground_truth = [[0, 1], [2, 3]]
        assert len(k) == len(ground_truth)
        for i in range(len(k)):
            assert np.isclose(k[i], ground_truth[i]).all()


    def test_case2(self):
        sample_size = 500
        random.seed(42)
        np.random.seed(42)
        L1 = np.random.uniform(-1, 1, size=sample_size)
        L2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
        L3 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + np.random.uniform(-1, 1, size=sample_size)
        X1 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X2 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X3 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X4 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X5 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X6 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X7 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X8 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X9 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        _, k = GIN(data)
        k = [sorted(k_i) for k_i in k]
        ground_truth = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert len(k) == len(ground_truth)
        for i in range(len(k)):
            assert np.isclose(k[i], ground_truth[i]).all()


    def test_case3(self):
        sample_size = 500
        random.seed(0)
        np.random.seed(0)
        L1 = np.random.uniform(-1, 1, size=sample_size)
        L2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
        L3 = np.random.uniform(0.5, 0.8) * L1 + np.random.uniform(0.5, 0.8) * L2 + np.random.uniform(-1, 1, size=sample_size)
        L4 = np.random.uniform(0.5, 0.8) * L1 + np.random.uniform(0.5, 0.8) * L2 + np.random.uniform(1.2, 1.8) * L3 + np.random.uniform(-1, 1, size=sample_size)

        X1 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X3 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X4 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X5 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X6 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X7 = np.random.uniform(1.2, 1.8) * L4 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        X8 = np.random.uniform(1.2, 1.8) * L4 + 0.2 * np.random.uniform(-1, 1, size=sample_size)


        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        k = [sorted(k_i) for k_i in k]
        ground_truth = [[0, 1, 2, 3], [4, 5], [6, 7]]
        for i in range(len(k)):
            assert np.isclose(k[i], ground_truth[i]).all()
