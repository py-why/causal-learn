import os
import unittest
import numpy as np
from causallearn.search.FCMBased.IGCI.IGCI import IGCI, igci

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'TestData')


class TestIGCI(unittest.TestCase):

    # # Data simulation
    # rng = np.random.default_rng(2026)
    # n = 1000
    #
    # # 1. Uniform cause, x^3
    # x1 = rng.uniform(0.05, 1.0, size=n)
    # y1 = x1**3
    # np.savetxt("tests/TestData/igci_simulation_1.txt", np.column_stack([x1, y1]), delimiter=",", fmt="%.8f")
    #
    # # 2. Uniform cause, exp(x)
    # x2 = rng.uniform(0.0, 1.0, size=n)
    # y2 = np.exp(x2)
    # np.savetxt("tests/TestData/igci_simulation_2.txt", np.column_stack([x2, y2]), delimiter=",", fmt="%.8f")
    #
    # # 3. Gaussian cause, tanh(x)
    # x3 = rng.normal(0.0, 1.0, size=n)
    # y3 = np.tanh(x3)
    # np.savetxt("tests/TestData/igci_simulation_3.txt", np.column_stack([x3, y3]), delimiter=",", fmt="%.8f")

    def test_igci_input_validation(self):
        # Invalid ref_measure
        with self.assertRaises(ValueError):
            IGCI(ref_measure='laplace')

        # Invalid estimator
        with self.assertRaises(ValueError):
            IGCI(estimator='kernel')

        model = IGCI()

        # Mismatched lengths
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            model.cause_or_effect(x, y)

        # Less than 2 samples
        with self.assertRaises(ValueError):
            model.cause_or_effect(np.array([1.0]), np.array([2.0]))

        # Multidimensional array (n, 2)
        with self.assertRaises(ValueError):
            model.cause_or_effect(np.ones((10, 2)), np.ones((10, 1)))

        # NaN values
        with self.assertRaises(ValueError):
            model.cause_or_effect(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))

        # Inf values
        with self.assertRaises(ValueError):
            model.cause_or_effect(np.array([1.0, np.inf, 3.0]), np.array([1.0, 2.0, 3.0]))

        # Constant input (zero variance)
        with self.assertRaises(ValueError):
            model.cause_or_effect(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))

        model_gauss = IGCI(ref_measure='gaussian')
        with self.assertRaises(ValueError):
            model_gauss.cause_or_effect(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_igci_simulation_1(self):
        # Simulated data: y = x^3, x ~ Uniform(0.05, 1)
        dataset = np.loadtxt(os.path.join(TEST_DATA_DIR, 'igci_simulation_1.txt'), delimiter=',')
        x = dataset[:, 0].reshape(-1, 1)
        y = dataset[:, 1].reshape(-1, 1)

        # 1. Slope estimator with uniform reference
        model_slope = IGCI(ref_measure='uniform', estimator='slope')
        sf_slope, sb_slope = model_slope.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_slope, -0.52799, places=4)
        self.assertAlmostEqual(sb_slope, 0.52799, places=4)
        self.assertTrue(sf_slope < sb_slope)
        self.assertEqual(model_slope.predict_direction(x, y), 1)
        self.assertEqual(model_slope.predict_direction(y, x), -1)
        self.assertTrue(model_slope.compute_score(x, y) > 0)

        # 2. Entropy estimator with uniform reference
        model_entropy = IGCI(ref_measure='uniform', estimator='entropy')
        sf_ent, sb_ent = model_entropy.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_ent, -0.52799, places=4)
        self.assertAlmostEqual(sb_ent, 0.52799, places=4)
        self.assertTrue(sf_ent < sb_ent)
        self.assertEqual(model_entropy.predict_direction(x, y), 1)
        self.assertEqual(model_entropy.predict_direction(y, x), -1)

    def test_igci_simulation_2(self):
        # Simulated data: y = exp(x), x ~ Uniform(0, 1)
        dataset = np.loadtxt(os.path.join(TEST_DATA_DIR, 'igci_simulation_2.txt'), delimiter=',')
        x = dataset[:, 0].reshape(-1, 1)
        y = dataset[:, 1].reshape(-1, 1)

        model_slope = IGCI(ref_measure='uniform', estimator='slope')
        sf_slope, sb_slope = model_slope.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_slope, -0.04843, places=4)
        self.assertAlmostEqual(sb_slope, 0.04843, places=4)
        self.assertTrue(sf_slope < sb_slope)
        self.assertEqual(model_slope.predict_direction(x, y), 1)

        model_entropy = IGCI(ref_measure='uniform', estimator='entropy')
        sf_ent, sb_ent = model_entropy.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_ent, -0.04843, places=4)
        self.assertAlmostEqual(sb_ent, 0.04843, places=4)
        self.assertTrue(sf_ent < sb_ent)
        self.assertEqual(model_entropy.predict_direction(x, y), 1)

    def test_igci_simulation_3(self):
        # Simulated data: y = tanh(x), x ~ Normal(0, 1)
        dataset = np.loadtxt(os.path.join(TEST_DATA_DIR, 'igci_simulation_3.txt'), delimiter=',')
        x = dataset[:, 0].reshape(-1, 1)
        y = dataset[:, 1].reshape(-1, 1)

        model_slope = IGCI(ref_measure='gaussian', estimator='slope')
        sf_slope, sb_slope = model_slope.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_slope, -0.30895, places=4)
        self.assertAlmostEqual(sb_slope, 0.30895, places=4)
        self.assertTrue(sf_slope < sb_slope)
        self.assertEqual(model_slope.predict_direction(x, y), 1)
        self.assertEqual(model_slope.predict_direction(y, x), -1)

        model_entropy = IGCI(ref_measure='gaussian', estimator='entropy')
        sf_ent, sb_ent = model_entropy.cause_or_effect(x, y)
        self.assertAlmostEqual(sf_ent, -0.30895, places=4)
        self.assertAlmostEqual(sb_ent, 0.30895, places=4)
        self.assertTrue(sf_ent < sb_ent)
        self.assertEqual(model_entropy.predict_direction(x, y), 1)

    def test_igci_noisy_case(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0.0, 1.0, size=n)
        y = x**3 + 0.05 * rng.normal(size=n)

        model_slope = IGCI(ref_measure='uniform', estimator='slope')
        sf_s, sb_s = model_slope.cause_or_effect(x, y)
        self.assertTrue(sf_s < sb_s)
        self.assertEqual(model_slope.predict_direction(x, y), 1)
        self.assertEqual(model_slope.predict_direction(y, x), -1)

        model_entropy = IGCI(ref_measure='uniform', estimator='entropy')
        sf_e, sb_e = model_entropy.cause_or_effect(x, y)
        self.assertTrue(sf_e < sb_e)
        self.assertEqual(model_entropy.predict_direction(x, y), 1)

    def test_igci_identical_distribution(self):
        x = np.linspace(0.0, 1.0, 100)
        y = x.copy()
        model = IGCI()
        sf, sb = model.cause_or_effect(x, y)
        self.assertAlmostEqual(sf, sb, places=7)
        self.assertEqual(model.predict_direction(x, y), 0)
        self.assertAlmostEqual(model.compute_score(x, y), 0.0, places=7)

    def test_functional_api_and_shapes(self):
        rng = np.random.default_rng(123)
        x_1d = rng.uniform(0, 1, 500)
        y_1d = x_1d**2
        x_2d = x_1d.reshape(-1, 1)
        y_2d = y_1d.reshape(-1, 1)

        model = IGCI(ref_measure='uniform', estimator='slope')
        res_1d = model.cause_or_effect(x_1d, y_1d)
        res_2d = model.cause_or_effect(x_2d, y_2d)
        self.assertEqual(res_1d, res_2d)

        res_fn = igci(x_1d, y_1d, ref_measure='uniform', estimator='slope')
        self.assertEqual(res_1d, res_fn)


if __name__ == '__main__':
    unittest.main()
