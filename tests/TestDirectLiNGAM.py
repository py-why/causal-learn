import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam


class TestDirectLiNGAM(unittest.TestCase):

    def test_DirectLiNGAM(self):
        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

        model = lingam.DirectLiNGAM()
        model.fit(X)

        print(model.causal_order_)
        print(model.adjacency_matrix_)


class TestDirectLiNGAMResidual(unittest.TestCase):

    def test_residual_matches_least_squares_slope(self):
        rng = np.random.default_rng(903)
        for n_samples in (5, 20, 200):
            for location, scale in ((0, 1), (7, 0.2), (-3, 5)):
                with self.subTest(n_samples=n_samples, location=location, scale=scale):
                    xj = location + scale * rng.laplace(size=n_samples)
                    xi = 2 * xj + rng.laplace(size=n_samples) + 4
                    centered_xj = xj - xj.mean()
                    centered_xi = xi - xi.mean()
                    slope = np.linalg.lstsq(centered_xj[:, None], centered_xi, rcond=None)[0][0]
                    residual = lingam.DirectLiNGAM()._residual(xi, xj)
                    np.testing.assert_allclose(residual, xi - slope * xj, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(
                        np.dot(residual - residual.mean(), centered_xj), 0, atol=1e-10
                    )

    def test_fit_uses_least_squares_residuals(self):
        class LeastSquaresDirectLiNGAM(lingam.DirectLiNGAM):
            def _residual(self, xi, xj):
                centered_xj = xj - xj.mean()
                slope = np.linalg.lstsq(
                    centered_xj[:, None], xi - xi.mean(), rcond=None
                )[0][0]
                return xi - slope * xj

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        X[:, 1] += 0.2 * X[:, 0]
        X[:, 2] += 0.3 * X[:, 1]
        X[:, 3] += 0.15 * X[:, 0]
        for measure in ('pwling', 'kernel'):
            with self.subTest(measure=measure):
                reference = LeastSquaresDirectLiNGAM(measure=measure).fit(X)
                actual = lingam.DirectLiNGAM(measure=measure).fit(X)
                self.assertEqual(actual.causal_order_, reference.causal_order_)
                np.testing.assert_allclose(
                    actual.adjacency_matrix_, reference.adjacency_matrix_, atol=1e-12
                )
