import unittest
from unittest.mock import patch

import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.HSCICCPT.HSCICCPT import (
    HSCICCPT,
    _gaussian_kernel,
    _hscic_statistic,
    _ridge_weights,
    _swap_log_odds,
    conditional_permutations,
    gaussian_log_likelihood_matrices,
)
from causallearn.utils.cit import CIT, hscic_cpt


class TestCIT_HSCICCPT(unittest.TestCase):
    def test_gaussian_kernel_uses_hscic_median_heuristic(self):
        values = np.asarray([[0.0], [2.0]])
        # The only positive distance is 2, so sigma=2/sqrt(2) and the
        # off-diagonal kernel value is exp(-1).
        np.testing.assert_allclose(
            _gaussian_kernel(values),
            np.asarray([[1.0, np.exp(-1.0)], [np.exp(-1.0), 1.0]]),
        )

    def test_gaussian_kernel_matches_torch_lower_median_with_duplicates(self):
        values = np.asarray([[0.0], [0.0], [2.0]])
        # Squared upper-triangle distances are {0, 4, 4}; their median is 4.
        expected = np.asarray([
            [1.0, 1.0, np.exp(-1.0)],
            [1.0, 1.0, np.exp(-1.0)],
            [np.exp(-1.0), np.exp(-1.0), 1.0],
        ])
        np.testing.assert_allclose(_gaussian_kernel(values), expected)

        # PyTorch uses the lower middle value for an even number of entries.
        # The six squared distances are {0, 1, 1, 4, 4, 9}, so sigma^2=1/2.
        even_values = np.asarray([[0.0], [0.0], [1.0], [3.0]])
        self.assertAlmostEqual(_gaussian_kernel(even_values)[0, 2], np.exp(-1.0))

    def test_conditional_permutations_branch_from_shared_centre(self):
        starts = []

        def fake_chain(initial, log_likelihood, n_steps, rng):
            starts.append(initial.copy())
            return np.roll(initial, 1)

        with patch(
            "causallearn.utils.HSCICCPT.HSCICCPT._swap_chain",
            side_effect=fake_chain,
        ):
            draws = conditional_permutations(
                np.zeros((6, 6)),
                n_permutations=4,
                n_steps=3,
                random_state=7,
            )
        centre = np.roll(np.arange(6), 1)
        self.assertEqual(len(starts), 5)
        np.testing.assert_array_equal(starts[0], np.arange(6))
        for start in starts[1:]:
            np.testing.assert_array_equal(start, centre)
        for draw in draws:
            np.testing.assert_array_equal(draw, np.roll(centre, 1))

    def test_conditional_sampler_returns_valid_permutations(self):
        draws = conditional_permutations(
            np.zeros((8, 8)),
            n_permutations=5,
            n_steps=3,
            random_state=11,
        )
        self.assertEqual(draws.shape, (5, 8))
        expected = np.arange(8)
        for draw in draws:
            np.testing.assert_array_equal(np.sort(draw), expected)

    def test_swap_log_odds_matches_cpt_reference_formula(self):
        likelihood = np.asarray([
            [0.1, -0.4, 0.7, 0.2],
            [0.8, 0.3, -0.2, 0.5],
            [-0.6, 0.9, 0.4, -0.1],
            [0.2, -0.7, 0.6, 1.0],
        ])
        permutation = np.asarray([2, 0, 3, 1])
        left = np.asarray([0, 1])
        right = np.asarray([3, 2])
        expected = np.asarray([
            (
                likelihood[permutation[i], j]
                + likelihood[permutation[j], i]
                - likelihood[permutation[i], i]
                - likelihood[permutation[j], j]
            )
            for i, j in zip(left, right)
        ])
        np.testing.assert_allclose(
            _swap_log_odds(permutation, left, right, likelihood),
            expected,
        )

    def test_gaussian_likelihood_matches_reference_formula(self):
        z_train = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
        x_train = 0.5 + 1.5 * z_train + np.asarray([[0.1], [-0.1], [0.2], [-0.2]])
        z_test = np.asarray([[-0.5], [1.5]])
        x_test = np.asarray([[0.2], [2.4]])
        train, test = gaussian_log_likelihood_matrices(
            x_train,
            z_train,
            x_test,
            z_test,
        )
        self.assertEqual(train.shape, (4, 4))
        self.assertEqual(test.shape, (2, 2))
        self.assertTrue(np.isfinite(train).all())
        self.assertTrue(np.isfinite(test).all())

        design = np.column_stack((np.ones(4), z_train))
        coefficients, *_ = np.linalg.lstsq(design, x_train[:, 0], rcond=None)
        residual = x_train[:, 0] - design @ coefficients
        variance = residual @ residual / 2
        mean = np.column_stack((np.ones(2), z_test)) @ coefficients
        expected = (
            -x_test[:, 0, None] ** 2 / (2.0 * variance)
            + x_test[:, 0, None] * mean[None, :] / variance
        )
        np.testing.assert_allclose(test, expected)

    def test_vectorised_hscic_matches_definition_at_each_landmark(self):
        rng = np.random.default_rng(12)
        features_x = rng.normal(size=(7, 3))
        features_y = rng.normal(size=(7, 2))
        kernel_x = features_x @ features_x.T
        kernel_y = features_y @ features_y.T
        weights = rng.normal(size=(7, 4))
        direct = []
        for column in range(weights.shape[1]):
            weight = weights[:, column]
            first = weight @ ((kernel_x * kernel_y) @ weight)
            second = weight @ ((kernel_x @ weight) * (kernel_y @ weight))
            third = (
                (weight @ kernel_x @ weight)
                * (weight @ kernel_y @ weight)
            )
            direct.append(first - 2.0 * second + third)
        self.assertAlmostEqual(
            _hscic_statistic(kernel_x, kernel_y, weights),
            float(np.mean(direct)),
        )

    def test_eigendecomposition_ridge_weights_match_direct_solve(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(12, 4))
        kernel_z = _gaussian_kernel(features)
        landmarks = np.asarray([0, 3, 8])
        ridge = 0.03
        expected = np.linalg.solve(
            kernel_z + ridge * np.eye(kernel_z.shape[0]),
            kernel_z[:, landmarks],
        )
        np.testing.assert_allclose(
            _ridge_weights(kernel_z, landmarks, ridge),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_cit_integration_is_cached_symmetric_and_reproducible(self):
        rng = np.random.default_rng(11)
        z = rng.normal(size=60)
        x = z + rng.normal(size=60)
        y = z + rng.normal(size=60)
        data = np.column_stack((x, y, z))
        parameters = dict(
            random_state=19,
            ridge_grid=(1e-3, 1e-1),
            n_pretests=3,
            n_reference_permutations=9,
            n_permutations=19,
            n_steps=3,
            n_landmarks=4,
        )
        test = CIT(data, hscic_cpt, **parameters)
        p_value = test(0, 1, [2])
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        self.assertEqual(p_value, test(1, 0, [2]))
        self.assertEqual(p_value, CIT(data, hscic_cpt, **parameters)(0, 1, [2]))

    def test_cit_result_is_invariant_to_endpoint_column_order(self):
        rng = np.random.default_rng(31)
        z = rng.normal(size=42)
        x = z + 0.4 * rng.normal(size=42)
        y = z * z + 0.4 * rng.normal(size=42)
        parameters = dict(
            random_state=8,
            ridge_grid=(1e-2,),
            n_pretests=2,
            n_reference_permutations=5,
            n_permutations=7,
            n_steps=2,
            n_landmarks=3,
        )
        p_xy = CIT(
            np.column_stack((x, y, z)),
            hscic_cpt,
            **parameters,
        )(0, 1, [2])
        p_yx = CIT(
            np.column_stack((y, x, z)),
            hscic_cpt,
            **parameters,
        )(0, 1, [2])
        self.assertEqual(p_xy, p_yx)

    def test_custom_conditional_density_callback(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(36, 3))
        calls = []

        def uniform_density(x_train, z_train, x_test, z_test):
            calls.append((x_train.shape, z_train.shape, x_test.shape, z_test.shape))
            return (
                np.zeros((x_train.shape[0], x_train.shape[0])),
                np.zeros((x_test.shape[0], x_test.shape[0])),
            )

        test = CIT(
            data,
            hscic_cpt,
            conditional_density=uniform_density,
            random_state=5,
            ridge_grid=(1e-2,),
            n_pretests=2,
            n_reference_permutations=4,
            n_permutations=5,
            n_steps=2,
            n_landmarks=3,
        )
        p_value = test(0, 1, [2])
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        self.assertEqual(
            calls,
            [
                ((18, 1), (18, 1), (18, 1), (18, 1)),
                ((18, 1), (18, 1), (18, 1), (18, 1)),
            ],
        )

    def test_unconditional_path_returns_a_valid_pvalue(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(30, 1))
        y = rng.normal(size=(30, 1))
        test = HSCICCPT(
            ridge_grid=(1e-2,),
            n_pretests=2,
            n_reference_permutations=4,
            n_permutations=9,
            n_steps=2,
        )
        p_value = test.compute_unconditional_pvalue(x, y, random_state=4)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_pc_accepts_hscic_cpt(self):
        rng = np.random.default_rng(23)
        z = rng.normal(size=36)
        data = np.column_stack((
            z + 0.2 * rng.normal(size=36),
            z + 0.2 * rng.normal(size=36),
            z,
        ))
        graph = pc(
            data,
            alpha=0.05,
            indep_test=hscic_cpt,
            stable=True,
            show_progress=False,
            max_k=1,
            random_state=9,
            ridge_grid=(1e-2,),
            n_pretests=2,
            n_reference_permutations=4,
            n_permutations=5,
            n_steps=2,
            n_landmarks=3,
        )
        self.assertEqual(graph.G.get_num_nodes(), 3)


if __name__ == "__main__":
    unittest.main()
