"""HSCIC with the conditional permutation test (HSCIC-CPT).

The HSCIC statistic follows Park and Muandet (2020). Conditional
permutations follow Algorithms 1--2 of Berrett et al. (2020): every reported
permutation is a separate Markov-chain branch from one shared centre.
Regularisation selection targets the evaluation rejection rate in Algorithm 1
of Laumann et al. (2023), using a shared exchangeable permutation pool on a
training block that is disjoint from the final test block.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.spatial.distance import pdist, squareform


DEFAULT_HSCIC_RIDGE_GRID = tuple(np.logspace(-4.0, -1.0, 18))


def _as_column_matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one- or two-dimensional array.")
    return array


def _gaussian_kernel(values):
    values = _as_column_matrix(values, "kernel input")
    distances = pdist(values, metric="euclidean")
    # Match the median heuristic used by the HSCIC implementation:
    # sigma^2 = median(||x_i - x_j||^2) / 2. PyTorch's median is the
    # lower middle value when the number of pairs is even, and duplicate
    # observations remain part of the empirical distance distribution.
    if distances.size:
        squared_distances = distances * distances
        middle = (squared_distances.size - 1) // 2
        median_squared = float(np.partition(squared_distances, middle)[middle])
        bandwidth = np.sqrt(0.5 * median_squared)
    else:
        bandwidth = 1.0
    bandwidth = max(bandwidth, np.finfo(float).eps)
    squared = squareform(distances * distances)
    return np.exp(-0.5 * squared / (bandwidth * bandwidth))


def _swap_log_odds(permutation, left, right, log_likelihood):
    """Compute the pairwise swap log odds in Berrett et al. Algorithm 1."""
    value_left = permutation[left]
    value_right = permutation[right]
    return (
        log_likelihood[value_left, right]
        + log_likelihood[value_right, left]
        - log_likelihood[value_left, left]
        - log_likelihood[value_right, right]
    )


def _swap_chain(initial, log_likelihood, n_steps, rng):
    """Run Algorithm 1 of Berrett et al. (2020)."""
    permutation = np.asarray(initial, dtype=int).copy()
    n = permutation.size
    n_pairs = n // 2
    for _ in range(int(n_steps)):
        shuffled = rng.permutation(n)
        left = shuffled[:n_pairs]
        right = shuffled[n_pairs:2 * n_pairs]
        value_left = permutation[left].copy()
        value_right = permutation[right].copy()
        log_odds = _swap_log_odds(
            permutation,
            left,
            right,
            log_likelihood,
        )
        swap = rng.random(n_pairs) < expit(np.maximum(-500.0, log_odds))
        permutation[left] = np.where(swap, value_right, value_left)
        permutation[right] = np.where(swap, value_left, value_right)
    return permutation


def conditional_permutations(
    log_likelihood,
    n_permutations=500,
    n_steps=50,
    random_state=None,
):
    """Draw exchangeable conditional permutations.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_samples, n_samples)
        Entry ``[i, j]`` is ``log q(X_i | Z_j)``.
    n_permutations : int
        Number of permutations to draw.
    n_steps : int
        Number of parallel pairwise-swap steps in each Markov-chain branch.
    random_state : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray, shape (n_permutations, n_samples)
        Rows contain zero-based permutation indices.
    """
    likelihood = np.asarray(log_likelihood, dtype=float)
    if (
        likelihood.ndim != 2
        or likelihood.shape[0] != likelihood.shape[1]
        or not np.isfinite(likelihood).all()
    ):
        raise ValueError("log_likelihood must be a finite square matrix.")
    if int(n_permutations) <= 0 or int(n_steps) <= 0:
        raise ValueError("n_permutations and n_steps must be positive.")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    identity = np.arange(likelihood.shape[0])
    centre = _swap_chain(identity, likelihood, int(n_steps), rng)
    return np.stack([
        _swap_chain(centre, likelihood, int(n_steps), rng)
        for _ in range(int(n_permutations))
    ])


def gaussian_log_likelihood_matrices(x_train, z_train, x_test, z_test):
    """Fit a scalar linear-Gaussian model and return CPT log-likelihoods.

    The model is fitted only on ``(x_train, z_train)``. The returned matrices
    have entries ``log qhat(x_i | z_j)`` for the training and test blocks.
    Terms depending only on ``z_j`` are omitted because they cancel from
    every CPT swap odds ratio.
    """
    train_x = _as_column_matrix(x_train, "x_train")
    test_x = _as_column_matrix(x_test, "x_test")
    train_z = _as_column_matrix(z_train, "z_train")
    test_z = _as_column_matrix(z_test, "z_test")
    if train_x.shape[1] != 1 or test_x.shape[1] != 1:
        raise ValueError("the built-in Gaussian conditional model requires scalar X.")
    if train_x.shape[0] != train_z.shape[0]:
        raise ValueError("x_train and z_train must have matching rows.")
    if test_x.shape[0] != test_z.shape[0] or train_z.shape[1] != test_z.shape[1]:
        raise ValueError("test rows and conditioning dimensions must match.")

    design = np.column_stack((np.ones(train_z.shape[0]), train_z))
    coefficients, *_ = np.linalg.lstsq(design, train_x[:, 0], rcond=None)
    residual = train_x[:, 0] - design @ coefficients
    degrees = max(train_z.shape[0] - design.shape[1], 1)
    variance = max(
        float(residual @ residual / degrees),
        np.finfo(float).eps,
    )

    def build(x_values, z_values):
        mean = np.column_stack((np.ones(z_values.shape[0]), z_values)) @ coefficients
        values = x_values[:, 0]
        return (
            -values[:, None] ** 2 / (2.0 * variance)
            + values[:, None] * mean[None, :] / variance
        )

    return build(train_x, train_z), build(test_x, test_z)


def _hscic_statistic(kernel_x, kernel_y, weights, kernel_y_weights=None):
    """Average the squared Park--Muandet HSCIC over fixed landmarks."""
    kyw = kernel_y @ weights if kernel_y_weights is None else kernel_y_weights
    kxw = kernel_x @ weights
    first = np.sum(weights * ((kernel_x * kernel_y) @ weights), axis=0)
    second = np.sum(weights * kxw * kyw, axis=0)
    third = np.sum(weights * kxw, axis=0) * np.sum(weights * kyw, axis=0)
    return float(np.mean(first - 2.0 * second + third))


def _biased_hsic(kernel_x, kernel_y):
    centred_x = (
        kernel_x
        - kernel_x.mean(axis=0, keepdims=True)
        - kernel_x.mean(axis=1, keepdims=True)
        + kernel_x.mean()
    )
    centred_y = (
        kernel_y
        - kernel_y.mean(axis=0, keepdims=True)
        - kernel_y.mean(axis=1, keepdims=True)
        + kernel_y.mean()
    )
    return float(np.mean(centred_x * centred_y))


def _ridge_weights(kernel_z, landmarks, ridge, decomposition=None):
    """Return ``(K_Z + ridge I)^{-1} K_Z[:, landmarks]`` stably."""
    if decomposition is None:
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_z)
        eigenvalues = np.maximum(eigenvalues, 0.0)
    else:
        eigenvalues, eigenvectors = decomposition
    right = kernel_z[:, landmarks]
    projected = eigenvectors.T @ right
    return eigenvectors @ (projected / (eigenvalues[:, None] + float(ridge)))


class HSCICCPT:
    """Park--Muandet HSCIC calibrated by conditional permutations."""

    def __init__(
        self,
        ridge_grid=DEFAULT_HSCIC_RIDGE_GRID,
        alpha=0.05,
        n_pretests=100,
        n_reference_permutations=199,
        n_permutations=500,
        n_steps=50,
        n_landmarks=None,
    ):
        candidates = tuple(float(value) for value in ridge_grid)
        if not candidates or any(
            not np.isfinite(value) or value <= 0.0
            for value in candidates
        ):
            raise ValueError("ridge_grid must contain finite positive values.")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        if min(
            int(n_pretests),
            int(n_reference_permutations),
            int(n_permutations),
            int(n_steps),
        ) <= 0:
            raise ValueError("permutation counts and n_steps must be positive.")
        if n_landmarks is not None and int(n_landmarks) <= 0:
            raise ValueError("n_landmarks must be positive or None.")
        self.ridge_grid = candidates
        self.alpha = float(alpha)
        self.n_pretests = int(n_pretests)
        self.n_reference_permutations = int(n_reference_permutations)
        self.n_permutations = int(n_permutations)
        self.n_steps = int(n_steps)
        self.n_landmarks = None if n_landmarks is None else int(n_landmarks)

    def _kernel_inputs(self, x, y, z, rng):
        kernel_x = _gaussian_kernel(x)
        kernel_y = _gaussian_kernel(y)
        kernel_z = _gaussian_kernel(z)
        n = kernel_z.shape[0]
        if self.n_landmarks is None or self.n_landmarks >= n:
            landmarks = np.arange(n)
        else:
            landmarks = np.sort(rng.choice(n, self.n_landmarks, replace=False))
        return kernel_x, kernel_y, kernel_z, landmarks

    def select_ridge(self, x, y, z, log_likelihood, random_state=None):
        """Select the HSCIC ridge on null-true training permutations.

        A single exchangeable CPT pool is shared across candidates. Its first
        ``n_pretests`` members act as null-true pretests and the remaining
        members form their common reference distribution. Sharing the pool
        preserves the exchangeable-rank argument while avoiding a separate
        CPT run for every candidate and pretest.
        """
        rng = np.random.default_rng(random_state)
        x = _as_column_matrix(x, "x")
        y = _as_column_matrix(y, "y")
        z = _as_column_matrix(z, "z")
        if x.shape[0] != y.shape[0] or x.shape[0] != z.shape[0]:
            raise ValueError("x, y and z must have matching rows.")
        likelihood = np.asarray(log_likelihood, dtype=float)
        if likelihood.shape != (x.shape[0], x.shape[0]) or not np.isfinite(
            likelihood
        ).all():
            raise ValueError("log_likelihood must be a finite (n, n) matrix.")
        kernel_x, kernel_y, kernel_z, landmarks = self._kernel_inputs(x, y, z, rng)
        permutations = conditional_permutations(
            likelihood,
            self.n_pretests + self.n_reference_permutations,
            self.n_steps,
            rng,
        )
        rates = []
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_z)
        decomposition = (np.maximum(eigenvalues, 0.0), eigenvectors)
        for ridge in self.ridge_grid:
            weights = _ridge_weights(
                kernel_z,
                landmarks,
                ridge,
                decomposition,
            )
            kyw = kernel_y @ weights
            statistics = np.asarray([
                _hscic_statistic(
                    kernel_x[np.ix_(permutation, permutation)],
                    kernel_y,
                    weights,
                    kyw,
                )
                for permutation in permutations
            ])
            observed = statistics[:self.n_pretests]
            reference = statistics[self.n_pretests:]
            p_values = (
                1.0
                + np.sum(reference[None, :] >= observed[:, None], axis=1)
            ) / (self.n_reference_permutations + 1.0)
            rates.append(float(np.mean(p_values <= self.alpha)))
        distance = min(abs(rate - self.alpha) for rate in rates)
        best = max(
            index
            for index, rate in enumerate(rates)
            if np.isclose(abs(rate - self.alpha), distance, rtol=0.0, atol=1e-12)
        )
        return self.ridge_grid[best], tuple(rates)

    def compute_pvalue(
        self,
        x_train,
        y_train,
        z_train,
        x_test,
        y_test,
        z_test,
        log_likelihood_train,
        log_likelihood_test,
        random_state=None,
    ):
        """Select regularisation on training data and test on held-out data."""
        seed_sequence = np.random.SeedSequence(random_state)
        ridge_seed, test_seed = [
            int(child.generate_state(1)[0])
            for child in seed_sequence.spawn(2)
        ]
        ridge, _ = self.select_ridge(
            x_train,
            y_train,
            z_train,
            log_likelihood_train,
            ridge_seed,
        )
        return self.compute_pvalue_fixed_ridge(
            x_test,
            y_test,
            z_test,
            log_likelihood_test,
            ridge,
            test_seed,
        )

    def compute_pvalue_fixed_ridge(
        self,
        x,
        y,
        z,
        log_likelihood,
        ridge,
        random_state=None,
    ):
        """Run the final HSCIC-CPT test at a fixed positive ridge."""
        x = _as_column_matrix(x, "x")
        y = _as_column_matrix(y, "y")
        z = _as_column_matrix(z, "z")
        if x.shape[0] != y.shape[0] or x.shape[0] != z.shape[0]:
            raise ValueError("x, y and z must have matching rows.")
        if not np.isfinite(ridge) or float(ridge) <= 0.0:
            raise ValueError("ridge must be finite and positive.")
        likelihood = np.asarray(log_likelihood, dtype=float)
        if likelihood.shape != (x.shape[0], x.shape[0]) or not np.isfinite(
            likelihood
        ).all():
            raise ValueError("log_likelihood must be a finite (n, n) matrix.")
        rng = np.random.default_rng(random_state)
        kernel_x, kernel_y, kernel_z, landmarks = self._kernel_inputs(x, y, z, rng)
        weights = _ridge_weights(kernel_z, landmarks, ridge)
        kyw = kernel_y @ weights
        observed = _hscic_statistic(kernel_x, kernel_y, weights, kyw)
        permutations = conditional_permutations(
            likelihood,
            self.n_permutations,
            self.n_steps,
            rng,
        )
        null = np.asarray([
            _hscic_statistic(
                kernel_x[np.ix_(permutation, permutation)],
                kernel_y,
                weights,
                kyw,
            )
            for permutation in permutations
        ])
        return float((1.0 + np.sum(null >= observed)) / (self.n_permutations + 1.0))

    def compute_unconditional_pvalue(self, x, y, random_state=None):
        """Use ordinary permutations and biased HSIC when no Z is supplied."""
        x = _as_column_matrix(x, "x")
        y = _as_column_matrix(y, "y")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have matching rows.")
        rng = np.random.default_rng(random_state)
        kernel_x = _gaussian_kernel(x)
        kernel_y = _gaussian_kernel(y)
        observed = _biased_hsic(kernel_x, kernel_y)
        null = np.asarray([
            _biased_hsic(
                kernel_x[np.ix_(permutation, permutation)],
                kernel_y,
            )
            for permutation in (
                rng.permutation(x.shape[0])
                for _ in range(self.n_permutations)
            )
        ])
        return float((1.0 + np.sum(null >= observed)) / (self.n_permutations + 1.0))
