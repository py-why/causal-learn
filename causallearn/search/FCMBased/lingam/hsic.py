"""
Python implementation of the LiNGAM algorithms.
 * some slight modification for speedup, 04/26/2022
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import time

import numpy as np
from scipy.stats import gamma
from statsmodels.nonparametric import bandwidths

__all__ = ['get_kernel_width', 'get_gram_matrix', 'hsic_teststat', 'hsic_test_gamma']


def get_kernel_width(X):
    """Calculate the bandwidth to median distance between points.
    Use at most 100 points (since median is only a heuristic,
    and 100 points is sufficient for a robust estimate).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    Returns
    -------
    float
        The bandwidth parameter.
    """
    n_samples = X.shape[0]
    if n_samples > 100:
        X_med = X[:100, :]
        n_samples = 100
    else:
        X_med = X

    G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
    dists = G + G.T - 2 * np.dot(X_med, X_med.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n_samples ** 2, 1)

    return np.sqrt(0.5 * np.median(dists[dists > 0]))

def _rbf_dot(X, Y, width):
    """Compute the inner product of radial basis functions."""
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    G = np.sum(X * X, 1).reshape(n_samples_X, 1)
    H = np.sum(Y * Y, 1).reshape(n_samples_Y, 1)
    Q = np.tile(G, (1, n_samples_Y))
    R = np.tile(H.T, (n_samples_X, 1))
    H = Q + R - 2 * np.dot(X, Y.T)

    return np.exp(-H / 2 / (width ** 2))

def _rbf_dot_XX(X, width):
    """rbf dot, in special case with X dot X"""
    G = np.sum(X * X, axis=1)
    H = G[None, :] + G[:, None] - 2 * np.dot(X, X.T)
    return np.exp(-H / 2 / (width ** 2))

def get_gram_matrix(X, width):
    """Get the centered gram matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    width : float
        The bandwidth parameter.

    Returns
    -------
    K, Kc : array
        the centered gram matrices.
    """
    n = X.shape[0]

    K = _rbf_dot_XX(X, width)
    K_colsums = K.sum(axis=0)
    K_rowsums = K.sum(axis=1)
    K_allsum = K_rowsums.sum()
    Kc = K - (K_colsums[None, :] + K_rowsums[:, None]) / n + (K_allsum / n ** 2)
    # equivalent to H @ K @ H, where H = np.eye(n) - 1 / n * np.ones((n, n)).
    return K, Kc


def hsic_teststat(Kc, Lc, n):
    """get the HSIC statistic.

    Parameters
    ----------
    K, Kc : array
        the centered gram matrices.

    n : float
        the number of samples.

    Returns
    -------
    float
        the HSIC statistic.
    """
    # test statistic m*HSICb under H1
    return 1 / n * np.sum(Kc.T * Lc)


def hsic_test_gamma(X, Y, bw_method='mdbs'):
    """get the HSIC statistic.

    Parameters
    ----------
    X, Y : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    bw_method : str, optional (default=``mdbs``)
        The method used to calculate the bandwidth of the HSIC.

        * ``mdbs`` : Median distance between samples.
        * ``scott`` : Scott's Rule of Thumb.
        * ``silverman`` : Silverman's Rule of Thumb.

    Returns
    -------
    test_stat : float
        the HSIC statistic.

    p : float
        the HSIC p-value.
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    if bw_method == 'scott':
        width_x = bandwidths.bw_scott(X)
        width_y = bandwidths.bw_scott(Y)
    elif bw_method == 'silverman':
        width_x = bandwidths.bw_silverman(X)
        width_y = bandwidths.bw_silverman(Y)
    # Get kernel width to median distance between points
    else:
        width_x = get_kernel_width(X)
        width_y = get_kernel_width(Y)

    # these are slightly biased estimates of centered gram matrices
    K, Kc = get_gram_matrix(X, width_x)
    L, Lc = get_gram_matrix(Y, width_y)

    # test statistic m*HSICb under H1
    n = X.shape[0]
    test_stat = hsic_teststat(Kc, Lc, n)

    var = (1 / 6 * Kc * Lc) ** 2
    # second subtracted term is bias correction
    var = 1 / n / (n - 1) * (np.sum(var) - np.trace(var))
    # variance under H0
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    K[np.diag_indices(n)] = 0
    L[np.diag_indices(n)] = 0
    mu_X = 1 / n / (n - 1) * K.sum()
    mu_Y = 1 / n / (n - 1) * L.sum()
    # mean under H0
    mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

    alpha = mean ** 2 / var
    # threshold for hsicArr*m
    beta = var * n / mean
    p = 1 - gamma.cdf(test_stat, alpha, scale=beta)

    return test_stat, p


if __name__ == '__main__':
    X = np.random.uniform(0, 1, (15000,))
    Y = X ** 2 + np.random.uniform(0, 1, (15000,))
    tic = time.time()
    test_stat, p = hsic_test_gamma(X, Y)
    print(f'now used: {time.time() - tic: .5f}s')

    from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma as hsic_test_gamma_old
    tic = time.time()
    test_stat_old, p_old = hsic_test_gamma_old(X, Y)
    print(f'originally used: {time.time() - tic: .5f}s')

    assert np.isclose(test_stat, test_stat_old)
    assert np.isclose(p, p_old)
    print('equivalent test passed.')

    '''
    now used:  6.78904s
    originally used:  65.28648s
    equivalent test passed.
    '''