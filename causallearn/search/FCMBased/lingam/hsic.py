"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

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
    Q = np.tile(G, (1, n_samples))
    R = np.tile(G.T, (n_samples, 1))

    dists = Q + R - 2 * np.dot(X_med, X_med.T)
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
    H = np.eye(n) - 1 / n * np.ones((n, n))

    K = _rbf_dot(X, X, width)
    Kc = np.dot(np.dot(H, K), H)

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
    return 1 / n * np.sum(np.sum(Kc.T * Lc))


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
    bone = np.ones((n, 1))
    test_stat = hsic_teststat(Kc, Lc, n)

    var = (1 / 6 * Kc * Lc) ** 2
    # second subtracted term is bias correction
    var = 1 / n / (n - 1) * (np.sum(np.sum(var)) - np.sum(np.diag(var)))
    # variance under H0
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))
    mu_X = 1 / n / (n - 1) * np.dot(bone.T, np.dot(K, bone))
    mu_Y = 1 / n / (n - 1) * np.dot(bone.T, np.dot(L, bone))
    # mean under H0
    mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

    alpha = mean ** 2 / var
    # threshold for hsicArr*m
    beta = np.dot(var, n) / mean
    p = 1 - gamma.cdf(test_stat, alpha, scale=beta)[0][0]

    return test_stat, p
