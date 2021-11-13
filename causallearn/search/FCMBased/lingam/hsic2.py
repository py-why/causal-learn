# import autograd.numpy as np
import numpy as np
from scipy.stats import gamma
from scipy.stats import scoreatpercentile as sap


def get_width(X):
    n = X.shape[0]

    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

    return width_x


# def median(lst):
# 	lst_sorted = sorted(lst)
# 	return lst_sorted[(len(lst) - 1) // 2]

def get_mean_width(X):
    n = X.shape[0]

    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.mean(dists[dists > 0]))

    return width_x


def bw_scott(x):
    A = select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)


def bw_silverman(x):
    A = select_sigma(x)
    n = len(x)
    return .9 * A * n ** (-0.2)


def select_sigma(X):
    normalize = 1.349
    IQR = (sap(X, 75) - sap(X, 25)) / normalize
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)


def rbf_dot(pattern1, pattern2, width):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2 * np.dot(pattern1, pattern2.T)

    H = np.exp(-H / 2 / (width ** 2))

    return H


def get_K(X, width_x):
    n = X.shape[0]

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    K = rbf_dot(X, X, width_x)
    Kc = np.dot(np.dot(H, K), H)

    return K, Kc


def hsic_gam(X=None, Y=None, alph=None, width_x=None, width_y=None, K=None, Kc=None, L=None, Lc=None, mode=None,
             kwdth="mdbs"):
    n = X.shape[0]

    if kwdth == "scott":
        width_x = bw_scott(X)
        width_y = bw_scott(Y)
    elif kwdth == "silverman":
        width_x = bw_silverman(X)
        width_y = bw_silverman(Y)
    elif isinstance(kwdth, float):
        width_x = kwdth
        width_y = kwdth
    else:
        if (width_x is None) and ((K is None) or (Kc is None)):
            width_x = get_width(X)
        if (width_y is None) and ((L is None) or (Lc is None)):
            width_y = get_width(Y)

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    if (K is None) or (Kc is None):
        K = rbf_dot(X, X, width_x)
        Kc = np.dot(np.dot(H, K), H)

    if (L is None) or (Lc is None):
        L = rbf_dot(Y, Y, width_y)
        Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    if mode == "testStat":
        return testStat

    varHSIC = (Kc * Lc / 6) ** 2

    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)

    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    if mode == "pvalue":
        p_value = 1 - gamma.cdf(testStat, al, scale=bet)
        return p_value

    thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

    if mode == "testStatMinusThres":
        return (testStat - thresh)

    return (testStat < thresh)
