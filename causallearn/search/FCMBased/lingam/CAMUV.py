# -*- coding: utf-8 -*-

import copy
import itertools
import pickle

import numpy as np
from pygam import LinearGAM
from scipy.stats import gamma
from scipy.stats import scoreatpercentile as sap

from causallearn.search.FCMBased.lingam import hsic2

# Usage
#
# execute(X, alpha, num_explanatory_vals) is the method to infer causal graphs.
#
# X: matrix
# alpha: the alpha level for independence testing
# num_explanatory_vals: the maximum number of variables to infer causal relationships. This is equivalent to d in the paper.
#
# Return
# P: P[i] contains the indices of the parents of Xi
# U: The indices of variable pairs having UCPs or UBPs
#
# Before using this, please install numpy and pygam.


def do_an_experiment(data_file, alpha):
    with open(data_file, 'rb') as f:
        obj = pickle.load(f)
    X = obj["data"]
    maxnum_vals = 3
    P, C = execute(X, alpha, maxnum_vals)
    return create_data_for_evaluation(P, C)


def get_neighborhoods(X, alpha):
    n = X.shape[0]
    d = X.shape[1]
    N = [set() for i in range(d)]
    for i in range(d):
        for j in range(d)[i + 1:]:
            independence = hsic2.hsic_gam(X=np.reshape(X[:, i], [n, 1]), Y=np.reshape(X[:, j], [n, 1]), mode="pvalue")
            if independence < alpha:
                N[i].add(j)
                N[j].add(i)
    return N


def find_parents(X, alpha, maxnum_vals, N):
    n = X.shape[0]
    d = X.shape[1]
    P = [set() for i in range(d)]  # Parents
    t = 2
    Y = copy.deepcopy(X)

    while (True):
        changed = False
        variables_set_list = list(itertools.combinations(set(range(d)), t))
        for variables_set in variables_set_list:
            variables_set = set(variables_set)

            if not check_identified_causality(variables_set, P):
                continue

            child, independence_with_K = get_child(X, variables_set, P, N, Y, alpha)
            if not independence_with_K > alpha:
                continue

            parents = variables_set - {child}
            if not check_independence_withou_K(parents, child, P, N, Y, alpha):
                continue

            for parent in parents:
                P[child].add(parent)
                changed = True
                Y = get_residuals_matrix(X, Y, P, child)

        if changed:
            t = 2
        else:
            t += 1
            if t > maxnum_vals:
                break

    for i in range(d):
        non_parents = set()
        for j in P[i]:
            residual_i = get_residual(X, i, P[i] - {j})
            residual_j = get_residual(X, j, P[j])
            independence = hsic2.hsic_gam(X=np.reshape(residual_i, [n, 1]), Y=np.reshape(residual_j, [n, 1]),
                                          mode="pvalue")
            if independence > alpha:
                non_parents.add(j)
        P[i] = P[i] - non_parents

    return P


def get_residuals_matrix(X, Y_old, P, child):
    n = X.shape[0]
    d = X.shape[1]

    Y = copy.deepcopy(Y_old)
    Y[:, child] = get_residual(X, child, P[child])
    return Y


def get_child(X, variables_set, P, N, Y, alpha):
    n = X.shape[0]
    d = X.shape[1]

    max_independence = 0.0
    max_independence_child = None

    for child in variables_set:
        parents = variables_set - {child}

        if not check_correlation(child, parents, N):
            continue

        residual = get_residual(X, child, parents | P[child])
        independence = hsic2.hsic_gam(X=np.reshape(residual, [n, 1]),
                                      Y=np.reshape(Y[:, list(parents)], [n, len(parents)]), mode="pvalue")
        if max_independence < independence:
            max_independence = independence
            max_independence_child = child

    return max_independence_child, max_independence


def check_independence_withou_K(parents, child, P, N, Y, alpha):
    n = Y.shape[0]
    for parent in parents:
        independence = hsic2.hsic_gam(X=np.reshape(Y[:, child], [n, 1]), Y=np.reshape(Y[:, parent], [n, 1]),
                                      mode="pvalue")
        if alpha < independence:
            return False
    return True


def check_identified_causality(variables_set, P):
    variables_list = list(variables_set)
    for i in variables_list:
        for j in variables_list[variables_list.index(i) + 1:]:
            if (j in P[i]) or (i in P[j]):
                return False
    return True


def check_correlation(child, parents, N):
    for parent in parents:
        if not parent in N[child]:
            return False
    return True


def execute(X, alpha, num_explanatory_vals):
    n = X.shape[0]
    d = X.shape[1]
    N = get_neighborhoods(X, alpha)
    P = find_parents(X, alpha, num_explanatory_vals, N)

    U = []

    for i in range(d):
        for j in range(d)[i + 1:]:
            if (i in P[j]) or (j in P[i]):
                continue
            if (not i in N[j]) or (not j in N[i]):
                continue
            i_residual = get_residual(X, i, P[i])
            j_residual = get_residual(X, j, P[j])
            independence = hsic2.hsic_gam(X=np.reshape(i_residual, [n, 1]), Y=np.reshape(j_residual, [n, 1]),
                                          mode="pvalue")
            if independence < alpha:
                if not set([i, j]) in U:
                    U.append(set([i, j]))

    return P, U


def get_residual(X, explained_i, explanatory_ids):
    n = X.shape[0]

    explanatory_ids = list(explanatory_ids)

    if len(explanatory_ids) == 0:
        residual = X[:, explained_i]

    else:
        gam = LinearGAM().fit(X[:, explanatory_ids], X[:, explained_i])

        residual = X[:, explained_i] - gam.predict(X[:, explanatory_ids])

    return residual


def create_data_for_evaluation(P, C):
    causal_pairs = []

    for i1 in range(len(P)):
        for i2 in P[i1]:
            causal_pairs.append([i2, i1])

    return causal_pairs, C


##################################################################################
# NOTE: Functions below are for CAMUV.py

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
