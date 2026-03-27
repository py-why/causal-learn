import math
from typing import Any, Dict, List, Tuple

import pandas as pd
from causallearn.utils.ScoreUtils import *
from numpy import ndarray


def local_score_BIC(Data: ndarray, i: int, PAi: List[int], parameters=None) -> float:
    """
    Local BIC score for the linear Gaussian case (higher is better).

    Background
    ----------
    The Bayesian Information Criterion (BIC) is derived from the Laplace approximation
    to the marginal likelihood (Schwarz, 1978). For a DAG model G with data D of n
    samples, the BIC approximation to the log marginal likelihood is (Chickering, 2002):

        S_BIC(G, D) = log p(D | θ_hat, G) - (d / 2) * log(n)

    where θ_hat is the MLE, and d is the number of free parameters.

    Derivation for linear Gaussian case
    ------------------------------------
    For node X_i with parents Pa_i, assume X_i = B^T * Pa_i + ε, ε ~ N(0, σ²).
    The maximized local log-likelihood is:

        log p(D_i | θ_hat) = -n/2 * log(2π) - n/2 * log(σ̂²) - n/2

    where σ̂² = Σ_{ii} - Σ_{i,Pa} * Σ_{Pa,Pa}^{-1} * Σ_{Pa,i} is the MLE residual
    variance (computed from the sample covariance matrix with ddof=0).

    The number of free parameters is |Pa_i| + 1 (regression weights + variance).
    Dropping the constant -n/2 * log(2π) which doesn't affect model comparison:

        score(i, Pa_i) = -n/2 * (1 + log(σ̂²)) - λ * (|Pa_i| + 1) * log(n)

    Score convention
    ----------------
    Higher score = better model. The score is always negative; closer to 0 is better.

    The hyperparameter λ controls the sparsity of the learned graph:
      - Default λ = 0.5 (standard BIC: 0.5 * log(n) penalty per free parameter)
      - Larger λ → stronger penalty → sparser graph (fewer edges)
      - Smaller λ → weaker penalty → denser graph (more edges)

    Relation to Juan Gamella's GES implementation (ges package):
      Juan uses lmbda = 0.5 * log(n) as a single merged penalty, i.e.
          penalty_juan = lmbda_juan * (|Pa| + 1)  where lmbda_juan = 0.5 * log(n)
      Here we separate λ and log(n):
          penalty_ours = λ * (|Pa| + 1) * log(n)
      So our default λ = 0.5 is equivalent to Juan's default.

    Parameters
    ----------
    Data : ndarray, shape (n_samples, n_features)
        The input data matrix.
    i : int
        Target variable index.
    PAi : list of int
        Parent variable indices.
    parameters : dict, optional
        'lambda_value': penalty coefficient (default 0.5).

    Returns
    -------
    score : float
        Local BIC score (higher is better).
    """

    cov = np.cov(Data.T, ddof=0)
    n = Data.shape[0]

    if parameters is None:
        lambda_value = 0.5
    else:
        lambda_value = parameters["lambda_value"]

    sigma = cov[i, i]
    if len(PAi) > 0:
        yX = cov[np.ix_([i], PAi)]
        XX = cov[np.ix_(PAi, PAi)]
        try:
            XX_inv = np.linalg.inv(XX)
        except np.linalg.LinAlgError:
            XX_inv = np.linalg.pinv(XX)
        sigma = float(cov[i, i] - yX @ XX_inv @ yX.T)

    if sigma <= 0:
        sigma = np.finfo(float).eps

    likelihood = -0.5 * n * (1 + np.log(sigma))
    penalty = lambda_value * (len(PAi) + 1) * np.log(n)
    return likelihood - penalty


def local_score_BIC_from_cov(
    Data: Tuple[ndarray, int], i: int, PAi: List[int], parameters=None
) -> float:
    """
    Local BIC score from pre-computed covariance matrix (higher is better).

    Same formula as local_score_BIC, but takes (cov, n) instead of raw data.
    See local_score_BIC for detailed derivation and parameter description.

        score = -0.5 * n * (1 + log(σ̂²)) - λ * (|Pa| + 1) * log(n)

    Parameters
    ----------
    Data : tuple (ndarray, int)
        (covariance matrix with shape (p, p), sample size n).
    i : int
        Target variable index.
    PAi : list of int
        Parent variable indices.
    parameters : dict, optional
        'lambda_value': penalty coefficient (default 0.5).

    Returns
    -------
    score : float
        Local BIC score (higher is better).
    """

    cov, n = Data

    if parameters is None:
        lambda_value = 0.5
    else:
        lambda_value = parameters["lambda_value"]

    sigma = cov[i, i]
    if len(PAi) > 0:
        yX = cov[np.ix_([i], PAi)]
        XX = cov[np.ix_(PAi, PAi)]
        try:
            XX_inv = np.linalg.inv(XX)
        except np.linalg.LinAlgError:
            XX_inv = np.linalg.pinv(XX)
        sigma = float(cov[i, i] - yX @ XX_inv @ yX.T)

    if sigma <= 0:
        sigma = np.finfo(float).eps

    likelihood = -0.5 * n * (1 + np.log(sigma))
    penalty = lambda_value * (len(PAi) + 1) * np.log(n)
    return likelihood - penalty


def local_score_BDeu(Data: ndarray, i: int, PAi: List[int], parameters=None) -> float:
    """
    Calculate the *negative* local score with BDeu for the discrete case

    Parameters
    ----------
    Data: (sample, features)
    i: current index
    PAi: parent indexes
    parameters:
                 sample_prior: sample prior
                 structure_prior: structure prior
                 r_i_map: number of states of the finite random variable X_{i}

    Returns
    -------
    score: local BDeu score
    """
    if parameters is None:
        sample_prior = 1  # default sample_prior = 1
        structure_prior = 1  # default structure_prior = 1
        r_i_map = {
            i: len(np.unique(np.asarray(Data[:, i]))) for i in range(Data.shape[1])
        }
    else:
        sample_prior = parameters["sample_prior"]
        structure_prior = parameters["structure_prior"]
        r_i_map = parameters["r_i_map"]

    # calculate q_{i}
    q_i = 1
    for pa in PAi:
        q_i *= r_i_map[pa]

    xi_col = "x{}".format(i)

    if len(PAi) != 0:
        # calculate N_{ij} and N_{ijk} using numpy for speed and pandas compatibility
        names = ["x{}".format(k) for k in range(Data.shape[1])]
        Data_pd = pd.DataFrame(Data, columns=names)
        parent_names = ["x{}".format(k) for k in PAi]
        Data_pd_group_Nij = Data_pd.groupby(parent_names)

        Nij_map = Data_pd_group_Nij.size().to_dict()
        Nij_map_keys_list = list(Nij_map.keys())

        # calculate N_{ijk}: for each parent config, count occurrences of each X_i value
        Nijk_map = {}
        for ij in Nij_map_keys_list:
            group = Data_pd_group_Nij.get_group((ij,) if not isinstance(ij, tuple) else ij)
            counts = group[xi_col].value_counts().reset_index()
            counts.columns = [xi_col, "times"]
            Nijk_map[ij] = counts
    else:
        # No parents: N_{ij} is just the total count
        Nij_map = {"": Data.shape[0]}
        Nij_map_keys_list = [""]

        # N_{ijk}: count occurrences of each X_i value
        xi_vals, xi_counts = np.unique(Data[:, i], return_counts=True)
        Nijk_map = {"": pd.DataFrame({xi_col: xi_vals, "times": xi_counts})}

    BDeu_score = 0
    # first term
    vm = Data.shape[1] - 1
    BDeu_score += len(PAi) * np.log(structure_prior / vm) + (vm - len(PAi)) * np.log(
        1 - (structure_prior / vm)
    )

    # second term
    for pa in range(len(Nij_map_keys_list)):
        Nij = Nij_map.get(Nij_map_keys_list[pa])
        first_term = math.lgamma(sample_prior / q_i) - math.lgamma(
            Nij + sample_prior / q_i
        )

        second_term = 0
        Nijk_list = Nijk_map.get(Nij_map_keys_list[pa])["times"].to_numpy()
        for Nijk in Nijk_list:
            second_term += math.lgamma(
                Nijk + sample_prior / (r_i_map[i] * q_i)
            ) - math.lgamma(sample_prior / (r_i_map[i] * q_i))

        BDeu_score += first_term + second_term

    return BDeu_score


def local_score_cv_general(
    Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                   kfold: k-fold cross validation
                   lambda: regularization parameter

    Returns
    -------
    score: local score
    """

    Data = Data
    PAi = list(PAi)

    T = Data.shape[0]
    X = Data[:, Xi].reshape(-1, 1)
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    n0 = math.floor(T / k)
    gamma = 0.01
    Thresh = 1e-5

    if len(PAi):
        PA = Data[:, PAi]

        # set the kernel for X
        GX = np.multiply(X, X).reshape(-1, 1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X @ X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
        width = width * 2
        theta = 1 / (width**2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = (
            np.eye(T) - np.ones((T, T)) / T
        )  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T)/2, np.min([400, math.floor(T/2)]), evals_only=False)   # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # eig_Kx = eig_Kx[IIx]
        # eix = eix[:, IIx]
        # mx = len(IIx)

        # set the kernel for PA
        Kpa = np.ones((T, T))

        for m in range(PA.shape[1]):
            G = np.multiply(PA[:, [m]], PA[:, [m]]).reshape(-1, 1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, [m]] @ PA[:, [m]].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
            width = width * 2
            theta = 1 / (width**2)
            Kpa = np.multiply(
                Kpa,
                kernel(PA[:, m].reshape(-1, 1), PA[:, m].reshape(-1, 1), (theta, 1))[0],
            )

        H0 = (
            np.eye(T) - np.ones((T, T)) / T
        )  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0  # kernel matrix for PA

        CV = 0
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0:T, n0:T]
                Kx_tr_te = Kx[n0:T, 0:n0]
                Kpa_te = Kpa[0:n0, 0:n0]
                Kpa_tr = Kpa[n0:T, n0:T]
                Kpa_tr_te = Kpa[n0:T, 0:n0]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                Kpa_te = Kpa[kk * n0 : T, kk * n0 : T]
                Kpa_tr = Kpa[0 : kk * n0, 0 : kk * n0]
                Kpa_tr_te = Kpa[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif kk < k - 1 and kk > 0:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                Kpa_te = Kpa[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kpa_tr = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kpa_tr_te = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.eye(n1))
            tmp2 = tmp1 @ Kx_tr @ tmp1
            tmp3 = tmp1 @ pdinv(np.eye(n1) + n1 * var_lambda**2 / gamma * tmp2) @ tmp1
            A = (
                Kx_te
                + Kpa_tr_te.T @ tmp2 @ Kpa_tr_te
                - 2 * Kx_tr_te.T @ tmp1 @ Kpa_tr_te
                - n1 * var_lambda**2 / gamma * Kx_tr_te.T @ tmp3 @ Kx_tr_te
                - n1
                * var_lambda**2
                / gamma
                * Kpa_tr_te.T
                @ tmp1
                @ Kx_tr
                @ tmp3
                @ Kx_tr
                @ tmp1
                @ Kpa_tr_te
                + 2
                * n1
                * var_lambda**2
                / gamma
                * Kx_tr_te.T
                @ tmp3
                @ Kx_tr
                @ tmp1
                @ Kpa_tr_te
            ) / gamma

            B = n1 * var_lambda**2 / gamma * tmp2 + np.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1).reshape(-1, 1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
        width = width * 2
        theta = 1 / (width**2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.eye(T) - np.ones((T, T)) / (
            T
        )  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 2)]), evals_only=False)  # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # mx = len(IIx)

        CV = 0
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0 : T, (kk + 1) * n0 : T]
                Kx_tr_te = Kx[(kk + 1) * n0 : T, kk * n0 : (kk + 1) * n0]
                nv = n0
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            A = (
                Kx_te
                - 1
                / (gamma * n1)
                * Kx_tr_te.T
                @ pdinv(np.eye(n1) + 1 / (gamma * n1) * Kx_tr)
                @ Kx_tr_te
            ) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = -CV  # cross-validated log-likelihood (higher is better)
    return score


def local_score_cv_multi(
    Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS
    for variables with multi-variate dimensions

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                  kfold: k-fold cross validation
                  lambda: regularization parameter
                  dlabel: for variables with multi-dimensions,
                                   indicate which dimensions belong to the i-th variable.

    Returns
    -------
    score: local score
    """

    T = Data.shape[0]
    X = Data[:, parameters["dlabel"][Xi]].reshape(-1, 1)
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    n0 = math.floor(T / k)
    gamma = 0.01
    Thresh = 1e-5

    if len(PAi):
        # set the kernel for X
        GX = np.multiply(X, X).reshape(-1, 1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
        width = width * 3  ###
        theta = 1 / (width**2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.eye(T) - np.ones((T, T)) / (
            T
        )  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # set the kernel for PA
        Kpa = np.ones((T, T), dtype=np.float64)

        for m in range(len(PAi)):
            PA = Data[:, parameters["dlabel"][PAi[m]]].reshape(-1, 1)
            G = np.multiply(PA, PA).reshape(-1, 1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
            width = width * 3  ###
            theta = 1 / (width**2 * PA.shape[1])
            Kpa = np.multiply(Kpa, kernel(PA, PA, (theta, 1))[0])

        H0 = np.eye(T) - np.ones((T, T)) / (
            T
        )  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0  # kernel matrix for PA

        CV = 0
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0:T, n0:T]
                Kx_tr_te = Kx[n0:T, 0:n0]
                Kpa_te = Kpa[0:n0, 0:n0]
                Kpa_tr = Kpa[n0:T, n0:T]
                Kpa_tr_te = Kpa[n0:T, 0:n0]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                Kpa_te = Kpa[kk * n0 : T, kk * n0 : T]
                Kpa_tr = Kpa[0 : kk * n0, 0 : kk * n0]
                Kpa_tr_te = Kpa[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                Kpa_te = Kpa[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kpa_tr = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kpa_tr_te = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.eye(n1))
            tmp2 = tmp1 @ Kx_tr @ tmp1
            tmp3 = tmp1 @ pdinv(np.eye(n1) + n1 * var_lambda**2 / gamma * tmp2) @ tmp1
            A = (
                Kx_te
                + Kpa_tr_te.T @ tmp2 @ Kpa_tr_te
                - 2 * Kx_tr_te.T @ tmp1 @ Kpa_tr_te
                - n1 * var_lambda**2 / gamma * Kx_tr_te.T @ tmp3 @ Kx_tr_te
                - n1
                * var_lambda**2
                / gamma
                * Kpa_tr_te.T
                @ tmp1
                @ Kx_tr
                @ tmp3
                @ Kx_tr
                @ tmp1
                @ Kpa_tr_te
                + 2
                * n1
                * var_lambda**2
                / gamma
                * Kx_tr_te.T
                @ tmp3
                @ Kx_tr
                @ tmp1
                @ Kpa_tr_te
            ) / gamma

            B = n1 * var_lambda**2 / gamma * tmp2 + np.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1).reshape(-1, 1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
        width = width * 3  ###
        theta = 1 / (width**2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.eye(T) - np.ones((T, T)) / (
            T
        )  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        CV = 0
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0 : T, (kk + 1) * n0 : T]
                Kx_tr_te = Kx[(kk + 1) * n0 : T, kk * n0 : (kk + 1) * n0]
                nv = n0
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            A = (
                Kx_te
                - 1
                / (gamma * n1)
                * Kx_tr_te.T
                @ pdinv(np.eye(n1) + 1 / (gamma * n1) * Kx_tr)
                @ Kx_tr_te
            ) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = -CV  # cross-validated log-likelihood (higher is better)
    return score


def local_score_marginal_general(
    Data: ndarray, Xi: int, PAi: List[int], parameters=None
) -> float:
    """
    Calculate the local score by negative marginal log-likelihood
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters: None

    Returns
    -------
    score: local score
    """

    T = Data.shape[0]
    X = Data[:, Xi].reshape(-1, 1)
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1).reshape(-1, 1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T**2, 1))
    width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
    width = width * 2.5  # kernel width
    theta = 1 / (width**2)
    H = np.eye(T) - np.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1e-5
    eig_Kx, eix = eigdec(
        (Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False
    )  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if len(PAi):
        PA = Data[:, PAi]

        widthPA = np.empty((PA.shape[1], 1))
        # set the kernel for PA
        for m in range(PA.shape[1]):
            G = np.multiply(PA[:, [m]], PA[:, [m]]).reshape(-1, 1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, [m]] * PA[:, [m]].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            widthPA[m] = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
        widthPA = widthPA * 2.5  # kernel width

        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]], dtype=object)
        logtheta0 = np.vstack([np.log(widthPA), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    else:
        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]], dtype=object)
        PA = np.zeros((T, 1))
        logtheta0 = np.array([[100], [0], [np.log(np.sqrt(0.1))]])
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    score = -nlml  # marginal log-likelihood (higher is better)
    return score


def local_score_marginal_multi(
    Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score by negative marginal log-likelihood
    based on a regression model in RKHS
    for variables with multi-variate dimensions

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                  dlabel: for variables with multi-dimensions,
                                   indicate which dimensions belong to the i-th variable.

    Returns
    -------
    score: local score
    """
    T = Data.shape[0]
    X = Data[:, parameters["dlabel"][Xi]].reshape(-1, 1)
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1).reshape(-1, 1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T**2, 1))
    widthX = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
    widthX = widthX * 2.5  # kernel width
    theta = 1 / (widthX**2)
    H = np.eye(T) - np.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1e-5
    eig_Kx, eix = eigdec(
        (Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False
    )  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if len(PAi):
        widthPA_all = np.empty((1, 0))
        # set the kernel for PA
        PA_all = np.empty((Data.shape[0], 0))
        for m in range(len(PAi)):
            PA = Data[:, parameters["dlabel"][PAi[m]]].reshape(-1, 1)
            PA_all = np.hstack([PA_all, PA])
            G = np.sum((np.multiply(PA, PA)), axis=1).reshape(-1, 1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            widthPA = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)]))
            widthPA_all = np.hstack(
                [
                    widthPA_all,
                    widthPA * np.ones((1, np.size(parameters["dlabel"][PAi[m]]))),
                ]
            )
        widthPA_all = widthPA_all * 2.5  # kernel width
        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]], dtype=object)
        logtheta0 = np.vstack([np.log(widthPA_all.T), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA_all,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA_all,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    else:
        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]], dtype=object)
        PA = np.zeros((T, 1))
        logtheta0 = np.array([[100], [0], [np.log(np.sqrt(0.1))]])
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    score = -nlml  # marginal log-likelihood (higher is better)
    return score
