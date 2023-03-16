"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import itertools
import numbers
import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import pearsonr, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample

from .bootstrap import BootstrapResult
from .hsic import (get_gram_matrix, get_kernel_width, hsic_test_gamma,
                   hsic_teststat)
from .utils import predict_adaptive_lasso


class RCD():
    """Implementation of RCD Algorithm [1]_

    References
    ----------
    .. [1] T.N.Maeda and S.Shimizu. RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent confounders.
       In Proc. 23rd International Conference on Artificial Intelligence and Statistics (AISTATS2020), Palermo, Sicily, Italy. PMLR  108:735-745, 2020.
    """

    def __init__(self, max_explanatory_num=2, cor_alpha=0.01, ind_alpha=0.01, shapiro_alpha=0.01, MLHSICR=False,
                 bw_method='mdbs'):
        """Construct a RCD model.

           Parameters
           ----------
            max_explanatory_num : int, optional (default=2)
                Maximum number of explanatory variables.
            cor_alpha : float, optional (default=0.01)
                Alpha level for pearson correlation.
            ind_alpha : float, optional (default=0.01)
                Alpha level for HSIC.
            shapiro_alpha : float, optional (default=0.01)
                Alpha level for Shapiro-Wilk test.
            MLHSICR : bool, optional (default=False)
                If True, use MLHSICR for multiple regression, if False, use OLS for multiple regression.
            bw_method : str, optional (default=``mdbs``)
                    The method used to calculate the bandwidth of the HSIC.

                * ``mdbs`` : Median distance between samples.
                * ``scott`` : Scott's Rule of Thumb.
                * ``silverman`` : Silverman's Rule of Thumb.
        """
        # Check parameters
        if max_explanatory_num <= 0:
            raise ValueError('max_explanatory_num must be > 0.')

        if cor_alpha < 0:
            raise ValueError('cor_alpha must be >= 0.')

        if ind_alpha < 0:
            raise ValueError('ind_alpha must be >= 0.')

        if shapiro_alpha < 0:
            raise ValueError('shapiro_alpha must be >= 0.')

        if bw_method not in ('mdbs', 'scott', 'silverman'):
            raise ValueError(
                "bw_method must be 'mdbs', 'scott' or 'silverman'.")

        self._max_explanatory_num = max_explanatory_num
        self._cor_alpha = cor_alpha
        self._ind_alpha = ind_alpha
        self._shapiro_alpha = shapiro_alpha
        self._MLHSICR = MLHSICR
        self._bw_method = bw_method
        self._ancestors_list = None
        self._adjacency_matrix = None
        # DEBAG: self._adjacency_matrix_no_lc = None

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X)

        # Extract a set of ancestors of each variable
        M = self._extract_ancestors(X)

        # Extract parents (direct causes) from the set of ancestors.
        P = self._extract_parents(X, M)

        # Find the pairs of variables affected by the same latent confounders.
        C = self._extract_vars_sharing_confounders(X, P)

        self._ancestors_list = M
        return self._estimate_adjacency_matrix(X, P, C)

    def _get_common_ancestors(self, M, U):
        """Get the set of common ancestors of U"""
        Mj_list = [M[xj] for xj in U]
        return set.intersection(*Mj_list)

    def _get_resid_and_coef(self, X, endog_idx, exog_idcs):
        """Get the residuals and coefficients of the ordinary least squares method"""
        lr = LinearRegression()
        lr.fit(X[:, exog_idcs], X[:, endog_idx])
        resid = X[:, endog_idx] - lr.predict(X[:, exog_idcs])
        return resid, lr.coef_

    def _get_residual_matrix(self, X, U, H_U):
        if len(H_U) == 0:
            return X

        Y = np.zeros_like(X)
        for xj in U:
            Y[:, xj], _ = self._get_resid_and_coef(X, xj, list(H_U))
        return Y

    def _is_non_gaussianity(self, Y, U):
        """Test whether a variable is generated from a non-Gaussian process using the Shapiro-Wilk test"""
        for xj in U:
            if shapiro(Y[:, xj])[1] > self._shapiro_alpha:
                return False
        return True

    def _is_correlated(self, a, b):
        """Estimate that the two variables are linearly correlated using the Pearson's correlation"""
        return pearsonr(a, b)[1] < self._cor_alpha

    def _exists_ancestor_in_U(self, M, U, xi, xj_list):
        # Check xi is not in Mj, the ancestor of xj.
        for xj in xj_list:
            if xi in M[xj]:
                return True

        # Check if xj_list is a subset of Mi, the ancestor of xi.
        if set(xj_list) == set(xj_list) & M[xi]:
            return True
        return False

    def _is_independent(self, X, Y):
        _, p = hsic_test_gamma(X, Y, bw_method=self._bw_method)
        return p > self._ind_alpha

    def _get_resid_and_coef_by_MLHSICR(self, Y, xi, xj_list):
        """Get the residuals and coefficients by minimizing the sum of HSICs using the L-BFGS method."""
        n_samples = Y.shape[0]
        width_list = []
        Lc_list = []
        for xj in xj_list:
            yj = np.reshape(Y[:, xj], [n_samples, 1])
            width_xj = get_kernel_width(yj)
            _, Lc = get_gram_matrix(yj, width_xj)

            width_list.append(width_xj)
            Lc_list.append(Lc)

        _, initial_coef = self._get_resid_and_coef(Y, xi, xj_list)
        width_xi = get_kernel_width(np.reshape(Y[:, xi], [n_samples, 1]))

        # Calculate the sum of the Hilbert-Schmidt independence criterion
        def sum_empirical_hsic(coef):
            resid = Y[:, xi]
            width = width_xi
            for j, xj in enumerate(xj_list):
                resid = resid - coef[j] * Y[:, xj]
                width = width - coef[j] * width_list[j]
            _, Kc = get_gram_matrix(np.reshape(resid, [n_samples, 1]), width)

            objective = 0.0
            for j, xj in enumerate(xj_list):
                objective += hsic_teststat(Kc, Lc_list[j], n_samples)
            return objective

        # Estimate coefficients by minimizing the sum of HSICs using the L-BFGS method.
        coefs, _, _ = fmin_l_bfgs_b(
            func=sum_empirical_hsic, x0=initial_coef, approx_grad=True)

        resid = Y[:, xi]
        for j, xj in enumerate(xj_list):
            resid = resid - coefs[j] * Y[:, xj]
        return resid, coefs

    def _is_independent_of_resid(self, Y, xi, xj_list):
        """Check whether the residuals obtained from multiple regressions are independent"""
        n_samples = Y.shape[0]

        # Multiple Regression with OLS.
        is_all_independent = True
        resid, _ = self._get_resid_and_coef(Y, xi, xj_list)
        for xj in xj_list:
            if not self._is_independent(np.reshape(resid, [n_samples, 1]), np.reshape(Y[:, xj], [n_samples, 1])):
                is_all_independent = False
                break

        if is_all_independent:
            return True
        elif len(xj_list) == 1 or self._MLHSICR == False:
            return False

        # Multiple Regression with MLHSICR.
        resid, _ = self._get_resid_and_coef_by_MLHSICR(Y, xi, xj_list)
        for xj in xj_list:
            if not self._is_independent(np.reshape(resid, [n_samples, 1]), np.reshape(Y[:, xj], [n_samples, 1])):
                return False
        return True

    def _extract_ancestors(self, X):
        """Extract a set of ancestors of each variable"""
        n_features = X.shape[1]
        M = [set() for i in range(n_features)]
        l = 1
        hu_history = {}

        while (True):
            changed = False
            U_list = itertools.combinations(range(n_features), l + 1)
            for U in U_list:
                U = list(U)
                U.sort()

                # Get the set of common ancestors of U
                H_U = self._get_common_ancestors(M, U)

                if tuple(U) in hu_history and H_U == hu_history[tuple(U)]:
                    continue

                Y = self._get_residual_matrix(X, U, H_U)

                # Test whether a variable is generated from a non-Gaussian process using the Shapiro-Wilk test
                if not self._is_non_gaussianity(Y, U):
                    continue

                # Estimate that the two variables are linearly correlated using the Pearson's correlation
                is_cor = True
                for xi, xj in itertools.combinations(U, 2):
                    if not self._is_correlated(Y[:, xi], Y[:, xj]):
                        is_cor = False
                        break
                if not is_cor:
                    continue

                sink_set = []
                for xi in U:
                    xj_list = list(set(U) - set([xi]))
                    if self._exists_ancestor_in_U(M, U, xi, xj_list):
                        continue

                    # Check whether the residuals obtained from multiple regressions are independent
                    if self._is_independent_of_resid(Y, xi, xj_list):
                        sink_set.append(xi)

                if len(sink_set) == 1:
                    xi = sink_set[0]
                    xj_list = list(set(U) - set(sink_set))

                    if not M[xi] == M[xi] | set(xj_list):
                        M[xi] = M[xi] | set(xj_list)
                        changed = True

                hu_history[tuple(U)] = H_U

            if changed:
                l = 1
            elif l < self._max_explanatory_num:
                l += 1
            else:
                break

        return M

    def _is_parent(self, X, M, xj, xi):
        if len(M[xi] - set([xj])) > 0:
            zi, _ = self._get_resid_and_coef(X, xi, list(M[xi] - set([xj])))
        else:
            zi = X[:, xi]

        if len(M[xi] & M[xj]) > 0:
            wj, _ = self._get_resid_and_coef(X, xj, list(M[xi] & M[xj]))
        else:
            wj = X[:, xj]

        # Check if zi and wj are correlated
        return self._is_correlated(wj, zi)

    def _extract_parents(self, X, M):
        """Extract parents (direct causes) from a set of ancestors"""
        n_features = X.shape[1]
        P = [set() for i in range(n_features)]

        for xi in range(n_features):
            for xj in M[xi]:
                # Check if xj is the parent of xi
                if self._is_parent(X, M, xj, xi):
                    P[xi].add(xj)

        return P

    def _get_resid_to_parent(self, X, idx, P):
        if len(P[idx]) == 0:
            return X[:, idx]

        resid, _ = self._get_resid_and_coef(X, idx, list(P[idx]))
        return resid

    def _extract_vars_sharing_confounders(self, X, P):
        """Find the pairs of variables affected by the same latent confounders."""
        n_features = X.shape[1]
        C = [set() for i in range(n_features)]

        for i, j in itertools.combinations(range(n_features), 2):
            if (i in P[j]) or (j in P[i]):
                continue
            resid_xi = self._get_resid_to_parent(X, i, P)
            resid_xj = self._get_resid_to_parent(X, j, P)
            if self._is_correlated(resid_xi, resid_xj):
                C[i].add(j)
                C[j].add(i)

        return C

    def _estimate_adjacency_matrix(self, X, P, C):
        """Estimate adjacency matrix by causal parents and confounders.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parents
        n_features = X.shape[1]
        B = np.zeros([n_features, n_features], dtype='float64')
        for xi in range(n_features):
            xj_list = list(P[xi])
            xj_list.sort()
            if len(xj_list) == 0:
                continue

            _, coef = self._get_resid_and_coef(X, xi, xj_list)
            for j, xj in enumerate(xj_list):
                B[xi, xj] = coef[j]

        # DEBAG: self._adjacency_matrix_no_lc = B.copy()

        # Check confounders
        for xi in range(n_features):
            xj_list = list(C[xi])
            xj_list.sort()
            if len(xj_list) == 0:
                continue

            for xj in xj_list:
                B[xi, xj] = np.nan

        self._adjacency_matrix = B
        return self

    def estimate_total_effect(self, X, from_index, to_index):
        # Check parameters
        X = check_array(X)

        # Check from/to ancestors
        if to_index in self._ancestors_list[from_index]:
            warnings.warn(f'The estimated causal effect may be incorrect because '
                          f'the causal order of the destination variable (to_index={to_index}) '
                          f'is earlier than the source variable (from_index={from_index}).')

        # Check confounders
        if True in np.isnan(self._adjacency_matrix[from_index]):
            warnings.warn(f'The estimated causal effect may be incorrect because '
                          f'the source variable (from_index={from_index}) is influenced by confounders.')
            return np.nan

        # from_index + parents indices
        parents = np.where(np.abs(self._adjacency_matrix[from_index]) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        coefs = predict_adaptive_lasso(X, predictors, to_index)

        return coefs[0]

    def get_error_independence_p_values(self, X):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        # Check parameters
        X = check_array(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        E = X - np.dot(self._adjacency_matrix, X.T).T
        nan_cols = list(
            set(np.argwhere(np.isnan(self._adjacency_matrix)).ravel()))
        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            if i in nan_cols or j in nan_cols:
                p_values[i, j] = np.nan
                p_values[j, i] = np.nan
            else:
                _, p_value = hsic_test_gamma(np.reshape(E[:, i], [n_samples, 1]),
                                             np.reshape(E[:, j], [n_samples, 1]))
                p_values[i, j] = p_value
                p_values[j, i] = p_value

        return p_values

    @property
    def ancestors_list_(self):
        """Estimated ancestors list.

        Returns
        -------
        ancestors_list_ : array-like, shape (n_features)
            The list of causal ancestors sets, where
            n_features is the number of features.
        """
        return self._ancestors_list

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features, n_features)
            The adjacency matrix B of fitted model, where
            n_features is the number of features.
            Set np.nan if order is unknown.
        """
        return self._adjacency_matrix

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        for i in range(n_sampling):
            self.fit(resample(X, replace=False,
                              n_samples=X.shape[0] - n_sampling))
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for to, ancestors in enumerate(self._ancestors_list):
                for from_ in ancestors:
                    total_effects[i, to, from_] = self.estimate_total_effect(
                        X, from_, to)

        return BootstrapResult(adjacency_matrices, total_effects)
