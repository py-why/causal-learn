"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import itertools
import numbers
import warnings

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.utils import check_array, resample

from .bootstrap import BootstrapResult
from .hsic import hsic_test_gamma
from .utils import predict_adaptive_lasso


class BottomUpParceLiNGAM():
    """Implementation of ParceLiNGAM Algorithm [1]_

    References
    ----------
    .. [1] T. Tashiro, S. Shimizu, and A. Hyvärinen.
       ParceLiNGAM: a causal ordering method robust against latent confounders.
       Neural computation, 26.1: 57-83, 2014.
    """

    def __init__(self, random_state=None, alpha=0.1, regressor=None, prior_knowledge=None):
        """Construct a BottomUpParceLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        alpha : float, optional (default=0.1)
            Significant level of statistical test. If alpha=0.0, rejection does not occur in statistical tests.
        regressor : regressor object implementing 'fit' and 'predict' function (default=None)
            Regressor to compute residuals.
            This regressor object must have ``fit``method and ``predict`` function like scikit-learn's model.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior background_knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior background_knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior background_knowledge is available to know if either of the two cases above (0 or 1) is true.
        """
        # Check parameters
        if regressor is not None:
            if not (hasattr(regressor, 'fit') and hasattr(regressor, 'predict')):
                raise ValueError("'regressor' has no fit or predict method.")

        if alpha < 0.0:
            raise ValueError('alpha must be an float greater than 0.')

        self._random_state = random_state
        self._alpha = alpha
        self._causal_order = None
        self._adjacency_matrix = None
        self._reg = regressor
        self._Aknw = prior_knowledge

        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

            # Extract all partial orders in prior background_knowledge matrix
            self._partial_orders = self._extract_partial_orders(self._Aknw)

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
        self._causal_order = None
        self._adjacency_matrices = None

        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Check prior background_knowledge
        if self._Aknw is not None:
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError(
                    'The shape of prior background_knowledge must be (n_features, n_features)')

        # Center variables for each group
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))

        # bonferroni correction
        thresh_p = self._alpha / (n_features - 1)

        # Search causal orders one by one from the bottom upward
        K_bttm, p_bttm = self._search_causal_order(X, thresh_p)

        U_res = list(np.setdiff1d(np.arange(n_features), K_bttm))
        K = []
        # Add a list of features whose order is unknown.
        if len(U_res) > 1:
            K = [U_res]
        for i in K_bttm:
            K.append(i)

        self._causal_order = K
        self._p_list = p_bttm
        return self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)

    def _extract_partial_orders(self, pk):
        """ Extract partial orders from prior background_knowledge."""
        path_pairs = np.array(np.where(pk == 1)).transpose()
        no_path_pairs = np.array(np.where(pk == 0)).transpose()

        # Check for inconsistencies in pairs with path
        check_pairs = np.concatenate([path_pairs, path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            if len(pairs[counts > 1]) > 0:
                raise ValueError(
                    f'The prior background_knowledge contains inconsistencies at the following indices: {pairs[counts > 1].tolist()}')

        # Check for inconsistencies in pairs without path
        # If there are duplicate pairs without path, they cancel out and are not ordered.
        check_pairs = np.concatenate([no_path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            check_pairs = np.concatenate([no_path_pairs, pairs[counts > 1]])
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            no_path_pairs = pairs[counts < 2]

        check_pairs = np.concatenate([path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) == 0:
            # If no pairs are extracted from the specified prior background_knowledge,
            # discard the prior background_knowledge.
            self._Aknw = None
            return None

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]  # [to, from] -> [from, to]

    def _search_candidate(self, U):
        """ Search for candidate features """
        # If no prior background_knowledge is specified, nothing to do.
        if self._Aknw is None:
            return U

        # Candidate features that are not to the left of the partial orders
        Uc = [i for i in U if i not in self._partial_orders[:, 0]]
        return Uc

    def _search_causal_order(self, X, thresh_p):
        """Search causal orders one by one from the bottom upward."""
        U = np.arange(X.shape[1])
        K_bttm = []
        p_bttm = []
        is_search_causal_order = True

        if len(U) <= 1:
            K_bttm = np.append(U, K_bttm).astype(np.int64)
            p_bttm.insert(0, 0.0)
            is_search_causal_order = False

        while is_search_causal_order:
            # Search for candidate features
            Uc = self._search_candidate(U)

            if len(Uc) == 1:
                # If there is only one variable in Uc,
                # calculate HSIC with the rest of the variables
                m = np.array([Uc[0]])
                predictors = np.setdiff1d(U, Uc[0])
                R = self._compute_residuals(X, predictors, m)
                fisher_p, _ = self._fisher_hsic_test(
                    X[:, predictors], R, np.inf)
            else:
                # Find the most sink variable
                m, _, fisher_p = self._find_exo_vec(X, Uc)

            # Conduct statistical test by the p-value or the statistic
            # If statistical test is not rejected
            if fisher_p >= thresh_p:
                # Add index of the exogenous variable to K_bttm
                K_bttm = np.append(m, K_bttm).astype(np.int64)
                p_bttm.insert(0, fisher_p)

                # Update U and partial orders
                U = U[U != m]
                if self._Aknw is not None:
                    self._partial_orders = self._partial_orders[self._partial_orders[:, 1] != m]

                # If there is only one candidate for sink variable, the search ends
                if len(U) <= 1:
                    K_bttm = np.append(U, K_bttm).astype(np.int64)
                    p_bttm.insert(0, 0.0)
                    is_search_causal_order = False
            # If statistical test is rejected
            else:
                is_search_causal_order = False

        return K_bttm, p_bttm

    def _find_exo_vec(self, X, U):
        """Find the most exogenous vector."""
        max_p = -np.inf
        max_p_stat = np.inf

        exo_vec = []
        for j in range(len(U)):
            xi_index = np.setdiff1d(U, U[j])
            xj_index = np.array([U[j]])

            # Compute residuals
            R = self._compute_residuals(X, xi_index, xj_index)

            # HSIC test with Fisher's method
            fisher_p, fisher_stat = self._fisher_hsic_test(
                X[:, xi_index], R, max_p_stat)

            # Update output
            if fisher_stat < max_p_stat or fisher_p > max_p:
                exo_vec = xi_index
                max_p = fisher_p
                max_p_stat = fisher_stat

        m = np.setdiff1d(U, exo_vec)
        return m, exo_vec, max_p

    def _compute_residuals(self, X, predictors, target):
        """Compute residuals"""
        if self._reg is None:
            # Compute residuals of least square regressions
            cov = np.cov(X.T)
            coef = np.dot(np.linalg.pinv(cov[np.ix_(predictors, predictors)]),
                          cov[np.ix_(target, predictors)].reshape(predictors.shape[0], 1))
            R = X[:, target] - np.dot(X[:, predictors], coef)
        else:
            self._reg.fit(X[:, predictors], np.ravel(X[:, target]))
            R = X[:, target] - \
                self._reg.predict(X[:, predictors]).reshape(-1, 1)

        return R

    def _fisher_hsic_test(self, X, R, max_p_stat):
        """Conduct statistical test by HSIC and Fisher's method."""
        fisher_stat = 0
        n_features = X.shape[1]

        if n_features == 1:
            fisher_stat, fisher_p = hsic_test_gamma(X, R)
        else:
            for i in range(n_features):
                _, hsic_p = hsic_test_gamma(X[:, [i]], R)
                fisher_stat += np.inf if hsic_p == 0 else - 2 * np.log(hsic_p)

                if fisher_stat > max_p_stat:
                    break
            fisher_p = 1 - chi2.cdf(fisher_stat, df=2 * n_features)

        return fisher_p, fisher_stat

    def _flatten(self, arr):
        """Return a copy of an array flattened in one dimension."""
        return [val for item in arr for val in
                (self._flatten(item) if hasattr(item, '__iter__') and not isinstance(item, str) else [item])]

    def _estimate_adjacency_matrix(self, X, prior_knowledge=None):
        """Estimate adjacency matrix by causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
            Prior background_knowledge matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if prior_knowledge is not None:
            pk = prior_knowledge.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype='float64')
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]

            # Flatten the array into one dimension
            predictors = self._flatten(self._causal_order[:i])

            # Exclude variables specified in no_path with prior background_knowledge
            if prior_knowledge is not None:
                predictors = [p for p in predictors if pk[target, p] != 0]

            # target is exogenous variables if predictors are empty
            if len(predictors) == 0:
                continue

            B[target, predictors] = predict_adaptive_lasso(
                X, predictors, target)

        # Set np.nan if order is unknown
        for unk_order in self._causal_order:
            if hasattr(unk_order, '__iter__') and not isinstance(unk_order, str):
                for i in range(len(unk_order) - 1):
                    xi = unk_order[i]
                    for xj in unk_order[i + 1:]:
                        B[xi, xj] = np.nan
                        B[xj, xi] = np.nan

        self._adjacency_matrix = B
        return self

    def estimate_total_effect(self, X, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check parameters
        X = check_array(X)

        # Check from/to causal order
        for i, order in enumerate(self._causal_order):
            if hasattr(order, '__iter__') and from_index in order:
                from_order = i
                break
            elif not hasattr(order, '__iter__') and int(from_index) == int(order):
                from_order = i
                break

        for i, order in enumerate(self._causal_order):
            if hasattr(order, '__iter__') and to_index in order:
                to_order = i
                break
            elif not hasattr(order, '__iter__') and int(to_index) == int(order):
                to_order = i
                break

        if from_order > to_order:
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
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
            Set the features as a list if order is unknown.
        """
        return self._causal_order

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
            self.fit(resample(X))
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1:]:
                    if hasattr(from_, '__iter__'):
                        for from_item in from_:
                            total_effects[i, to, from_item] = self.estimate_total_effect(
                                X, from_item, to)
                    else:
                        total_effects[i, to, from_] = self.estimate_total_effect(
                            X, from_, to)

        return BootstrapResult(adjacency_matrices, total_effects)
