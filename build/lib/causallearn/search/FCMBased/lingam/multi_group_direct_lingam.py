"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import itertools
import numbers
import warnings

import numpy as np
from sklearn.utils import check_array, resample

from .bootstrap import BootstrapResult
from .direct_lingam import DirectLiNGAM
from .hsic import hsic_test_gamma
from .utils import predict_adaptive_lasso


class MultiGroupDirectLiNGAM(DirectLiNGAM):
    """Implementation of DirectLiNGAM Algorithm with multiple groups [1]_

    References
    ----------
    .. [1] S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. Neurocomputing, 81: 104-107, 2012. 
    """

    def __init__(self, random_state=None, prior_knowledge=None, apply_prior_knowledge_softly=False):
        """Construct a model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior background_knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior background_knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior background_knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior background_knowledge softly.
        """
        super().__init__(random_state, prior_knowledge, apply_prior_knowledge_softly)

    def fit(self, X_list):
        """Fit the model to multiple datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X_list = self._check_X_list(X_list)

        if self._Aknw is not None:
            if (self._n_features, self._n_features) != self._Aknw.shape:
                raise ValueError(
                    'The shape of prior background_knowledge must be (n_features, n_features)')

        # Causal discovery
        U = np.arange(self._n_features)
        K = []
        X_list_ = [np.copy(X) for X in X_list]
        for _ in range(self._n_features):
            m = self._search_causal_order(X_list_, U)
            for i in U:
                if i != m:
                    for d in range(len(X_list_)):
                        X_list_[d][:, i] = self._residual(
                            X_list_[d][:, i], X_list_[d][:, m])
            K.append(m)
            U = U[U != m]
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[self._partial_orders[:, 0] != m]

        self._causal_order = K

        self._adjacency_matrices = []
        for X in X_list:
            self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)
            self._adjacency_matrices.append(self._adjacency_matrix)
        return self

    def bootstrap(self, X_list, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        results : array-like, shape (BootstrapResult, ...)
            Returns the results of bootstrapping for multiple datasets.
        """
        # Check parameters
        X_list = self._check_X_list(X_list)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices_list = np.zeros(
            [len(X_list), n_sampling, self._n_features, self._n_features])
        total_effects_list = np.zeros(
            [len(X_list), n_sampling, self._n_features, self._n_features])
        for n in range(n_sampling):
            resampled_X_list = [resample(X) for X in X_list]
            self.fit(resampled_X_list)

            for i, am in enumerate(self._adjacency_matrices):
                adjacency_matrices_list[i][n] = am

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1:]:
                    effects = self.estimate_total_effect(
                        resampled_X_list, from_, to)
                    for i, effect in enumerate(effects):
                        total_effects_list[i, n, to, from_] = effect

        result_list = []
        for am, te in zip(adjacency_matrices_list, total_effects_list):
            result_list.append(BootstrapResult(am, te))

        return result_list

    def estimate_total_effect(self, X_list, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
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
        X_list = self._check_X_list(X_list)

        # Check from/to causal order
        from_order = self._causal_order.index(from_index)
        to_order = self._causal_order.index(to_index)
        if from_order > to_order:
            warnings.warn(f'The estimated causal effect may be incorrect because '
                          f'the causal order of the destination variable (to_index={to_index}) '
                          f'is earlier than the source variable (from_index={from_index}).')

        effects = []
        for X, am in zip(X_list, self._adjacency_matrices):
            # from_index + parents indices
            parents = np.where(np.abs(am[from_index]) > 0)[0]
            predictors = [from_index]
            predictors.extend(parents)

            # Estimate total effect
            coefs = predict_adaptive_lasso(X, predictors, to_index)

            effects.append(coefs[0])

        return effects

    def get_error_independence_p_values(self, X_list):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_datasets, n_features, n_features)
            p-value matrix of independence between error variables.
        """
        # Check parameters
        X_list = self._check_X_list(X_list)

        p_values = np.zeros([len(X_list), self._n_features, self._n_features])
        for d, (X, am) in enumerate(zip(X_list, self._adjacency_matrices)):
            n_samples = X.shape[0]
            E = X - np.dot(am, X.T).T
            for i, j in itertools.combinations(range(self._n_features), 2):
                _, p_value = hsic_test_gamma(np.reshape(E[:, i], [n_samples, 1]),
                                             np.reshape(E[:, j], [n_samples, 1]))
                p_values[d, i, j] = p_value
                p_values[d, j, i] = p_value

        return p_values

    def _check_X_list(self, X_list):
        """Check input X list."""
        if not isinstance(X_list, list):
            raise ValueError('X_list must be a list.')

        if len(X_list) < 2:
            raise ValueError(
                'X_list must be a list containing at least two items')

        self._n_features = check_array(X_list[0]).shape[1]
        X_list_ = []
        for X in X_list:
            X_ = check_array(X)
            if X_.shape[1] != self._n_features:
                raise ValueError(
                    'X_list must be a list with the same number of features')
            X_list_.append(X_)

        return np.array(X_list_)

    def _search_causal_order(self, X_list, U):
        """Search the causal ordering."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        total_size = 0
        for X in X_list:
            total_size += len(X)

        MG_list = []
        for i in Uc:
            MG = 0
            for X in X_list:
                M = 0
                for j in U:
                    if i != j:
                        xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                        xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                        ri_j = xi_std if i in Vj and j in Uc else self._residual(
                            xi_std, xj_std)
                        rj_i = xj_std if j in Vj and i in Uc else self._residual(
                            xj_std, xi_std)
                        M += np.min([0, self._diff_mutual_info(xi_std,
                                                               xj_std, ri_j, rj_i)]) ** 2
                MG += M * (len(X) / total_size)
            MG_list.append(-1.0 * MG)
        return Uc[np.argmax(MG_list)]

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (B, ...)
            The list of adjacency matrix B for multiple datasets.
            The shape of B is (n_features, n_features), where 
            n_features is the number of features.
        """
        return self._adjacency_matrices
