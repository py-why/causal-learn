"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import itertools
import warnings

import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array, resample
from statsmodels.tsa.vector_ar.var_model import VAR

from .base import _BaseLiNGAM
from .bootstrap import TimeseriesBootstrapResult
from .direct_lingam import DirectLiNGAM
from .hsic import hsic_test_gamma
from .utils import predict_adaptive_lasso


class VARLiNGAM:
    """Implementation of VAR-LiNGAM Algorithm [1]_

    References
    ----------
    .. [1] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer.
       Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity. Journal of Machine Learning Research, 11: 1709-1731, 2010.
    """

    def __init__(self, lags=1, criterion='bic', prune=False, ar_coefs=None, lingam_model=None, random_state=None):
        """Construct a VARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=False)
            Whether to prune the adjacency matrix or not.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
            Shape must be (``lags``, n_features, n_features).
        lingam_model : lingam object inherits 'lingam._BaseLiNGAM', optional (default=None)
            LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._lags = lags
        self._criterion = criterion
        self._prune = prune
        self._ar_coefs = check_array(
            ar_coefs, allow_nd=True) if ar_coefs is not None else None
        self._lingam_model = lingam_model
        self._random_state = random_state

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        returns
        -------
        self : object
            Returns the instance itself.
        """
        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        lingam_model = self._lingam_model
        if lingam_model is None:
            lingam_model = DirectLiNGAM()
        elif not isinstance(lingam_model, _BaseLiNGAM):
            raise ValueError('lingam_model must be a subclass of _BaseLiNGAM')

        M_taus = self._ar_coefs

        if M_taus is None:
            M_taus, lags, residuals = self._estimate_var_coefs(X)
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)

        model = lingam_model
        model.fit(residuals)

        B_taus = self._calc_b(X, model.adjacency_matrix_, M_taus)

        if self._prune:
            B_taus = self._pruning(X, B_taus, model.causal_order_)

        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals

        self._causal_order = model.causal_order_
        self._adjacency_matrices = B_taus

        return self

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
        result : TimeseriesBootstrapResult
            Returns the result of bootstrapping.
        """
        X = check_array(X)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        lags = self._lags

        criterion = self._criterion
        self._criterion = None

        self.fit(X)

        residuals = self._residuals
        ar_coefs = self._ar_coefs

        total_effects = np.zeros(
            [n_sampling, n_features, n_features * (1 + self._lags)])

        adjacency_matrices = []
        for i in range(n_sampling):
            sampled_residuals = resample(residuals, n_samples=n_samples)

            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < lags:
                    resampled_X[j, :] = sampled_residuals[j]
                    continue

                ar = np.zeros((1, n_features))
                for t, M in enumerate(ar_coefs):
                    ar += np.dot(M, resampled_X[j - t - 1, :].T).T

                resampled_X[j, :] = ar + sampled_residuals[j]

            self.fit(resampled_X)
            am = np.concatenate([*self._adjacency_matrices], axis=1)
            adjacency_matrices.append(am)

            # total effects
            for c, to in enumerate(reversed(self._causal_order)):
                # time t
                for from_ in self._causal_order[:n_features - (c + 1)]:
                    total_effects[i, to, from_] = self.estimate_total_effect(
                        resampled_X, from_, to)

                # time t-tau
                for lag in range(self._lags):
                    for from_ in range(n_features):
                        total_effects[i, to, from_ + n_features] = self.estimate_total_effect(
                            resampled_X, from_, to, lag + 1)

        self._criterion = criterion

        return TimeseriesBootstrapResult(adjacency_matrices, total_effects)

    def estimate_total_effect(self, X, from_index, to_index, from_lag=0):
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
        X = check_array(X)
        n_features = X.shape[1]

        # Check from/to causal order
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(f'The estimated causal effect may be incorrect because '
                              f'the causal order of the destination variable (to_index={to_index}) '
                              f'is earlier than the source variable (from_index={from_index}).')

        # X + lagged X
        X_joined = np.zeros((X.shape[0], X.shape[1] * (1 + self._lags + from_lag)))
        for p in range(1 + self._lags + from_lag):
            pos = n_features * p
            X_joined[:, pos:pos +
                            n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        # from_index + parents indices
        am = np.concatenate([*self._adjacency_matrices], axis=1)
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        from_index = from_index if from_lag == 0 else from_index + \
                                                      (n_features * from_lag)
        parents = parents if from_lag == 0 else parents + (n_features * from_lag)
        predictors = [from_index]
        predictors.extend(parents)

        # estimate total effect
        coefs = predict_adaptive_lasso(X_joined, predictors, to_index)

        return coefs[0]

    def get_error_independence_p_values(self):
        """Calculate the p-value matrix of independence between error variables.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        nn = self.residuals_
        B0 = self._adjacency_matrices[0]
        E = np.dot(np.eye(B0.shape[0]) - B0, nn.T).T
        n_samples = E.shape[0]
        n_features = E.shape[1]

        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(np.reshape(E[:, i], [n_samples, 1]),
                                         np.reshape(E[:, j], [n_samples, 1]))
            p_values[i, j] = p_value
            p_values[j, i] = p_value

        return p_values

    def _estimate_var_coefs(self, X):
        """Estimate coefficients of VAR"""
        # XXX: VAR.fit() is not searching lags correctly
        if self._criterion not in ['aic', 'fpe', 'hqic', 'bic']:
            var = VAR(X)
            result = var.fit(maxlags=self._lags, trend='nc')
        else:
            min_value = float('Inf')
            result = None

            for lag in range(1, self._lags + 1):
                var = VAR(X)
                fitted = var.fit(maxlags=lag, ic=None, trend='nc')

                value = getattr(fitted, self._criterion)
                if value < min_value:
                    min_value = value
                    result = fitted

        return result.coefs, result.k_ar, result.resid

    def _calc_residuals(self, X, M_taus, lags):
        """Calculate residulas"""
        X = X.T
        n_features = X.shape[0]
        n_samples = X.shape[1]

        residuals = np.zeros((n_features, n_samples))
        for t in range(n_samples):
            if t - lags < 0:
                continue

            estimated = np.zeros((X.shape[0], 1))
            for tau in range(1, lags + 1):
                estimated += np.dot(M_taus[tau - 1],
                                    X[:, t - tau].reshape((-1, 1)))

            residuals[:, t] = X[:, t] - estimated.reshape((-1,))

        residuals = residuals[:, lags:].T

        return residuals

    def _calc_b(self, X, B0, M_taus):
        """Calculate B"""
        n_features = X.shape[1]

        B_taus = np.array([B0])

        for M in M_taus:
            B_t = np.dot((np.eye(n_features) - B0), M)
            B_taus = np.append(B_taus, [B_t], axis=0)

        return B_taus

    def _pruning(self, X, B_taus, causal_order):
        """Prune edges"""
        n_features = X.shape[1]

        stacked = [np.flip(X, axis=0)]
        for i in range(self._lags):
            stacked.append(np.roll(stacked[-1], -1, axis=0))
        blocks = np.array(list(zip(*stacked)))[:-self._lags]

        for i in range(n_features):
            causal_order_no = causal_order.index(i)
            ancestor_indexes = causal_order[:causal_order_no]

            obj = np.zeros((len(blocks)))
            exp = np.zeros(
                (len(blocks), causal_order_no + n_features * self._lags))
            for j, block in enumerate(blocks):
                obj[j] = block[0][i]
                exp[j:] = np.concatenate(
                    [block[0][ancestor_indexes].flatten(), block[1:][:].flatten()], axis=0)

            # adaptive lasso
            gamma = 1.0
            lr = LinearRegression()
            lr.fit(exp, obj)
            weight = np.power(np.abs(lr.coef_), gamma)
            reg = LassoLarsIC(criterion='bic')
            reg.fit(exp * weight, obj)
            coef = reg.coef_ * weight

            B_taus[0][i, ancestor_indexes] = coef[:causal_order_no]
            for j in range(len(B_taus[1:])):
                B_taus[j + 1][i, :] = coef[causal_order_no + n_features *
                                           j: causal_order_no + n_features * j + n_features]

        return B_taus

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where 
            n_features is the number of features.
        """
        return self._causal_order

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (lags, n_features, n_features)
            The adjacency matrix of fitted model, where 
            n_features is the number of features.
        """
        return self._adjacency_matrices

    @property
    def residuals_(self):
        """Residuals of regression.

        Returns
        -------
        residuals_ : array-like, shape (n_samples)
            Residuals of regression, where n_samples is the number of samples.
        """
        return self._residuals
