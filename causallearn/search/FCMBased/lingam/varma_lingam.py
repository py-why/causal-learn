"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import itertools
import warnings

import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array, resample
from statsmodels.tsa.statespace.varmax import VARMAX

from .base import _BaseLiNGAM
from .bootstrap import TimeseriesBootstrapResult
from .direct_lingam import DirectLiNGAM
from .hsic import hsic_test_gamma
from .utils import predict_adaptive_lasso


class VARMALiNGAM:
    """Implementation of VARMA-LiNGAM Algorithm [1]_

    References
    ----------
    .. [1] Yoshinobu Kawahara, Shohei Shimizu, Takashi Washio.
       Analyzing relationships among ARMA processes based on non-Gaussianity of external influences. Neurocomputing, Volume 74: 2212-2221, 2011
    """

    def __init__(self, order=(1, 1), criterion='bic', prune=False, max_iter=100, ar_coefs=None, ma_coefs=None,
                 lingam_model=None, random_state=None):
        """Construct a VARMALiNGAM model.

        Parameters
        ----------
        order : turple, length = 2, optional (default=(1, 1))
            Number of lags for AR and MA model.
        criterion : {'aic', 'bic', 'hqic', None}, optional (default='bic')
            Criterion to decide the best order in the all combinations of ``order``.
            Searching the best order is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=False)
            Whether to prune the adjacency matrix or not.
        max_iter : int, optional (default=100)
            Maximm number of iterations to estimate VARMA model.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR of ARMA. Estimating ARMA model is skipped if specified ``ar_coefs`` and `ma_coefs`.
            Shape must be (``order[0]``, n_features, n_features).
        ma_coefs : array-like, optional (default=None)
            Coefficients of MA of ARMA. Estimating ARMA model is skipped if specified ``ar_coefs`` and `ma_coefs`.
            Shape must be (``order[1]``, n_features, n_features).
        lingam_model : lingam object inherits 'lingam._BaseLiNGAM', optional (default=None)
            LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._order = order
        self._criterion = criterion
        self._prune = prune
        self._max_iter = max_iter
        self._ar_coefs = check_array(
            ar_coefs, allow_nd=True) if ar_coefs is not None else None
        self._ma_coefs = check_array(
            ma_coefs, allow_nd=True) if ma_coefs is not None else None
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

        phis = self._ar_coefs
        thetas = self._ma_coefs
        order = self._order

        if phis is None or thetas is None:
            phis, thetas, order, residuals = self._estimate_varma_coefs(X)
        else:
            p = phis.shape[0]
            q = thetas.shape[0]
            residuals = self._calc_residuals(
                X, phis, thetas, p, q)

        model = lingam_model
        model.fit(residuals)

        psis, omegas = self._calc_psi_and_omega(
            model.adjacency_matrix_, phis, thetas, order)

        if self._prune:
            ee = np.dot(np.eye(
                model.adjacency_matrix_.shape[0]) - model.adjacency_matrix_, residuals.T).T
            psis, omegas = self._pruning(X, ee, order, model.causal_order_)

        self._ar_coefs = phis
        self._ma_coefs = thetas
        self._order = order
        self._residuals = residuals

        self._causal_order = model.causal_order_
        self._adjacency_matrices = (psis, omegas)

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
        (p, q) = self._order

        criterion = self._criterion
        self._criterion = None

        self.fit(X)

        residuals = self._residuals
        ar_coefs = self._ar_coefs
        ma_coefs = self._ma_coefs

        total_effects = np.zeros([n_sampling, n_features, n_features * (1 + p)])

        adjacency_matrices = []
        for i in range(n_sampling):
            sampled_residuals = resample(residuals, n_samples=n_samples)

            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < max(p, q):
                    resampled_X[j, :] = sampled_residuals[j]
                    continue

                ar = np.zeros((1, n_features))
                for t, M in enumerate(ar_coefs):
                    ar += np.dot(M, resampled_X[j - t - 1, :].T).T

                ma = np.zeros((1, n_features))
                for t, M in enumerate(ma_coefs):
                    ma += np.dot(M, sampled_residuals[j - t - 1, :].T).T

                resampled_X[j, :] = ar + sampled_residuals[j] + ma

            self.fit(resampled_X)

            psi = self._adjacency_matrices[0]
            omega = self._adjacency_matrices[1]
            am = np.concatenate([*psi, *omega], axis=1)
            adjacency_matrices.append(am)

            ee = np.dot(np.eye(psi[0].shape[0]) -
                        psi[0], sampled_residuals.T).T

            # total effects
            for c, to in enumerate(reversed(self._causal_order)):
                # time t
                for from_ in self._causal_order[:n_features - (c + 1)]:
                    total_effects[i, to, from_] = self.estimate_total_effect(
                        resampled_X, ee, from_, to)

                # time t-tau
                for lag in range(p):
                    for from_ in range(n_features):
                        total_effects[i, to, from_ + n_features] = self.estimate_total_effect(
                            resampled_X, ee, from_, to, lag + 1)

        self._criterion = criterion

        return TimeseriesBootstrapResult(adjacency_matrices, total_effects)

    def estimate_total_effect(self, X, E, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        E : array-like, shape (n_samples, n_features)
            Original error data, where n_samples is the number of samples
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
        X_joined = np.zeros(
            (X.shape[0], X.shape[1] * (1 + from_lag + self._order[0] + self._order[1])))

        for p in range(1 + self._order[0]):
            pos = n_features * p
            X_joined[:, pos:pos +
                            n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        for q in range(self._order[1]):
            pos = n_features * (1 + self._order[0]) + n_features * q
            X_joined[:, pos:pos +
                            n_features] = np.roll(E[:, 0:n_features], q + 1, axis=0)

        # concat psi and omega
        psi = self._adjacency_matrices[0]
        omega = self._adjacency_matrices[1]
        am = np.concatenate([*psi, *omega], axis=1)

        # from_index + parents indices
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        from_index = from_index if from_lag == 0 else from_index + n_features
        parents = parents if from_lag == 0 else parents + n_features
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        coefs = predict_adaptive_lasso(X_joined, predictors, to_index)

        return coefs[0]

    def get_error_independence_p_values(self):
        """Calculate the p-value matrix of independence between error variables.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        eps = self.residuals_
        psi0 = self._adjacency_matrices[0][0]
        E = np.dot(np.eye(psi0.shape[0]) - psi0, eps.T).T
        n_samples = E.shape[0]
        n_features = E.shape[1]

        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(np.reshape(E[:, i], [n_samples, 1]),
                                         np.reshape(E[:, j], [n_samples, 1]))
            p_values[i, j] = p_value
            p_values[j, i] = p_value

        return p_values

    def _estimate_varma_coefs(self, X):
        if self._criterion not in ['aic', 'bic', 'hqic']:
            result = VARMAX(X, order=self._order, trend='c').fit(
                maxiter=self._max_iter)
        else:
            min_value = float('Inf')
            result = None

            orders = [(p, q) for p in range(self._order[0] + 1)
                      for q in range(self._order[1] + 1)]
            orders.remove((0, 0))

            for order in orders:
                fitted = VARMAX(X, order=order, trend='c').fit(
                    maxiter=self._max_iter)

                value = getattr(fitted, self._criterion)
                if value < min_value:
                    min_value = value
                    result = fitted

        return result.coefficient_matrices_var, result.coefficient_matrices_vma, result.specification[
            'order'], result.resid

    def _calc_residuals(self, X, ar_coefs, ma_coefs, p, q):
        X = X.T
        n_features = X.shape[0]
        n_samples = X.shape[1]

        start_index = max(p, q)

        epsilon = np.zeros([n_features, n_samples])
        for t in range(n_samples):
            if t < start_index:
                epsilon[:, t] = np.random.normal(size=(n_features))
                continue

            ar = np.zeros([n_features, 1])
            for i in range(p):
                ar += np.dot(ar_coefs[i], X[:, t - i - 1].reshape(-1, 1))

            ma = np.zeros([n_features, 1])
            for j in range(q):
                ma += np.dot(ma_coefs[j], epsilon[:, t - j - 1].reshape(-1, 1))

            epsilon[:, t] = X[:, t] - (ar.flatten() + ma.flatten())

        residuals = epsilon[:, start_index:].T

        return residuals

    def _calc_psi_and_omega(self, psi0, phis, thetas, order):
        psis = [psi0]
        for i in range(order[0]):
            psi = np.dot(np.eye(psi0.shape[0]) - psi0, phis[i])
            psis.append(psi)

        omegas = []
        for j in range(order[1]):
            omega = np.dot(np.eye(
                psi0.shape[0]) - psi0, thetas[j], np.linalg.inv(np.eye(psi0.shape[0]) - psi0))
            omegas.append(omega)

        return np.array(psis), np.array(omegas)

    def _pruning(self, X, ee, order, causal_order):
        """"""
        n_features = X.shape[1]

        # join X(t), X(t-1) and e(t-1)
        X_joined = np.zeros((X.shape[0], X.shape[1] * (1 + order[0] + order[1])))
        for p in range(1 + order[0]):
            pos = n_features * p
            X_joined[:, pos:pos +
                            n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        for q in range(order[1]):
            pos = n_features * (1 + order[0]) + n_features * q
            X_joined[:, pos:pos +
                            n_features] = np.roll(ee[:, 0:n_features], q + 1, axis=0)

        # pruned by adaptive lasso
        psi_omega = np.zeros((n_features, n_features * (1 + order[0] + order[1])))
        for i, target in enumerate(causal_order):
            predictors = [j for j in range(
                X_joined.shape[1]) if j not in causal_order[i:]]

            # adaptive lasso
            gamma = 1.0
            lr = LinearRegression()
            lr.fit(X_joined[:, predictors], X_joined[:, target])
            weight = np.power(np.abs(lr.coef_), gamma)
            reg = LassoLarsIC(criterion='bic')
            reg.fit(X_joined[:, predictors] * weight, X_joined[:, target])

            psi_omega[target, predictors] = reg.coef_ * weight

        # split psi and omega
        psis = np.zeros(((1 + order[0]), n_features, n_features))
        for p in range(1 + order[0]):
            pos = n_features * p
            psis[p] = psi_omega[:, pos:pos + n_features]

        omegas = np.zeros((order[1], n_features, n_features))
        for q in range(order[1]):
            pos = n_features * (1 + order[0]) + n_features * q
            omegas[q] = psi_omega[:, pos:pos + n_features]

        return psis, omegas

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
        adjacency_matrices_ : array-like, shape ((p, n_features, n_features), (q, n_features, n_features))
            The adjacency matrix psi and omega of fitted model, where 
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
