"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array

from .base import _BaseLiNGAM


class CausalEffect(object):
    """Implementation of causality and prediction. [1]_

    References
    ----------
    .. [1] P. Blöbaum and S. Shimizu. Estimation of interventional effects of features on prediction. 
       In Proc. 2017 IEEE International Workshop on Machine Learning for Signal Processing (MLSP2017), pp. 1--6, Tokyo, Japan, 2017.
    """

    def __init__(self, causal_model):
        """Construct a CausalEffect.

        Parameters
        ----------
        causal_model : lingam object inherits 'lingam._BaseLiNGAM' or array-like with shape (n_features, n_features)
            Causal model for calculating causal effects.
            The lingam object is ``lingam.DirectLiNGAM`` or ``lingam.ICALiNGAM``, and ``fit`` function needs to be executed already.
            For array-like, adjacency matrix to estimate causal effect, where ``n_features`` is the number of features.
        """
        self._causal_model = causal_model
        self._B = None
        self._causal_order = None

    def _check_init_params(self):
        """Check initial parameters."""
        # already checked
        if self._B is not None:
            return

        if isinstance(self._causal_model, _BaseLiNGAM):
            self._B = self._causal_model.adjacency_matrix_
            self._causal_order = self._causal_model.causal_order_
            return
        elif type(self._causal_model) is np.ndarray or type(self._causal_model) is list:
            B = self._causal_model if type(
                self._causal_model) is np.ndarray else np.array(self._causal_model)
            if len(B.shape) != 2:
                raise ValueError("Specified 'causal_model' is not matrix")
            if B.shape[0] != B.shape[1]:
                raise ValueError(
                    "Specified 'causal_model' is not square matrix.")

            original_index = np.arange(B.shape[0])
            causal_order = []

            B_ = B
            for _ in range(B.shape[0]):
                zero_rows = np.where(np.sum(np.abs(B_), axis=1) < 1e-10)[0]
                if len(zero_rows) == 0:
                    raise ValueError(
                        "Specified 'causal_model' is not lower triangular matrix.")

                causal_order.append(original_index[zero_rows[0]])
                original_index = np.delete(original_index, zero_rows[0], 0)
                mask = np.delete(np.arange(len(B_)), zero_rows[0], 0)
                B_ = B_[mask][:, mask]

            self._B = B
            self._causal_order = causal_order
            return

        else:
            raise ValueError("Specified 'causal_model' cannot be used.")

        return

    def _get_propagated_effects(self, En, intervention_index, intervention_value):
        """Get propagated effects according to causal order.

        Parameters
        ----------
        En : array-like, shpae (n_features)
            Expectations of each noise variable.
        intervention_index : int
            Index of variable to apply intervention.
        intervention_value : float
            Value of intervention.

        Returns
        -------
        propagated_effects : array-like, shpae (n_features)
            Propagated effects, where ``n_features`` is the number of features.
        """
        effects = np.zeros(len(self._causal_order))
        for i in self._causal_order:
            if i == intervention_index:
                effects[i] = intervention_value
            else:
                effects[i] = np.dot(self._B[i, :], effects) + En[i]

        return effects

    def _predict(self, X, pred_model):
        """Predict the expectation with prediction model.

        Parameters
        ----------
        X : array-like, shpae (n_predictors)
            Predictors, where ``n_predictors`` is the number of variables.
        pred_model : model object implementing 'predict' or 'predict_proba'
            Model to predict the expectation. For linear regression or non-linear reggression, model object must have ``predict`` method.
            For logistic regression, model object must have ``predict_proba`` method.

        Returns
        -------
        pred : float
            Predicted value.
        """
        if hasattr(pred_model, 'predict_proba'):
            p0, p1 = pred_model.predict_proba(X.reshape(1, -1))[0]
            pred = p0 - p1
        elif hasattr(pred_model, 'predict'):
            pred = pred_model.predict(X.reshape(1, -1))[0]
        else:
            raise ValueError("'pred_model' has no prediction method.")
        return pred

    def estimate_effects_on_prediction(self, X, target_index, pred_model):
        """ Estimate the intervention effect with the prediction model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        target_index : int
            Index of target variable.
        pred_model : model object implementing 'predict' or 'predict_proba'
            Model to predict the expectation. For linear regression or non-linear reggression, model object must have ``predict`` method.
            For logistic regression, model object must have ``predict_proba`` method.

        Returns
        -------
        intervention_effects : array-like, shape (n_features, 2)
            Estimated values of intervention effect. 
            The first column of the list is the value of 'E[Y|do(Xi=mean)]-E[Y|do(Xi=mean+std)]',
            and the second column is the value of 'E[Y|do(Xi=mean)]–E[Y|do(Xi=mean-std)]'.
            The maximum value in this array is the feature having the greatest intervention effect.
        """
        # Check parameters
        X = check_array(X)
        self._check_init_params()

        vars_ = [i for i in range(X.shape[1]) if i != target_index]
        Ex = X.mean(axis=0)
        En = Ex - np.dot(self._B, Ex)

        effects = []
        for i in range(X.shape[1]):
            # E[Y|do(Xi=mean)]
            Ex_do = self._get_propagated_effects(En, i, Ex[i])
            Ey_do = self._predict(Ex_do[vars_], pred_model)

            # E[Y|do(Xi=mean)]-E[Y|do(Xi=mean+std)]
            Ex_do = self._get_propagated_effects(En, i, Ex[i] + X[:, i].std())
            Ey1 = Ey_do - self._predict(Ex_do[vars_], pred_model)

            # E[Y|do(Xi=mean)]–E[Y|do(Xi=mean-std)]
            Ex_do = self._get_propagated_effects(En, i, Ex[i] - X[:, i].std())
            Ey2 = Ey_do - self._predict(Ex_do[vars_], pred_model)

            effects.append([np.abs(Ey1), np.abs(Ey2)])

        return np.array(effects)

    def estimate_optimal_intervention(self, X, target_index, pred_model, intervention_index, desired_output):
        """ Estimate of the intervention such that the expectation of
        the prediction of the post-intervention observations is equal
        or close to a specified value.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        target_index : int
            Index of target variable.
        pred_model : model object.
            Model to predict the expectation. Only linear regression model can be specified.
            Model object musst have ``coef_`` and ``intercept_`` attributes.
        intervention_index : int
            Index of variable to apply intervention.
        desired_output : 
            Desired expected post-intervention output of prediction.

        Returns
        -------
        optimal_intervention : float
            Optimal intervention on ``intervention_index`` variable.
        """
        # Check parameters
        X = check_array(X)
        self._check_init_params()

        # Allow linear regression model.
        if not hasattr(pred_model, 'coef_') and not hasattr(pred_model, 'intercept_'):
            raise ValueError(
                "'pred_model' does not have regression coefficient attributes.")
        if hasattr(pred_model, 'predict_proba'):
            raise ValueError("'pred_model' is not linear regression model.")

        Ex = X.mean(axis=0)
        En = Ex - np.dot(self._B, Ex)

        s = [i for i in range(X.shape[1]) if i != intervention_index]
        root_vars = np.where(np.sum(self._B, axis=1) == 0)[0].tolist()
        s = [i for i in s if i not in root_vars]
        alpha = np.zeros(self._B.shape[1])

        alpha[intervention_index] = 1
        Ex[intervention_index] = 0
        En[intervention_index] = 0

        while len(s) > 0:
            k = np.random.choice(s)
            parents = np.where(np.abs(self._B[k]) > 0)[0].tolist()
            if len(list(set(parents) & set(s))) == 0:
                a = 0  # local alpha
                u = 0  # local mew
                for q in parents:
                    a = a + self._B[k, q] * alpha[q]
                    if q != intervention_index:
                        u = u + self._B[k, q] * Ex[q]
                Ex[k] = u + En[k]
                alpha[k] = a
                s.remove(k)

        coefs = np.insert(pred_model.coef_, target_index, 0)

        return (desired_output - np.dot(coefs, Ex) - pred_model.intercept_) / np.dot(coefs, alpha)
