"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.utils import check_array

from .base import _BaseLiNGAM


class ICALiNGAM(_BaseLiNGAM):
    """Implementation of ICA-based LiNGAM Algorithm [1]_ 

    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen. 
       A linear non-gaussian acyclic model for causal discovery. 
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    """

    def __init__(self, random_state=None, max_iter=1000):
        """Construct a ICA-based LiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        max_iter : int, optional (default=1000)
            The maximum number of iterations of FastICA.
        """
        super().__init__(random_state)
        self._max_iter = max_iter

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
            Returns the instance of self.
        """
        X = check_array(X)

        # obtain a unmixing matrix from the given data
        ica = FastICA(max_iter=self._max_iter, random_state=self._random_state)
        ica.fit(X)
        W_ica = ica.components_

        # obtain a permuted W_ica
        _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
        PW_ica = np.zeros_like(W_ica)
        PW_ica[col_index] = W_ica

        # obtain a vector to scale
        D = np.diag(PW_ica)[:, np.newaxis]

        # estimate an adjacency matrix
        W_estimate = PW_ica / D
        B_estimate = np.eye(len(W_estimate)) - W_estimate

        causal_order = self._estimate_causal_order(B_estimate)
        self._causal_order = causal_order

        return self._estimate_adjacency_matrix(X)

    def _search_causal_order(self, matrix):
        """Obtain a causal order from the given matrix strictly.

        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.

        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = []

        row_num = matrix.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(matrix):
            # find a row all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from matrix
            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order

    def _estimate_causal_order(self, matrix):
        """Obtain a lower triangular from the given matrix approximately.

        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        
        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = None

        # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
        pos_list = np.argsort(np.abs(matrix), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
        initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
        for i, j in pos_list[:initial_zero_num]:
            matrix[i, j] = 0

        for i, j in pos_list[initial_zero_num:]:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

            causal_order = self._search_causal_order(matrix)
            if causal_order is not None:
                break

        return causal_order
