from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy import eye, shape, ndarray
from numpy.linalg import pinv


class Kernel(object):
    def __init__(self):
        pass

    @abstractmethod
    def kernel(self, X: ndarray, Y: ndarray | None = None):
        raise NotImplementedError()

    @staticmethod
    def centering_matrix(n: int):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n

    @staticmethod
    def center_kernel_matrix(K: ndarray):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        [Updated @Haoyue 06/24/2022]
        equivalent to:
            H = eye(n) - 1.0 / n
            return H.dot(K.dot(H))
        since n is always big, we can save time on the dot product by plugging H into dot and expand as sum.
        time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element).
        Also, consider the fact that here K (both Kx and Ky) are symmetric matrices, so K_colsums == K_rowsums
        """
        # assert np.all(K == K.T), 'K should be symmetric'
        n = shape(K)[0]
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)

    def center_kernel_matrix_regression(K: ndarray, Kz: ndarray, epsilon: float):
        """
        Centers the kernel matrix via a centering matrix R=I-Kz(Kz+\epsilonI)^{-1} and returns RKR
        """
        n = shape(K)[0]
        Rz = epsilon * pinv(Kz + epsilon * eye(n))
        return Rz.dot(K.dot(Rz)), Rz
