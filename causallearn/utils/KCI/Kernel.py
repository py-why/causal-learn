from abc import abstractmethod

from numpy import eye, shape
from numpy.linalg import pinv


class Kernel(object):
    def __init__(self):
        pass

    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()

    @staticmethod
    def centering_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n

    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return H.dot(K.dot(H))

    def center_kernel_matrix_regression(K, Kz, epsilon):
        """
        Centers the kernel matrix via a centering matrix R=I-Kz(Kz+\epsilonI)^{-1} and returns RKR
        """
        n = shape(K)[0]
        Rz = epsilon * pinv(Kz + epsilon * eye(n))
        return Rz.dot(K.dot(Rz))
