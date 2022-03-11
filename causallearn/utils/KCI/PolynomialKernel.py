from __future__ import annotations

from numpy import ndarray

from causallearn.utils.KCI.Kernel import Kernel


class PolynomialKernel(Kernel):
    def __init__(self, degree: int = 2, const: float = 1.0):
        Kernel.__init__(self)
        self.degree = degree
        self.const = const

    def kernel(self, X: ndarray, Y: ndarray | None = None):
        """
        Computes the polynomial kernel k(x,y)=(c+<x,y>)^degree
        """
        if Y is None:
            Y = X
        return pow(self.const + X.dot(Y.T), self.degree)
