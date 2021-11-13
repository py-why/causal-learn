from causallearn.utils.KCI.Kernel import Kernel


class PolynomialKernel(Kernel):
    def __init__(self, degree=2, const=1.0):
        Kernel.__init__(self)
        self.degree = degree
        self.const = const

    def kernel(self, X, Y=None):
        """
        Computes the polynomial kernel k(x,y)=(c+<x,y>)^degree
        """
        if Y == None:
            Y = X
        return pow(self.const + X.dot(Y.T), self.degree)
