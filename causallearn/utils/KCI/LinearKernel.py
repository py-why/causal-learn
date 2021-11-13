from causallearn.utils.KCI.Kernel import Kernel


class LinearKernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)

    def kernel(self, X, Y=None):
        """
        Computes the linear kernel k(x,y)=x^Ty
        """
        if Y == None:
            Y = X
        return X.dot(Y.T)
