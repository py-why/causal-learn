from .Kernel import Kernel
from numpy import exp, shape, sqrt, median
from numpy.random import permutation
from scipy.spatial.distance import squareform, pdist, cdist


class GaussianKernel(Kernel):
    def __init__(self, theta=1.0, is_sparse=False):
        Kernel.__init__(self)
        self.width = theta
        self.is_sparse = is_sparse

    def __str__(self):
        s = self.__class__.__name__ + "["
        s += "width=" + str(self.width)
        s += "]"
        return s

    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)

        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples, can be None in which case its replaced by X
        """
        if self.is_sparse:
            X = X.todense()
            Y = Y.todense()
        assert (len(shape(X)) == 2)

        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            assert (len(shape(Y)) == 2)
            assert (shape(X)[1] == shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')

        K = exp(-0.5 * (sq_dists) * self.width)
        return K

    # kernel width using median trick
    @staticmethod
    def get_width_median(X, is_sparse=False):

        if is_sparse:
            X = X.todense()
        n = shape(X)[0]
        if n > 1000:
            X = X[permutation(n)[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = median(dists[dists > 0])
        width = sqrt(2.) * median_dist
        theta = 1.0 / (width ** 2)
        return theta

    # use empirical kernel width instead of the median
    @staticmethod
    def get_width_empirical_kci(X, is_sparse=False):
        if is_sparse:
            X = X.todense()
        n = shape(X)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        return theta

    @staticmethod
    def get_width_empirical_hsic(X, is_sparse=False):
        if is_sparse:
            X = X.todense()
        n = shape(X)[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        theta = 1.0 / (width ** 2)
        return theta
