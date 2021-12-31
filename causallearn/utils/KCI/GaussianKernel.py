from numpy import exp, median, shape, sqrt
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform

from causallearn.utils.KCI.Kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, width=1.0):
        Kernel.__init__(self)
        self.width = 1.0 / width ** 2

    def kernel(self, X, Y=None):
        """
        Computes the Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)
        """
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            assert (shape(X)[1] == shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')
        K = exp(-0.5 * (sq_dists) * self.width)
        return K

    # kernel width using median trick
    def set_width_median(self, X):
        n = shape(X)[0]
        if n > 1000:
            X = X[permutation(n)[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = median(dists[dists > 0])
        width = sqrt(2.) * median_dist
        theta = 1.0 / (width ** 2)
        self.width = theta

    # use empirical kernel width instead of the median
    def set_width_empirical_kci(self, X):
        n = shape(X)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        self.width = theta / X.shape[1]

    def set_width_empirical_hsic(self, X):
        n = shape(X)[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        theta = 1.0 / (width ** 2)
        self.width = theta * X.shape[1]
