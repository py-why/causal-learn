import numpy as np
import itertools
from scipy.stats import chi2
from scipy.linalg import pinv
from momentchi2 import hbe, sw, lpb4
from scipy.spatial.distance import pdist, squareform


class RCIT(object):
    """
    Python implementation of Randomized Conditional Independence Test (RCIT) test.
    The original R implementation can be found at https://github.com/ericstrobl/RCIT/tree/master

    References
    ----------
    [1] Strobl, E. V., Zhang, K., and Visweswaran, S. (2019). "Approximate kernel-based conditional
    independence tests for fast non-parametric causal discovery." Journal of Causal Inference, 7(1), 20180017.
    """
    def __init__(self, approx="lpd4", num_f=100, num_f2=5, rcit=True):
        """
        Initialize the RCIT object.

        Parameters
        ----------
        approx : str
            Method for approximating the null distribution.
            - "lpd4" for the Lindsay-Pilla-Basak method
            - "hbe" for the Hall-Buckley-Eagleson method
            - "gamma" for the Satterthwaite-Welch method
            - "chi2" for a normalized chi-squared statistic
            - "perm" for permutation testing
            Default is "lpd4".
        num_f : int
            Number of features for conditioning set. Default is 25.
        num_f2 : int
            Number of features for non-conditioning sets. Default is 5.
        rcit : bool
            Whether to use RCIT or RCoT. Default is True.
        """
        self.approx = approx
        self.num_f = num_f
        self.num_f2 = num_f2
        self.rcit = rcit

    def compute_pvalue(self, data_x, data_y, data_z):
        """
        Compute the p value and return it together with the test statistic.

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        -------
        p: p value
        sta: test statistic
        """
        d = data_z.shape[1]
        r = data_x.shape[0]
        r1 = 500 if (r > 500) else r

        data_x = (data_x - data_x.mean(axis=0)) / data_x.std(axis=0, ddof=1)
        data_y = (data_y - data_y.mean(axis=0)) / data_y.std(axis=0, ddof=1)
        data_z = (data_z - data_z.mean(axis=0)) / data_z.std(axis=0, ddof=1)

        if self.rcit:
            data_y = np.column_stack((data_y, data_z))

        sigma = dict()
        for key, value in [("x", data_x), ("y", data_y), ("z", data_z)]:
            distances = pdist(value[:r1, :], metric='euclidean')
            flattened_distances = squareform(distances).ravel()
            non_zero_distances = flattened_distances[flattened_distances != 0]
            sigma[key] = np.median(non_zero_distances)

        four_z = self.random_fourier_features(data_z, num_f=self.num_f, sigma=sigma["z"])
        four_x = self.random_fourier_features(data_x, num_f=self.num_f2, sigma=sigma["x"])
        four_y = self.random_fourier_features(data_y, num_f=self.num_f2, sigma=sigma["y"])

        f_x = (four_x - four_x.mean(axis=0)) / four_x.std(axis=0, ddof=1)
        f_y = (four_y - four_y.mean(axis=0)) / four_y.std(axis=0, ddof=1)
        f_z = (four_z - four_z.mean(axis=0)) / four_z.std(axis=0, ddof=1)

        Cxy = f_x.T @ f_y / (f_x.shape[0] - 1)

        Czz = np.cov(f_z, rowvar=False)

        regularized_Czz = Czz + np.eye(self.num_f) * 1e-10
        L = np.linalg.cholesky(regularized_Czz)
        i_Czz = np.linalg.inv(L).T.dot(np.linalg.inv(L))

        Cxz = f_x.T @ f_z / (f_x.shape[0] - 1)
        Czy = f_z.T @ f_y / (f_z.shape[0] - 1)

        z_i_Czz = f_z @ i_Czz
        e_x_z = z_i_Czz @ Cxz.T
        e_y_z = z_i_Czz @ Czy

        res_x = f_x - e_x_z
        res_y = f_y - e_y_z

        if self.num_f2 == 1:
            self.approx = "hbe"

        if self.approx == "perm":
            Cxy_z = self.matrix_cov(res_x, res_y)
            sta = r * np.sum(Cxy_z**2)

            nperm = 1000

            stas = []
            for _ in range(nperm):
                perm = np.random.choice(np.arange(r), size=r, replace=False)
                Cxy = self.matrix_cov(res_x[perm, ], res_y)
                sta_p = r * np.sum(Cxy**2)
                stas.append(sta_p)

            p = 1 - (np.sum(sta >= stas) / len(stas))

        else:
            Cxy_z = Cxy - Cxz @ i_Czz @ Czy
            sta = r * np.sum(Cxy_z**2)

            d = list(itertools.product(range(f_x.shape[1]), range(f_y.shape[1])))
            res = np.array([res_x[:, idx_x] * res_y[:, idx_y] for idx_x, idx_y in d]).T
            Cov = 1/r * res.T @ res

            if self.approx == "chi2":
                i_Cov = pinv(Cov)

                sta = r * (np.dot(Cxy_z.flatten(), np.dot(i_Cov, Cxy_z.flatten())))
                p = 1 - chi2.cdf(sta, Cxy_z.size)

            else:
                eigenvalues, eigenvectors = np.linalg.eigh(Cov)
                eig_d = eigenvalues[eigenvalues > 0]

                if self.approx == "gamma":
                    p = 1 - sw(eig_d, sta)

                elif self.approx == "hbe":
                    p = 1 - hbe(eig_d, sta)

                elif self.approx == "lpd4":
                    try:
                        p = 1 - lpb4(eig_d, sta)
                    except Exception:
                        p = 1 - hbe(eig_d, sta)
                    if np.isnan(p):
                        p = 1 - hbe(eig_d, sta)

        if (p < 0):
            p = 0

        return p, sta

    def random_fourier_features(self, x, w=None, b=None, num_f=None, sigma=None):
        """
        Generate random Fourier features.

        Parameters
        ----------
        x : np.ndarray
            Random variable x.
        w : np.ndarray
            RRandom coefficients.
        b : np.ndarray
            Random offsets.
        num_f : int
            Number of random Fourier features.
        sigma : float
            Smooth parameter of RBF kernel.

        Returns
        -------
        feat : np.ndarray
            Random Fourier features.
        """
        if num_f is None:
            num_f = 25

        r = x.shape[0]
        c = x.shape[1]

        if ((sigma == 0) | (sigma is None)):
            sigma = 1

        if w is None:
            w = (1/sigma) * np.random.normal(size=(num_f, c))
            b = np.tile(2*np.pi*np.random.uniform(size=(num_f, 1)), (1, r))

        feat = np.sqrt(2) * (np.cos(w[0:num_f, 0:c] @ x.T + b[0:num_f, :])).T

        return feat

    def matrix_cov(self, mat_a, mat_b):
        """
        Compute the covariance matrix between two matrices.
        Equivalent to ``cov()`` between two matrices in R.

        Parameters
        ----------
        mat_a : np.ndarray
            First data matrix.
        mat_b : np.ndarray
            Second data matrix.

        Returns
        -------
        mat_cov : np.ndarray
            Covariance matrix.
        """
        n_obs = mat_a.shape[0]

        assert mat_a.shape == mat_b.shape
        mat_a = mat_a - mat_a.mean(axis=0)
        mat_b = mat_b - mat_b.mean(axis=0)

        mat_cov = mat_a.T @ mat_b / (n_obs - 1)

        return mat_cov


class RIT(object):
    """
    Python implementation of Randomized Independence Test (RIT) test.
    The original R implementation can be found at https://github.com/ericstrobl/RCIT/tree/master

    References
    ----------
    [1] Strobl, E. V., Zhang, K., and Visweswaran, S. (2019). "Approximate kernel-based conditional
    independence tests for fast non-parametric causal discovery." Journal of Causal Inference, 7(1), 20180017.
    """
    def __init__(self, approx="lpd4"):
        """
        Initialize the RIT object.

        Parameters
        ----------
        approx : str
            Method for approximating the null distribution.
            - "lpd4" for the Lindsay-Pilla-Basak method
            - "hbe" for the Hall-Buckley-Eagleson method
            - "gamma" for the Satterthwaite-Welch method
            - "chi2" for a normalized chi-squared statistic
            - "perm" for permutation testing
            Default is "lpd4".
        """
        self.approx = approx

    def compute_pvalue(self, data_x, data_y):
        """
        Compute the p value and return it together with the test statistic.

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        -------
        p: p value
        sta: test statistic
        """
        r = data_x.shape[0]
        r1 = 500 if (r > 500) else r

        data_x = (data_x - data_x.mean(axis=0)) / data_x.std(axis=0, ddof=1)
        data_y = (data_y - data_y.mean(axis=0)) / data_y.std(axis=0, ddof=1)

        sigma = dict()
        for key, value in [("x", data_x), ("y", data_y)]:
            distances = pdist(value[:r1, :], metric='euclidean')
            flattened_distances = squareform(distances).ravel()
            non_zero_distances = flattened_distances[flattened_distances != 0]
            sigma[key] = np.median(non_zero_distances)

        four_x = self.random_fourier_features(data_x, num_f=5, sigma=sigma["x"])
        four_y = self.random_fourier_features(data_y, num_f=5, sigma=sigma["y"])

        f_x = (four_x - four_x.mean(axis=0)) / four_x.std(axis=0, ddof=1)
        f_y = (four_y - four_y.mean(axis=0)) / four_y.std(axis=0, ddof=1)

        Cxy = self.matrix_cov(f_x, f_y)
        sta = r * np.sum(Cxy**2)

        if self.approx == "perm":
            nperm = 1000

            stas = []
            for _ in range(nperm):
                perm = np.random.choice(np.arange(r), size=r, replace=False)
                Cxy = self.matrix_cov(f_x[perm, ], f_y)
                sta_p = r * np.sum(Cxy**2)
                stas.append(sta_p)

            p = 1 - (np.sum(sta >= stas) / len(stas))

        else:
            res_x = f_x - f_x.mean(axis=0)
            res_y = f_y - f_y.mean(axis=0)

            d = list(itertools.product(range(f_x.shape[1]), range(f_y.shape[1])))
            res = np.array([res_x[:, idx_x] * res_y[:, idx_y] for idx_x, idx_y in d]).T
            Cov = 1/r * res.T @ res

            if self.approx == "chi2":
                i_Cov = pinv(Cov)

                sta = r * (np.dot(Cxy.flatten(), np.dot(i_Cov, Cxy.flatten())))
                p = 1 - chi2.cdf(sta, Cxy.size)

            else:
                eigenvalues, eigenvectors = np.linalg.eigh(Cov)
                eig_d = eigenvalues[eigenvalues > 0]

                if self.approx == "gamma":
                    p = 1 - sw(eig_d, sta)

                elif self.approx == "hbe":
                    p = 1 - hbe(eig_d, sta)

                elif self.approx == "lpd4":
                    try:
                        p = 1 - lpb4(eig_d, sta)
                    except Exception:
                        p = 1 - hbe(eig_d, sta)
                    if np.isnan(p):
                        p = 1 - hbe(eig_d, sta)

        if (p < 0):
            p = 0

        return p, sta

    def random_fourier_features(self, x, w=None, b=None, num_f=None, sigma=None):
        """
        Generate random Fourier features.

        Parameters
        ----------
        x : np.ndarray
            Random variable x.
        w : np.ndarray
            RRandom coefficients.
        b : np.ndarray
            Random offsets.
        num_f : int
            Number of random Fourier features.
        sigma : float
            Smooth parameter of RBF kernel.

        Returns
        -------
        feat : np.ndarray
            Random Fourier features.
        """
        if num_f is None:
            num_f = 25

        r = x.shape[0]
        c = x.shape[1]

        if ((sigma == 0) | (sigma is None)):
            sigma = 1

        if w is None:
            w = (1/sigma) * np.random.normal(size=(num_f, c))
            b = np.tile(2*np.pi*np.random.uniform(size=(num_f, 1)), (1, r))

        feat = np.sqrt(2) * (np.cos(w[0:num_f, 0:c] @ x.T + b[0:num_f, :])).T

        return feat

    def matrix_cov(self, mat_a, mat_b):
        """
        Compute the covariance matrix between two matrices.
        Equivalent to ``cov()`` between two matrices in R.

        Parameters
        ----------
        mat_a : np.ndarray
            First data matrix.
        mat_b : np.ndarray
            Second data matrix.

        Returns
        -------
        mat_cov : np.ndarray
            Covariance matrix.
        """
        n_obs = mat_a.shape[0]

        assert mat_a.shape == mat_b.shape
        mat_a = mat_a - mat_a.mean(axis=0)
        mat_b = mat_b - mat_b.mean(axis=0)

        mat_cov = mat_a.T @ mat_b / (n_obs - 1)

        return mat_cov
