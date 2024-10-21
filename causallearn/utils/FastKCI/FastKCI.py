from causallearn.utils.KCI.KCI import GaussianKernel, Kernel
import numpy as np
from numpy.linalg import eigh
import scipy.stats as stats
from scipy.special import logsumexp
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
import warnings


class FastKCI_CInd(object):
    """
    Python implementation of as speed-up version of the Kernel-based Conditional Independence (KCI) test.
    Unconditional version.

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
    "A kernel-based conditional independence test and application in causal discovery", In UAI 2011.
    [2] M. Zhang, and S. Williamson,
    "Embarrassingly Parallel Inference for Gaussian Processes", In JMLR 20 (2019)
    [3] O. Schacht, and B. Huang
    "FastKCI: A fast Kernel-based Conditional Independence test with application to causal discovery",
    Working Paper.

    """
    def __init__(self, K=10, J=8, alpha=500, epsilon=1e-3, eig_thresh=1e-6, use_gp=False):
        """
        Initialize the FastKCI_CInd object.

        Parameters
        ----------
        K: Number of Gaussians that are assumed to be in the mixture.
        J: Number of independent repittitions.
        alpha: Parameter for the Dirichlet distribution.
        epsilon: Penalty for the matrix ridge regression.
        eig_threshold: Threshold for Eigenvalues.
        use_gp: Whether to use Gaussian Process Regression to determine the kernel widths
        """
        self.K = K
        self.J = J
        self.alpha = alpha
        self.epsilon = epsilon
        self.eig_thresh = eig_thresh
        self.use_gp = use_gp
        self.nullss = 5000

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        """
        Main function: compute the p value of H_0: X|Y conditional on Z.

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        pvalue: p value (scalar)
        test_stat: test statistic (scalar)
        """
        self.data = [data_x, data_y]
        self.data_z = data_z
        self.n = data_x.shape[0]

        self.Z_proposal = Parallel(n_jobs=-1)(delayed(self.partition_data)() for i in range(self.J))
        block_res = Parallel(n_jobs=-1)(delayed(self.pvalue_onblocks)(self.Z_proposal[i]) for i in range(self.J))
        test_stat, null_samples, log_likelihood = zip(*block_res)

        log_likelihood = np.array(log_likelihood)
        self.all_null_samples = np.vstack(null_samples)
        self.all_p = np.array([np.sum(self.all_null_samples[i,] > test_stat[i]) / float(self.nullss) for i in range(self.J)])
        self.prob_weights = np.around(np.exp(log_likelihood-logsumexp(log_likelihood)), 6)
        self.all_test_stats = np.array(test_stat)
        self.test_stat = np.average(np.array(test_stat), weights=self.prob_weights)
        self.null_samples = np.average(null_samples, axis=0, weights=self.prob_weights)
        self.pvalue = np.sum(self.null_samples > self.test_stat) / float(self.nullss)

        return self.pvalue, self.test_stat

    def partition_data(self):
        """
        Partitions data into K Gaussians following an expectation maximization approach on Z.

        Returns
        _________
        Z: Latent Gaussian that was drawn for each observation (nx1 array)
        prob_Z: Log-Likelihood of that random assignment (scalar)
        """
        Z_mean = self.data_z.mean(axis=0)
        Z_sd = self.data_z.std(axis=0)
        mu_k = np.random.normal(Z_mean, Z_sd, size=(self.K, self.data_z.shape[1]))
        sigma_k = np.eye(self.data_z.shape[1])
        pi_j = np.random.dirichlet([self.alpha]*self.K)
        ll = np.tile(np.log(pi_j), (self.n, 1))
        for k in range(self.K):
            ll[:, k] += stats.multivariate_normal.logpdf(self.data_z, mu_k[k, :], cov=sigma_k, allow_singular=True)
        Z = np.array([np.random.multinomial(1, np.exp(ll[n, :]-logsumexp(ll[n, :]))).argmax() for n in range(self.n)])
        le = LabelEncoder()
        Z = le.fit_transform(Z)
        return Z

    def pvalue_onblocks(self, Z_proposal):
        """
        Calculate p value on given partitions of the data.

        Parameters
        ----------
        Z_proposal: partition of the data into K clusters (nxd1 array)

        Returns
        _________
        test_stat: test statistic (scalar)
        null_samples: bootstrapped sampled of the null distribution (self.nullssx1 array)
        log_likelihood: estimated probability P(X,Y|Z,k)
        """
        unique_Z_j = np.unique(Z_proposal)
        test_stat = 0
        log_likelihood = 0
        null_samples = np.zeros((1, self.nullss))
        for k in unique_Z_j:
            K_mask = (Z_proposal == k)
            X_k = np.copy(self.data[0][K_mask])
            Y_k = np.copy(self.data[1][K_mask])
            Z_k = np.copy(self.data_z[K_mask])
            if (Z_k.shape[0] < 6):  # small blocks cause problems in GP
                continue
            Kx, Ky, Kzx, Kzy, epsilon_x, epsilon_y, likelihood_x, likelihood_y = self.kernel_matrix(X_k, Y_k, Z_k)
            KxR, Rzx = Kernel.center_kernel_matrix_regression(Kx, Kzx, epsilon_x)
            if epsilon_x != epsilon_y:
                KyR, Rzy = Kernel.center_kernel_matrix_regression(Ky, Kzy, epsilon_y)
            else:
                KyR = Rzx.dot(Ky.dot(Rzx))
            test_stat += np.einsum('ij,ji->', KxR, KyR)
            uu_prod, size_u = self.get_uuprod(KxR, KyR)
            null_samples += self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
            log_likelihood += likelihood_x + likelihood_y
        return test_stat, null_samples, log_likelihood

    def kernel_matrix(self, data_x, data_y, data_z):
        """
        Calculates the Gaussian Kernel for given data inputs as well as the shared kernel.

        Returns
        _________
        K: Kernel matrices (n_kxn_k array)
        """
        kernel_obj = GaussianKernel()
        kernel_obj.set_width_empirical_kci(data_z)

        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.

        data_y = stats.zscore(data_y, ddof=1, axis=0)
        data_y[np.isnan(data_y)] = 0.

        data_z = stats.zscore(data_z, ddof=1, axis=0)
        data_z[np.isnan(data_z)] = 0.

        data_x = np.concatenate((data_x, 0.5 * data_z), axis=1)

        kernelX = GaussianKernel()
        kernelX.set_width_empirical_kci(data_z)
        kernelY = GaussianKernel()
        kernelY.set_width_empirical_kci(data_z)

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)

        # centering kernel matrix
        Kx = Kernel.center_kernel_matrix(Kx)
        Ky = Kernel.center_kernel_matrix(Ky)

        if not self.use_gp:
            kernelZ = GaussianKernel()
            kernelZ.set_width_empirical_kci(data_z)
            Kzx = kernelZ.kernel(data_z)
            Kzx = Kernel.center_kernel_matrix(Kzx)
            Kzy = Kzx
            epsilon_x, epsilon_y = self.epsilon, self.epsilon
            gpx = GaussianProcessRegressor()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                # P(X|Z)
                gpx.fit(X=data_z, y=data_x)
            likelihood_x = gpx.log_marginal_likelihood_value_
            gpy = GaussianProcessRegressor()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                # P(Y|X,Z)
                gpy.fit(X=np.c_[data_x, data_z], y=data_y)
            likelihood_y = gpy.log_marginal_likelihood_value_

        else:
            n, Dz = data_z.shape

            widthz = np.sqrt(1.0 / (kernelX.width * data_x.shape[1]))

            # Instantiate a Gaussian Process model for x
            wx, vx = eigh(Kx)
            topkx = int(np.max([np.min([400, np.floor(n / 4)]), np.min([n+1, 8])]))
            idx = np.argsort(-wx)
            wx = wx[idx]
            vx = vx[:, idx]
            wx = wx[0:topkx]
            vx = vx[:, 0:topkx]
            vx = vx[:, np.abs(wx) > np.abs(wx).max() * 1e-10]
            wx = wx[np.abs(wx) > np.abs(wx).max() * 1e-10]
            vx = 2 * np.sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / np.sqrt(wx[0])
            kernelx = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) \
                + WhiteKernel(0.1, (1e-10, 1e+1))
            gpx = GaussianProcessRegressor(kernel=kernelx)
            # fit Gaussian process, including hyperparameter optimization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                gpx.fit(data_z, vx)

            # construct Gaussian kernels according to learned hyperparameters
            Kzx = gpx.kernel_.k1(data_z, data_z)
            epsilon_x = np.exp(gpx.kernel_.theta[-1])
            likelihood_x = gpx.log_marginal_likelihood_value_

            # Instantiate a Gaussian Process model for y
            wy, vy = eigh(Ky)
            topky = int(np.max([np.min([400, np.floor(n / 4)]), np.min([n+1, 8])]))
            idy = np.argsort(-wy)
            wy = wy[idy]
            vy = vy[:, idy]
            wy = wy[0:topky]
            vy = vy[:, 0:topky]
            vy = vy[:, np.abs(wy) > np.abs(wy).max() * 1e-10]
            wy = wy[np.abs(wy) > np.abs(wy).max() * 1e-10]
            vy = 2 * np.sqrt(n) * vy.dot(np.diag(np.sqrt(wy))) / np.sqrt(wy[0])
            kernely = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) \
                + WhiteKernel(0.1, (1e-10, 1e+1))
            gpy = GaussianProcessRegressor(kernel=kernely)
            # fit Gaussian process, including hyperparameter optimization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                gpy.fit(data_z, vy)

            # construct Gaussian kernels according to learned hyperparameters
            Kzy = gpy.kernel_.k1(data_z, data_z)
            epsilon_y = np.exp(gpy.kernel_.theta[-1])
            likelihood_y = gpy.log_marginal_likelihood_value_

        return Kx, Ky, Kzx, Kzy, epsilon_x, epsilon_y, likelihood_x, likelihood_y

    def get_uuprod(self, Kx, Ky):
        """
        Compute eigenvalues for null distribution estimation

        Parameters
        ----------
        Kx: centralized kernel matrix for data_x (nxn)
        Ky: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of produced eigenvectors

        """
        n_block = Kx.shape[0]
        wx, vx = eigh(Kx)
        wy, vy = eigh(Ky)
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.eig_thresh]
        wx = wx[wx > np.max(wx) * self.eig_thresh]
        vy = vy[:, wy > np.max(wy) * self.eig_thresh]
        wy = wy[wy > np.max(wy) * self.eig_thresh]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # calculate their product
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((n_block, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > self.n:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def get_kappa(self, mean_appr, var_appr):
        """
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky

        Returns
        ----------
        k_appr, theta_appr: approximated parameters of the gamma distribution

        """
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def null_sample_spectral(self, uu_prod, size_u, T):
        """
        Simulate data from null distribution

        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of produced eigenvectors
        T: sample size

        Returns
        _________
        null_dstr: samples from the null distribution

        """
        eig_uu = np.linalg.eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.eig_thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr


class FastKCI_UInd(object):
    """
    Python implementation of as speed-up version of the Kernel-based Conditional Independence (KCI) test. 
    Unconditional version.

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
    "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    [2] M. Zhang, and S. Williamson,
    "Embarrassingly Parallel Inference for Gaussian Processes" In JMLR 20 (2019)
    [3] O. Schacht, and B. Huang
    "FastKCI: A fast Kernel-based Conditional Independence test with application to causal discovery",
    Working Paper.
    """
    def __init__(self, K=10, J=8, alpha=500):
        """
        Construct the FastKCI_UInd model.

        Parameters
        ----------
        K: Number of Gaussians that are assumed to be in the mixture
        J: Number of independent repittitions.
        alpha: Parameter for the Dirichlet distribution.
        """
        self.K = K
        self.J = J
        self.alpha = alpha
        self.nullss = 5000
        self.eig_thresh = 1e-5

    def compute_pvalue(self, data_x=None, data_y=None):
        """
        Main function: compute the p value and return it together with the test statistic

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)

        Returns
        _________
        pvalue: p value (scalar)
        test_stat: test statistic (scalar)
        """
        self.data_x = data_x
        self.data_y = data_y
        self.n = data_x.shape[0]

        Z_proposal = Parallel(n_jobs=-1)(delayed(self.partition_data)() for i in range(self.J))
        self.Z_proposal, self.prob_Y = zip(*Z_proposal)
        block_res = Parallel(n_jobs=-1)(delayed(self.pvalue_onblocks)(self.Z_proposal[i]) for i in range(self.J))
        test_stat, null_samples, log_likelihood = zip(*block_res)
        self.prob_weights = np.around(np.exp(log_likelihood-logsumexp(log_likelihood)), 6)
        self.test_stat = np.average(np.array(test_stat), weights=self.prob_weights)
        self.null_samples = np.average(null_samples, axis=0, weights=self.prob_weights)
        self.pvalue = np.sum(self.null_samples > self.test_stat) / float(self.nullss)

        return self.pvalue, self.test_stat

    def partition_data(self):
        """
        Partitions data into K Gaussians

        Returns
        _________
        Z: Latent Gaussian that was drawn for each observation (nx1 array)
        prob_Y: Log-Likelihood of that random assignment (scalar)
        """
        Y_mean = self.data_y.mean(axis=0)
        Y_sd = self.data_y.std(axis=0)
        mu_k = np.random.normal(Y_mean, Y_sd, size=(self.K, self.data_y.shape[1]))
        sigma_k = np.eye(self.data_y.shape[1])
        pi_j = np.random.dirichlet([self.alpha]*self.K)
        ll = np.tile(np.log(pi_j), (self.n, 1))
        for k in range(self.K):
            ll[:, k] += stats.multivariate_normal.logpdf(self.data_y, mu_k[k, :], cov=sigma_k, allow_singular=True)
        Z = np.array([np.random.multinomial(1, np.exp(ll[n, :]-logsumexp(ll[n, :]))).argmax() for n in range(self.n)])
        prop_Y = np.take_along_axis(ll, Z[:, None], axis=1).sum()
        le = LabelEncoder()
        Z = le.fit_transform(Z)
        return (Z, prop_Y)

    def pvalue_onblocks(self, Z_proposal):
        unique_Z_j = np.unique(Z_proposal)
        test_stat = 0
        log_likelihood = 0
        null_samples = np.zeros((1, self.nullss))
        for k in unique_Z_j:
            K_mask = (Z_proposal == k)
            X_k = np.copy(self.data_x[K_mask])
            Y_k = np.copy(self.data_y[K_mask])

            Kx = self.kernel_matrix(X_k)
            Ky = self.kernel_matrix(Y_k)

            v_stat, Kxc, Kyc = self.HSIC_V_statistic(Kx, Ky)

            null_samples += self.null_sample_spectral(Kxc, Kyc)
            test_stat += v_stat

            gpx = GaussianProcessRegressor()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                # P(X|Y)
                gpx.fit(X=Y_k, y=X_k)
            likelihood = gpx.log_marginal_likelihood_value_
            log_likelihood += likelihood

        return test_stat, null_samples, log_likelihood

    def kernel_matrix(self, data):
        """
        Calculates the Gaussian Kernel for given data inputs.

        Returns
        _________
        K: Kernel matrix (n_kxn_k array)
        """
        kernel_obj = GaussianKernel()
        kernel_obj.set_width_empirical_hsic(data)

        data = stats.zscore(data, ddof=1, axis=0)
        data[np.isnan(data)] = 0.

        K = kernel_obj.kernel(data)

        return K

    def get_kappa(self, mean_appr, var_appr):
        """
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)

        Returns
        _________
        k_appr, theta_appr: approximated parameters of the gamma distribution
        """
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def null_sample_spectral(self, Kxc, Kyc):
        """
        Simulate data from null distribution

        Parameters
        ----------
        Kxc: centralized kernel matrix for data_x (nxn)
        Kyc: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        null_dstr: samples from the null distribution

        """
        T = Kxc.shape[0]
        if T > 1000:
            num_eig = int(np.floor(T / 2))
        else:
            num_eig = T
        lambdax = np.linalg.eigvalsh(Kxc)
        lambday = np.linalg.eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)
        lambday = -np.sort(-lambday)
        lambdax = lambdax[0:num_eig]
        lambday = lambday[0:num_eig]
        lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape(
            (num_eig ** 2, 1))
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * self.eig_thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.nullss))
        null_dstr = lambda_prod.T.dot(f_rand) / T
        return null_dstr

    def HSIC_V_statistic(self, Kx, Ky):
        """
        Compute V test statistic from kernel matrices Kx and Ky
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)

        Returns
        _________
        Vstat: HSIC v statistics
        Kxc: centralized kernel matrix for data_x (nxn)
        Kyc: centralized kernel matrix for data_y (nxn)
        """
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        V_stat = np.einsum('ij,ij->', Kxc, Kyc)
        return V_stat, Kxc, Kyc
