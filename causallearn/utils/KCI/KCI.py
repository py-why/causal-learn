import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.Kernel import Kernel
from causallearn.utils.KCI.LinearKernel import LinearKernel
from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel


# Cannot find reference 'xxx' in '__init__.pyi | __init__.pyi | __init__.pxd' is a bug in pycharm, please ignore
class KCI_UInd(object):
    '''
    Python implementation of Kernel-based Conditional Independence (KCI) test. Unconditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
    "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    [2] A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, and A. Smola, "A kernel
       Statistical test of independence." In NIPS 21, 2007.
    '''

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', null_ss=1000, approx=True, est_width='empirical',
                 polyd=2, kwidthx=None, kwidthy=None):
        '''
        Construct the KCI_UInd model.

        Parameters
        ----------
        kernelX: kernel function for input data x
            'Gaussian': Gaussian kernel
            'Polynomial': Polynomial kernel
            'Linear': Linear kernel
        kernelY: kernel function for input data y
        est_width: set kernel width for Gaussian kernels
            'empirical': set kernel width using empirical rules
            'median': set kernel width using the median trick
            'manual': set by users
        null_ss: sample size in simulating the null distribution
        approx: whether to use gamma approximation (default=True)
        polyd: polynomial kernel degrees (default=1)
        kwidthx: kernel width for data x (standard deviation sigma)
        kwidthy: kernel width for data y (standard deviation sigma)
        '''

        self.kernelX = kernelX
        self.kernelY = kernelY
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.nullss = null_ss
        self.thresh = 1e-6
        self.approx = approx

    def compute_pvalue(self, data_x=None, data_y=None):
        '''
        Main function: compute the p value and return it together with the test statistic

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)

        Returns
        _________
        pvalue: p value (scalar)
        test_stat: test statistic (scalar)
        '''

        Kx, Ky = self.kernel_matrix(data_x, data_y)
        test_stat, Kxc, Kyc = self.HSIC_V_statistic(Kx, Ky)
        if self.approx:
            k_appr, theta_appr = self.get_kappa(Kxc, Kyc)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_dstr = self.null_sample_spectral(Kxc, Kyc)
            pvalue = sum(null_dstr.squeeze() > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def kernel_matrix(self, data_x, data_y):
        '''
        Compute kernel matrix for data x and data y

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)

        Returns
        _________
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        '''
        if self.kernelX == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthx is not None:
                    kernelX = GaussianKernel(self.kwidthx)
                else:
                    raise Exception('specify kwidthx')
            else:
                kernelX = GaussianKernel()
                if self.est_width == 'median':
                    kernelX.set_width_median(data_x)
                elif self.est_width == 'empirical':
                    kernelX.set_width_empirical_hsic(data_x)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelX == 'Polynomial':
            kernelX = PolynomialKernel(self.polyd)
        elif self.kernelX == 'Linear':
            kernelX = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        if self.kernelY == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthy is not None:
                    kernelY = GaussianKernel(self.kwidthy)
                else:
                    raise Exception('specify kwidthy')
            else:
                kernelY = GaussianKernel()
                if self.est_width == 'median':
                    kernelY.set_width_median(data_y)
                elif self.est_width == 'empirical':
                    kernelY.set_width_empirical_hsic(data_y)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelY == 'Polynomial':
            kernelY = PolynomialKernel(self.polyd)
        elif self.kernelY == 'Linear':
            kernelY = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        data_x = stats.zscore(data_x, axis=0)
        data_y = stats.zscore(data_y, axis=0)
        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)
        return Kx, Ky

    def HSIC_V_statistic(self, Kx, Ky):
        '''
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
        '''
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        V_stat = np.sum(Kxc * Kyc)
        return V_stat, Kxc, Kyc

    def null_sample_spectral(self, Kxc, Kyc):
        '''
        Simulate data from null distribution

        Parameters
        ----------
        Kxc: centralized kernel matrix for data_x (nxn)
        Kyc: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        null_dstr: samples from the null distribution

        '''
        T = Kxc.shape[0]
        if T > 1000:
            num_eig = np.int(np.floor(T / 2))
        else:
            num_eig = T
        lambdax = eigvalsh(Kxc)
        lambday = eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)
        lambday = -np.sort(-lambday)
        lambdax = lambdax[0:num_eig]
        lambday = lambday[0:num_eig]
        lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape(
            (num_eig ** 2, 1))
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * Thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.nullss))
        null_dstr = lambda_prod.T.dot(f_rand) / T
        return null_dstr

    def get_kappa(self, Kx, Ky):
        '''
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)

        Returns
        _________
        k_appr, theta_appr: approximated parameters of the gamma distribution

        '''
        T = Kx.shape[0]
        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr


class KCI_CInd(object):
    '''
    Python implementation of Kernel-based Conditional Independence (KCI) test. Conditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf, "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    '''

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', nullss=5000, est_width='empirical',
                 use_gp=False, approx=True, polyd=2, kwidthx=None, kwidthy=None, kwidthz=None):
        '''
        Construct the KCI_CInd model.
        Parameters
        ----------
        kernelX: kernel function for input data x
            'Gaussian': Gaussian kernel
            'Polynomial': Polynomial kernel
            'Linear': Linear kernel
        kernelY: kernel function for input data y
        kernelZ: kernel function for input data z (conditional variable)
        est_width: set kernel width for Gaussian kernels
            'empirical': set kernel width using empirical rules
            'median': set kernel width using the median trick
            'manual': set by users
        null_ss: sample size in simulating the null distribution
        use_gp: whether use gaussian process to determine kernel width for z
        approx: whether to use gamma approximation (default=True)
        polyd: polynomial kernel degrees (default=1)
        kwidthx: kernel width for data x (standard deviation sigma, default None)
        kwidthy: kernel width for data y (standard deviation sigma)
        kwidthz: kernel width for data z (standard deviation sigma)
        '''
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz
        self.nullss = nullss
        self.epsilon_x = 0.01
        self.epsilon_y = 0.01
        self.use_gp = use_gp
        self.thresh = 1e-5
        self.approx = approx

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        '''
        Main function: compute the p value and return it together with the test statistic
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        pvalue: p value
        test_stat: test statistic
        '''
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        uu_prod, size_u = self.get_uuprod(KxR, KyR)
        if self.approx:
            k_appr, theta_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_samples = self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
            pvalue = sum(null_samples > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def kernel_matrix(self, data_x, data_y, data_z):
        '''
        Compute kernel matrix for data x, data y, and data_z
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)
        '''
        # normalize the data
        data_x = stats.zscore(data_x, axis=0)
        data_y = stats.zscore(data_y, axis=0)
        data_z = stats.zscore(data_z, axis=0)

        # concatenate x and z
        data_x = np.concatenate((data_x, 0.5 * data_z), axis=1)
        if self.kernelX == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthx is not None:
                    kernelX = GaussianKernel(self.kwidthx)
                else:
                    raise Exception('specify kwidthx')
            else:
                kernelX = GaussianKernel()
                if self.est_width == 'median':
                    kernelX.set_width_median(data_x)
                elif self.est_width == 'empirical':
                    kernelX.set_width_empirical_kci(data_x)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelX == 'Polynomial':
            kernelX = PolynomialKernel(self.polyd)
        elif self.kernelX == 'Linear':
            kernelX = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        if self.kernelY == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthy is not None:
                    kernelY = GaussianKernel(self.kwidthy)
                else:
                    raise Exception('specify kwidthy')
            else:
                kernelY = GaussianKernel()
                if self.est_width == 'median':
                    kernelY.set_width_median(data_y)
                elif self.est_width == 'empirical':
                    kernelY.set_width_empirical_kci(data_y)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelY == 'Polynomial':
            kernelY = PolynomialKernel(self.polyd)
        elif self.kernelY == 'Linear':
            kernelY = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)

        # centering kernel matrix
        Kx = Kernel.center_kernel_matrix(Kx)
        Ky = Kernel.center_kernel_matrix(Ky)

        if self.kernelZ == 'Gaussian':
            if not self.use_gp:
                if self.est_width == 'manual':
                    if self.kwidthz is not None:
                        kernelZ = GaussianKernel(self.kwidthz)
                    else:
                        raise Exception('specify kwidthz')
                else:
                    kernelZ = GaussianKernel()
                    if self.est_width == 'median':
                        kernelZ.set_width_median(data_z)
                    elif self.est_width == 'empirical':
                        kernelZ.set_width_empirical_kci(data_z)
                Kzx = kernelZ.kernel(data_z)
                Kzy = Kzx
            else:
                # learning the kernel width of Kz using Gaussian process
                n, Dz = data_z.shape
                if self.kernelX == 'Gaussian':
                    widthz = sqrt(1.0 / (kernelX.width * data_x.shape[1]))
                else:
                    widthz = 1.0
                # Instantiate a Gaussian Process model for x
                wx, vx = eigh(0.5 * (Kx + Kx.T))
                topkx = int(np.min((400, np.floor(n / 4))))
                idx = np.argsort(-wx)
                wx = wx[idx]
                vx = vx[:, idx]
                wx = wx[0:topkx]
                vx = vx[:, 0:topkx]
                vx = vx[:, wx > wx.max() * self.thresh]
                wx = wx[wx > wx.max() * self.thresh]
                vx = 2 * sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / sqrt(wx[0])
                kernelx = C(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
                gpx = GaussianProcessRegressor(kernel=kernelx)
                # fit Gaussian process, including hyperparameter optimization
                gpx.fit(data_z, vx)

                # construct Gaussian kernels according to learned hyperparameters
                Kzx = gpx.kernel_.k1(data_z, data_z)
                self.epsilon_x = np.exp(gpx.kernel_.theta[-1])

                # Instantiate a Gaussian Process model for y
                wy, vy = eigh(0.5 * (Ky + Ky.T))
                topky = int(np.min((400, np.floor(n / 4))))
                idy = np.argsort(-wy)
                wy = wy[idy]
                vy = vy[:, idy]
                wy = wy[0:topky]
                vy = vy[:, 0:topky]
                vy = vy[:, wy > wy.max() * self.thresh]
                wy = wy[wy > wy.max() * self.thresh]
                vy = 2 * sqrt(n) * vy.dot(np.diag(np.sqrt(wy))) / sqrt(wy[0])
                kernely = C(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
                gpy = GaussianProcessRegressor(kernel=kernely)
                # fit Gaussian process, including hyperparameter optimization
                gpy.fit(data_z, vy)

                # construct Gaussian kernels according to learned hyperparameters
                Kzy = gpy.kernel_.k1(data_z, data_z)
                self.epsilon_y = np.exp(gpy.kernel_.theta[-1])
        elif self.kernelY == 'Polynomial':
            kernelZ = PolynomialKernel(self.polyd)
            Kzx = kernelZ.kernel(data_z)
            Kzy = Kzx
        elif self.kernelY == 'Linear':
            kernelZ = LinearKernel()
            Kzx = kernelZ.kernel(data_z)
            Kzy = Kzx
        else:
            raise Exception('Undefined kernel function')
        return Kx, Ky, Kzx, Kzy

    def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
        '''
        Compute V test statistic from kernel matrices Kx and Ky
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)

        Returns
        _________
        Vstat: KCI v statistics
        KxR: centralized kernel matrix for data_x (nxn)
        KyR: centralized kernel matrix for data_y (nxn)
        '''
        KxR = Kernel.center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        KyR = Kernel.center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        Vstat = np.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def get_uuprod(self, Kx, Ky):
        '''
        Compute eigenvalues for null distribution estimation

        Parameters
        ----------
        Kx: centralized kernel matrix for data_x (nxn)
        Ky: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors

        '''
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.thresh]
        wx = wx[wx > np.max(wx) * self.thresh]
        vy = vy[:, wy > np.max(wy) * self.thresh]
        wy = wy[wy > np.max(wy) * self.thresh]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # calculate their product
        T = Kx.shape[0]
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def null_sample_spectral(self, uu_prod, size_u, T):
        '''
        Simulate data from null distribution

        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors
        T: sample size

        Returns
        _________
        null_dstr: samples from the null distribution

        '''
        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr

    def get_kappa(self, uu_prod):
        '''
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky

        Returns
        ----------
        k_appr, theta_appr: approximated parameters of the gamma distribution

        '''
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr
