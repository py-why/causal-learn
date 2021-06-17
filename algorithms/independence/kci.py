'''
    File: kci.py
    https://github.com/cmu-phil/pytrad/blob/mgong/kci/KCI.py
'''

import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, svd, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from ..utils.kernel import Kernel


class KCI_UInd(object):
    def __init__(self, sample_size, kernelX=None, kernelY=None, est_width='empirical', null_sample_size=1000,
                 approx=True):
        self.T = sample_size
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.est_width = est_width
        self.nullss = null_sample_size
        self.thresh = 1e-6
        if self.T > 1000:
            self.num_eig = np.int(np.floor(self.T / 2))
        else:
            self.num_eig = self.T
        self.approx = approx

    def kernel_matrix(self, data_x, data_y):
        if data_x.ndim == 1:
            data_x = np.expand_dims(data_x, axis=1)

        if data_y.ndim == 1:
            data_y = np.expand_dims(data_y, axis=1)

        if self.kernelX.__class__.__name__ == 'GaussianKernel':
            if self.est_width == 'median':
                thetax = self.kernelX.get_width_median(data_x)
                self.kernelX.set_width(float(thetax * data_x.shape[1]))
            elif self.est_width == 'empirical':
                thetax = self.kernelX.get_width_empirical_hsic(data_x)
                self.kernelX.set_width(float(thetax * data_x.shape[1]))

        if self.kernelY.__class__.__name__ == 'GaussianKernel':
            if self.est_width == 'median':
                thetay = self.kernelY.get_width_median(data_y)
                self.kernelY.set_width(float(thetay * data_y.shape[1]))
            elif self.est_width == 'empirical':
                thetay = self.kernelY.get_width_empirical_hsic(data_y)
                self.kernelY.set_width(float(thetay * data_y.shape[1]))

        data_x = stats.zscore(data_x, axis=0)
        data_y = stats.zscore(data_y, axis=0)
        Kx = self.kernelX.kernel(data_x)
        Ky = self.kernelY.kernel(data_y)
        return Kx, Ky

    def HSIC_V_statistic(self, Kx, Ky):
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        return np.sum(Kxc * Kyc), Kxc, Kyc

    def null_sample_spectral(self, Kxc, Kyc):
        lambdax = eigvalsh(Kxc)
        lambday = eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)
        lambday = -np.sort(-lambday)
        lambdax = lambdax[0:self.num_eig]
        lambday = lambday[0:self.num_eig]
        lambda_prod = np.dot(lambdax.reshape(self.num_eig, 1), lambday.reshape(1, self.num_eig)).reshape(
            (self.num_eig ** 2, 1))
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * self.thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.nullss))
        null_dstr = lambda_prod.T.dot(f_rand) / self.T
        return null_dstr

    def get_kappa(self, Kx, Ky):
        mean_appr = np.trace(Kx) * np.trace(Ky) / self.T
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / self.T / self.T
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def compute_pvalue(self, data_x=None, data_y=None):
        # compute kernel
        Kx, Ky = self.kernel_matrix(data_x, data_y)
        hsic_statistic, Kxc, Kyc = self.HSIC_V_statistic(Kx, Ky)
        if self.approx:
            k_appr, theta_appr = self.get_kappa(Kxc, Kyc)
            pvalue = 1 - stats.gamma.cdf(hsic_statistic, k_appr, 0, theta_appr)
        else:
            null_dstr = self.null_sample_spectral(Kxc, Kyc)
            pvalue = sum(null_dstr.squeeze() > hsic_statistic) / float(self.nullss)
        return pvalue


class KCI_CInd(object):
    def __init__(self, sample_size, kernelX=None, kernelY=None, kernelZ=None, est_width='empirical', nullss=5000,
                 use_gp=True, approx=True):
        self.T = sample_size
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.nullss = nullss
        self.epsilon_x = 0.01
        self.epsilon_y = 0.01
        self.use_gp = use_gp
        self.num_eig = self.T
        self.thresh = 1e-5
        self.approx = approx

    def kernel_matrix(self, data_x, data_y, data_z):
        # normalize the datasets
        data_x = stats.zscore(data_x, axis=0)
        data_y = stats.zscore(data_y, axis=0)
        data_z = stats.zscore(data_z, axis=0)

        # concatenate x and z
        data_x = np.concatenate((data_x, 0.5 * data_z), axis=1)
        if self.kernelX.__class__.__name__ == 'GaussianKernel':
            if self.est_width == 'median':
                thetax = self.kernelX.get_width_median(data_x)
                self.kernelX.set_width(float(thetax / data_z.shape[1]))
            elif self.est_width == 'empirical':
                thetax = self.kernelX.get_width_empirical_kci(data_x)
                self.kernelX.set_width(float(thetax / data_z.shape[1]))

        if self.kernelY.__class__.__name__ == 'GaussianKernel':
            if self.est_width == 'median':
                thetay = self.kernelY.get_width_median(data_y)
                self.kernelY.set_width(float(thetay / data_z.shape[1]))
            elif self.est_width == 'empirical':
                thetay = self.kernelY.get_width_empirical_kci(data_y)
                self.kernelY.set_width(float(thetay / data_z.shape[1]))

        Kx = self.kernelX.kernel(data_x)
        Ky = self.kernelY.kernel(data_y)

        # centering kernel matrix
        Kx = Kernel.center_kernel_matrix(Kx)
        Ky = Kernel.center_kernel_matrix(Ky)

        # learning the kernel width of Kz using Gaussian process
        if self.use_gp and self.kernelZ.__class__.__name__ == 'GaussianKernel':
            n, Dz = data_z.shape
            widthz = sqrt(1.0 / thetax)

            # Instantiate a Gaussian Process model for x
            wx, vx = eigh(0.5 * (Kx + Kx.T))
            topkx = np.int(np.min((400, np.floor(n / 4))))
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
            topky = np.int(np.min((400, np.floor(n / 4))))
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

        return Kx, Ky, Kzx, Kzy

    def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
        KxR = Kernel.center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        KyR = Kernel.center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        return np.sum(KxR * KyR), KxR, KyR

    def get_uuprod(self, Kx, Ky):
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
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((self.T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > self.T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def null_sample_spectral(self, uu_prod, size_u):

        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((self.T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr

    def get_kappa(self, uu_prod):
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        # compute kernel
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        kci_statistic, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        uu_prod, size_u = self.get_uuprod(KxR, KyR)
        if self.approx:
            k_appr, theta_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(kci_statistic, k_appr, 0, theta_appr)
        else:
            null_samples = self.null_sample_spectral(uu_prod, size_u)
            pvalue = sum(null_samples > kci_statistic) / float(self.nullss)
        return pvalue
