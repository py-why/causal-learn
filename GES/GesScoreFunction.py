import numpy.matlib
import math
from GES.KCI.algorithms import *
from GES.GesUtils import *

def local_score_CV_general(Data ,Xi ,PAi ,parameters):
    Data = np.asmatrix(Data)
    PAi = list(PAi)
    # calculate the local score
    # using negative k-fold cross-validated log likelihood as the score
    # based on a regression model in RKHS

    T = Data.shape[0]
    X = Data[:, Xi]
    # lambda 是 python 中的关键字，这里用 var_lambda 代替 lambda
    var_lambda = parameters['lambda']   # regularization parameter
    k = parameters['kfold'] # k-fold cross validation
    n0 = math.floor( T /k)
    gamma = 0.01
    Thresh = 1E-5

    if (len(PAi)):
        PA = Data[:, PAi]

        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, ( T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]) # median value
        width = width * 2
        theta = 1 / (width ** 2)

        Kx, _ = kernel(X, X, (theta, 1)) # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T) # for centering of the data in feature space
        Kx = H0 * Kx * H0 # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T)/2, np.min([400, math.floor(T/2)]), evals_only=False)   # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # eig_Kx = eig_Kx[IIx]
        # eix = eix[:, IIx]
        # mx = len(IIx)

        # set the kernel for PA
        Kpa = np.matlib.ones((T, T))

        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m] ,PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 2
            theta = 1 / (width ** 2)
            Kpa = np.multiply(Kpa, kernel(PA[:, m], PA[:, m], (theta, 1))[0])

        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0 # kernel matrix for PA


        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0: T, n0: T]
                Kx_tr_te = Kx[n0: T, 0: n0]
                Kpa_te = Kpa[0:n0, 0: n0]
                Kpa_tr = Kpa[n0: T, n0: T]
                Kpa_tr_te = Kpa[n0: T, 0: n0]
                nv = n0 # sample size of validated data
            if (kk == k- 1):
                Kx_te = Kx[kk * n0:T, kk * n0: T]
                Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                    np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                       np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.matlib.eye(n1))
            tmp2 = tmp1 * Kx_tr * tmp1
            tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * var_lambda ** 2 / gamma * tmp2) * tmp1
            A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                 + 2 * n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

            B = n1 * var_lambda ** 2 / gamma * tmp2 + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 2
        theta = 1 / (width ** 2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 2)]), evals_only=False)  # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # mx = len(IIx)

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                nv = n0
            if (kk == k - 1):
                Kx_te = Kx[kk * n0: T, kk * n0: T]
                Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


def local_score_CV_multi(Data, Xi, PAi, parameters):  # calculate the local score
    # calculate the local score
    # using negative k-fold cross-validated log likelihood as the score
    # based on a regression model in RKHS
    # for variables with multi-variate dimensions
    #
    # parameters.d_label: index of each variable
    T = Data.shape[0]
    X = Data[:, parameters['dlabel'][Xi]]
    # lambda 是 python 中的关键字，这里用 var_lambda 代替 lambda
    var_lambda = parameters['lambda']  # regularization parameter
    k = parameters['kfold']  # k-fold cross validation
    n0 = math.floor(T / k)
    gamma = 0.01
    Thresh = 1E-5

    if (len(PAi)):
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 3  ###
        theta = 1 / (width ** 2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # set the kernel for PA
        Kpa = np.matlib.ones((T, T))

        for m in range(len(PAi)):
            PA = Data[:, parameters['dlabel'][PAi[m]]]
            G = np.sum((np.multiply(PA, PA)), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 3  ###
            theta = 1 / (width ** 2 * PA.shape[1])
            Kpa = np.multiply(Kpa, kernel(PA, PA, (theta, 1))[0])

        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0  # kernel matrix for PA

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0: T, n0: T]
                Kx_tr_te = Kx[n0: T, 0: n0]
                Kpa_te = Kpa[0:n0, 0: n0]
                Kpa_tr = Kpa[n0: T, n0: T]
                Kpa_tr_te = Kpa[n0: T, 0: n0]
                nv = n0  # sample size of validated data
            if (kk == k - 1):
                Kx_te = Kx[kk * n0:T, kk * n0: T]
                Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                    np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                       np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.matlib.eye(n1))
            tmp2 = tmp1 * Kx_tr * tmp1
            tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * var_lambda ** 2 / gamma * tmp2) * tmp1
            A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                 + 2 * n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

            B = n1 * var_lambda ** 2 / gamma * tmp2 + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 3  ###
        theta = 1 / (width ** 2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                nv = n0
            if (kk == k - 1):
                Kx_te = Kx[kk * n0: T, kk * n0: T]
                Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


def local_score_marginal_general(Data, Xi, PAi, parameters):
    # calculate the local score by negative marginal likelihood
    # based on a regression model in RKHS

    T = Data.shape[0]
    X = Data[:, Xi]
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T ** 2, 1))
    width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
    width = width * 2.5  # kernel width
    theta = 1 / (width ** 2)
    H = np.matlib.eye(T) - np.matlib.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1E-5
    eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False)  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if (len(PAi)):
        PA = Data[:, PAi]

        widthPA = np.matlib.empty((PA.shape[1], 1))
        # set the kernel for PA
        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            widthPA[m] = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        widthPA = widthPA * 2.5  # kernel width

        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        logtheta0 = np.vstack([np.log(widthPA), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    else:
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        PA = np.matlib.zeros((T, 1))
        logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    score = nlml  # negative log-likelihood
    return score


def local_score_marginal_multi(Data, Xi, PAi, parameters):
    # calculate the local score by negative marginal likelihood
    # based on a regression model in RKHS
    # for variables with multi-variate dimensions
    #
    # parameters.d_label: index of each variable
    T = Data.shape[0]
    X = Data[:, parameters['dlabel'][Xi]]
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T ** 2, 1))
    widthX = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
    widthX = widthX * 2.5  # kernel width
    theta = 1 / (widthX ** 2)
    H = np.matlib.eye(T) - np.matlib.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1E-5
    eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False)  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if (len(PAi)):
        widthPA_all = np.matlib.empty((1, 0))
        # set the kernel for PA
        PA_all = np.matlib.empty((Data.shape[0], 0))
        for m in range(len(PAi)):
            PA = Data[:, parameters['dlabel'][PAi[m]]]
            PA_all = np.hstack([PA_all, PA])
            G = np.sum((np.multiply(PA, PA)), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            widthPA = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
            widthPA_all = np.hstack([widthPA_all, widthPA * np.matlib.ones((1, np.size(parameters['dlabel'][PAi[m]])))])
        widthPA_all = widthPA_all * 2.5  # kernel width
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        logtheta0 = np.vstack([np.log(widthPA_all.T), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA_all,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA_all,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    else:
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        PA = np.matlib.zeros((T, 1))
        logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    score = nlml  # negative log-likelihood
    return score