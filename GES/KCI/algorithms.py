import sys
import numpy.matlib
import scipy.sparse.linalg
from GES.KCI.gpml import *

def minimize(X, f, length, *varargin):
    # Minimize a differentiable multivariate function.
    #
    # Usage: X, fX, i = minimize(X, f, length, P1, P2, P3, ... )
    #
    # where the starting point is given by "X" (D by 1), and the function named in
    # the string "f", must return a function value and a vector of partial
    # derivatives of f wrt X, the "length" gives the length of the run: if it is
    # positive, it gives the maximum number of line searches, if negative its
    # absolute gives the maximum allowed number of function evaluations. You can
    # (optionally) give "length" a second component, which will indicate the
    # reduction in function value to be expected in the first line-search (defaults
    # to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
    #
    # The function returns when either its length is up, or if no further progress
    # can be made (ie, we are at a (local) minimum, or so close that due to
    # numerical problems, we cannot get any closer). NOTE: If the function
    # terminates within a few iterations, it could be an indication that the
    # function values and derivatives are not consistent (ie, there may be a bug in
    # the implementation of your "f" function). The function returns the found
    # solution "X", a vector of function values "fX" indicating the progress made
    # and "i" the number of iterations (line searches or function evaluations,
    # depending on the sign of "length") used.
    #
    # The Polack-Ribiere flavour of conjugate gradients is used to compute search
    # directions, and a line search using quadratic and cubic polynomial
    # approximations and the Wolfe-Powell stopping criteria is used together with
    # the slope ratio method for guessing initial step sizes. Additionally a bunch
    # of checks are made to make sure that exploration is taking place and that
    # extrapolation will not be unboundedly large.
    #
    # See also: checkgrad

    INT = 0.1  # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0  # extrapolate maximum 3 times the current step-size
    MAX = 20  # max 20 function evaluations per line search
    RATIO = 10  # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2  # SIG and RHO are the constants controlling the Wolfe-
    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the
    # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with RHO.

    # The code falls naturally into 3 parts, after the initial line search is
    # started in the direction of steepest descent. 1) we first enter a while loop
    # which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
    # have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
    # enter the second loop which takes p2, p3 and p4 chooses the subinterval
    # containing a (local) minimum, and interpolates it, unil an acceptable point
    # is found (Wolfe-Powell conditions). Note, that points are always maintained
    # in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
    # conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
    # was a problem in the previous line-search. Return the best value so far, if
    # two consecutive line-searches fail, or whenever we run out of function
    # evaluations or line-searches. During extrapolation, the "f" function may fail
    # either with an error or returning Nan or Inf, and minimize should handle this
    # gracefully.

    if np.size(length) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1

    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    i = 0  # zero the run length counter
    ls_failed = 0  # no previous line search has failed
    temp = [f, X]
    temp.extend(varargin)
    temp.extend([None, 2])  # 标记输出结果为 2 个元素
    f0, df0 = feval(temp)  # get function value and gradient
    fX = f0
    i = i + (1 if length < 0 else 0)  # count epochs?!
    s = -df0
    d0 = (-s.T * s)[0, 0]  # initial search direction (steepest) and slope
    x3 = red / (1 - d0)  # initial step is red/(|s|+1)

    while i < abs(length):  # while not finished
        i = i + (1 if length > 0 else 0)  # count iterations?!
        X0 = X  # make a copy of current values
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while 1:  # keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = False

            while (not success and M > 0):
                try:
                    M = M - 1
                    i = i + (1 if length < 0 else 0)  # count epochs?!
                    temp = [f, X + x3 * s]
                    temp.extend(varargin)
                    temp.extend([None, 2])  # # 标记输出结果为 2 个元素
                    f3, df3 = feval(temp)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3)) or np.any(np.isinf(df3)):
                        raise Exception('')
                    success = True
                except Exception as e:  # catch any error which occured in f
                    x3 = (x2 + x3) / 2  # bisect and try again

            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            d3 = (df3.T * s)[0, 0]  # new slope
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:  # are we done extrapolating?
                break

            x1 = x2  # move point 2 to point 1
            f1 = f2
            d1 = d2
            x2 = x3  # move point 3 to point 2
            f2 = f3
            d2 = d3
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)  # make cubic extrapolation
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * (x2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))  # num. error possible, ok!
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:  # num prob | wrong sign?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 > x2 * EXT:  # new point beyond extrapolation limit?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 < x2 + INT * (x2 - x1):  # new point too close to previous point?
                x3 = x2 + INT * (x2 - x1)
        # end extrapolation

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:  # keep interpolating
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:  # choose subinterval
                x4 = x3  # move point 3 to point 4
                f4 = f3
                d4 = d3
            else:
                x2 = x3  # move point 3 to point 2
                f2 = f3
                d2 = d3

            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))  # quadratic interpolation
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)  # cubic interpolation
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A  # num. error possible, ok!

            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2  # if we had a numerical problem then bisect

            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))  # don't accept too close
            temp = [f, X + x3 * s]
            temp.extend(varargin)
            temp.extend([None, 2])  # # 标记输出结果为 2 个元素
            f3, df3 = feval(temp)
            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            M = M - 1
            i = i + (1 if length < 0 else 0)  # count epochs?!
            d3 = (df3.T * s)[0, 0]  # new slope
        # end interpolation

        if (abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0):  # if line search succeeded
            X = X + x3 * s
            f0 = f3
            fX = np.vstack([fX, f0])  # update variables
            s = ((df3.T * df3)[0, 0] - df0.T * df3[0, 0]) / (df0.T * df0)[0, 0] * s - df3  # Polack-Ribiere CG direction
            df0 = df3  # swap derivatives
            d3 = d0
            d0 = (df0.T * s)[0, 0]
            if (d0 > 0):  # new slope must be negative
                s = -df0
                d0 = -(s.T * s)[0, 0]
            x3 = x3 * min(RATIO, d3 / (d0 - sys.float_info.min))  # slope ratio but max RATIO
            ls_failed = 0  # this line search did not fail
        else:
            X = X0  # restore best point so far
            f0 = F0
            df0 = dF0
            if (ls_failed or i > abs(length)):  # line search failed twice in a row
                break  # or we ran out of time, so we give up
            s = -df0  # try steepest
            d0 = -(s.T * s)[0, 0]
            x3 = 1 / (1 - d0)
            ls_failed = 1  # this line search failed
    return X, fX, i


def kernel(x, xKern, theta):
    # KERNEL Compute the rbf kernel
    n2 = dist2(x, xKern)
    if (theta[0] == 0):
        theta[0] = 2 / np.median(n2[np.where(np.tril(n2) > 0)])
        theta_new = theta[0]
    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]

    return kx, bw_new

def dist2(x, c):
    # DIST2	Calculates squared distance between two sets of points.
    #
    # Description
    # D = DIST2(X, C) takes two matrices of vectors and calculates the
    # squared Euclidean distance between them.  Both matrices must be of
    # the same column dimension.  If X has M rows and N columns, and C has
    # L rows and N columns, then the result has M rows and L columns.  The
    # I, Jth entry is the  squared distance from the Ith row of X to the
    # Jth row of C.
    #
    # See also
    # GMMACTIV, KMEANS, RBFFWD
    #

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if (dimx != dimc):
        raise Exception('Data dimension does not match dimension of centres')

    n2 = (np.matlib.ones((ncentres, 1)) * np.sum(np.multiply(x, x).T, axis=0)).T + \
         np.matlib.ones((ndata, 1)) * np.sum(np.multiply(c, c).T, axis=0) - \
         2 * (x * c.T)

    # Rounding errors occasionally cause negative entries in n2
    n2[np.where(n2 < 0)] = 0
    return n2

def pdinv(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A).T
        invU = np.eye(numData).dot(np.linalg.inv(U))
        Ainv = invU.dot(invU.T)
    except numpy.linalg.LinAlgError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
    except Exception as e:
        raise e
    return np.matlib.asmatrix(Ainv)

def eigdec(x, N, evals_only=False):
    # EIGDEC	Sorted eigendecomposition
    #
    #	Description
    #	 EVALS = EIGDEC(X, N computes the largest N eigenvalues of the
    #	matrix X in descending order.  [EVALS, EVEC] = EIGDEC(X, N) also
    #	computes the corresponding eigenvectors.
    #
    #	See also
    #	PCA, PPCA
    #

    if (N != np.round(N) or N < 1 or N > x.shape[1]):
        raise Exception('Number of PCs must be integer, >0, < dim')

    # Find the eigenvalues of the data covariance matrix
    if (evals_only):
        # Use eig function as always more efficient than eigs here
        temp_evals, _ = np.linalg.eig(x)
    else:
        # Use eig function unless fraction of eigenvalues required is tiny
        if ((N / x.shape[1]) > 0.04):
            temp_evals, temp_evec = np.linalg.eig(x)
        else:
            temp_evals, temp_evec = scipy.sparse.linalg.eigs(x, k=N, which='LM')

    # Eigenvalues nearly always returned in descending order, but just to make sure.....
    evals = np.sort(-temp_evals)
    perm = np.argsort(-temp_evals)
    evals = -evals[0:N]

    if (not evals_only):
        if (np.all(evals == temp_evals[0:N])):
            # Originals were in order
            evec = temp_evec[:, 0: N]
        else:
            # Need to reorder the eigenvectors
            evec = np.empty_like(temp_evec[:, 0: N])
            for i in range(N):
                evec[:, i] = temp_evec[:, perm[i]]

        return evals.astype(float), evec.astype(float)
    else:
        return evals.astype(float)
def gpr_multi_new(logtheta=None, covfunc=None, x=None, y=None, xstar=None, nargout=1):
    # Here we change the function gpr to gpr_multi, in which y contains a set
    # of vectors on which we do repression from x

    # gpr - Gaussian process regression, with a named covariance function. Two
    # modes are possible: training and prediction: if no test data are given, the
    # function returns minus the log likelihood and its partial derivatives with
    # respect to the hyperparameters; this mode is used to fit the hyperparameters.
    # If test data are given, then (marginal) Gaussian predictions are computed,
    # whose mean and variance are returned. Note that in cases where the covariance
    # function has noise contributions, the variance returned in S2 is for noisy
    # test targets; if you want the variance of the noise-free latent function, you
    # must substract the noise variance.
    #
    # usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y)
    #    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar)
    #
    # where:
    #
    #   logtheta is a (column) vector of log hyperparameters
    #   covfunc  is the covariance function
    #   x        is a n by D matrix of training inputs
    #   y        is a (column) vector (of size n) of targets
    #   xstar    is a nn by D matrix of test inputs
    #   nlml     is the returned value of the negative log marginal likelihood
    #   dnlml    is a (column) vector of partial derivatives of the negative
    #                 log marginal likelihood wrt each log hyperparameter
    #   mu       is a (column) vector (of size nn) of prediced means
    #   S2       is a (column) vector (of size nn) of predicted variances
    #
    # For more help on covariance functions, see "covFunctions".

    if type(covfunc) == str:
        covfunc = [covfunc]  # convert to cell if needed
    n, D = x.shape
    n, m = y.shape
    if eval(feval(covfunc)) != logtheta.shape[0]:
        raise Exception('Error: Number of parameters do not agree with covariance function')

    temp = list(covfunc.copy())
    temp.append(logtheta)
    temp.append(x)
    K = feval(temp)  # compute training set covariance matrix

    L = np.linalg.cholesky(K)  # cholesky factorization of the covariance
    alpha = solve_chol(L.T, y)

    if (
            logtheta is not None and covfunc is not None and x is not None and y is not None and xstar is None):  # if no test cases, compute the negative log marginal likelihood
        out1 = 0.5 * np.trace(y.T * alpha) + m * np.sum(np.log(np.diag(L)), axis=0) + 0.5 * m * n * np.log(
            2 * np.pi)
        if nargout == 2:  # ... and if requested, its partial derivatives
            out2 = np.matlib.zeros((logtheta.shape[0], 1))  # set the size of the derivative vector
            W = m * (np.linalg.inv(L.T) * (
                        np.linalg.inv(L) * np.matlib.eye(n))) - alpha * alpha.T  # precompute for convenience
            for i in range(len(out2) - 1, len(out2)):
                temp = list(covfunc.copy())
                temp.append(logtheta)
                temp.append(x)
                temp.append(i)
                out2[i] = np.sum(np.multiply(W, feval(temp))) / 2
    else:  # ... otherwise compute (marginal) test predictions ...
        temp = list(covfunc.copy())
        temp.append(logtheta)
        temp.append(x)
        temp.append(xstar)
        temp.append(2)  # nargout == 2
        Kss, Kstar = feval(temp)  # test covariances
        out1 = Kstar.T * alpha  # predicted means

        if nargout == 2:
            v = np.linalg.inv(L) * Kstar
            v = np.asmatrix(v)
            out2 = Kss - np.sum(np.multiply(v, v), axis=0).T

    if nargout == 1:
        return out1
    else:
        return out1, out2

def feval(parameters):
    if parameters[0] == 'covNoise':
        if (len(parameters) == 1):
            return covNoise()
        elif (len(parameters) == 2):
            return covNoise(parameters[1])
        elif (len(parameters) == 3):
            return covNoise(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covNoise(parameters[1], parameters[2], parameters[3])
        else:
            return covNoise(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSEard':
        if (len(parameters) == 1):
            return covSEard()
        elif (len(parameters) == 2):
            return covSEard(parameters[1])
        elif (len(parameters) == 3):
            return covSEard(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSEard(parameters[1], parameters[2], parameters[3])
        else:
            return covSEard(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSum':
        if (len(parameters) == 1):
            return covSum()
        elif (len(parameters) == 2):
            return covSum(parameters[1])
        elif (len(parameters) == 3):
            return covSum(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSum(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
    elif parameters[0] == 'gpr_multi_new':
        if (len(parameters) == 1):
            return gpr_multi_new()
        elif (len(parameters) == 2):
            return gpr_multi_new(parameters[1])
        elif (len(parameters) == 3):
            return gpr_multi_new(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
        elif (len(parameters) == 7):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],
                                 parameters[6])
    else:
        raise Exception('请选择已定义的函数')