import numpy as np
import numpy.matlib
from GES.GesUtils import *

K = np.matlib.empty((0, 0))

def covNoise(logtheta=None, x=None, z=None, nargout=1):
    # Independent covariance function, ie "white noise", with specified variance.
    # The covariance function is specified as:
    #
    # k(x^p,x^q) = s2 * \delta(p,q)
    #
    # where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
    # which is 1 iff p=q and zero otherwise. The hyperparameter is
    #
    # logtheta = [ log(sqrt(s2)) ]
    #
    # For more help on design of covariance functions, see "covFunctions".

    if (logtheta is None and x is None and z is None): # report number of parameters
        A = '1'

        return A

    s2 = np.exp(2 * logtheta)[0, 0] # noise variance

    if (logtheta is not None and x is not None and z is None): # compute covariance matrix
        A = s2 * np.matlib.eye(x.shape[0])
    elif (nargout == 2): # compute test set covariances
        A = s2
        B = 0   # zeros cross covariance by independence
    else: # compute derivative matrix
        A = 2 * s2 * np.matlib.eye(x.shape[0])


    if (nargout == 2):
        return A, B
    else:
        return A


def covSEard(loghyper=None, x=None, z=None, nargout=1):
    # Squared Exponential covariance function with Automatic Relevance Detemination
    # (ARD) distance measure. The covariance function is parameterized as:
    #
    # k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    #
    # where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    # D is the dimension of the input space and sf2 is the signal variance. The
    # hyperparameters are:
    #
    # loghyper = [ log(ell_1)
    #              log(ell_2)
    #               .
    #              log(ell_D)
    #              log(sqrt(sf2)) ]
    #
    # For more help on design of covariance functions, see "covFunctions".
    global K

    if (loghyper is None and x is None and z is None):
        A = '(D+1)'

        return A # report number of parameters

    n ,D = x.shape
    loghyper = loghyper.T.tolist()[0]
    ell = np.exp(loghyper[0:D]) # characteristic length scale
    sf2 = np.exp(2 * loghyper[D]) # signal variance

    if (loghyper is not None and x is not None):
        K = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T ) /2)
        A = K
    elif nargout == 2: # compute test set covariances
        A = sf2 * np.matlib.ones((z, 1))
        B = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T, np.matlib.diag( 1 /ell ) *z) / 2)
    else:
        # check for correct dimension of the previously calculated kernel matrix
        if (K.shape[0] != n or K.shape[1] != n):
            K = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T ) /2)

        if z <= D:  # length scale parameters
            A = np.multiply(K, sq_dist(x[:, z].T / ell[z]))
        else: # magnitude parameter
            A = 2 * K
            K = np.matlib.empty((0, 0))

    if (nargout == 2):
        return A, B
    else:
        return A

def sq_dist(a, b=None, Q=None):
    # sq_dist - a function to compute a matrix of all pairwise squared distances
    # between two sets of vectors, stored in the columns of the two matrices, a
    # (of size D by n) and b (of size D by m). If only a single argument is given
    # or the second matrix is empty, the missing matrix is taken to be identical
    # to the first.
    #
    # Special functionality: If an optional third matrix argument Q is given, it
    # must be of size n by m, and in this case a vector of the traces of the
    # product of Q' and the coordinatewise squared distances is returned.
    #
    # NOTE: The program code is written in the C language for efficiency and is
    # contained in the file sq_dist.c, and should be compiled using matlabs mex
    # facility. However, this file also contains a (less efficient) matlab
    # implementation, supplied only as a help to people unfamiliar with mex. If
    # the C code has been properly compiled and is avaiable, it automatically
    # takes precendence over the matlab code in this file.
    #
    # Usage: C = sq_dist(a, b)
    #    or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])
    #    or: c = sq_dist(a, b, Q)
    # where the b matrix may be empty.
    #
    # where a is of size D by n, b is of size D by m (or empty), C and Q are of
    # size n by m and c is of size D by 1.

    if b is None or len(b) == 0: # input arguments are taken to be identical if b is missing or empty
        b = a

    D, n = a.shape
    d, m = b.shape

    if d != D:
        raise Exception('Error: column lengths must agree.')

    if Q is None:
        C = np.matlib.zeros((n, m))
        for d in range(D):
            temp = np.tile(b[d, :], (n, 1)) - np.tile(a[d, :].T, (1, m))
            C = C + np.multiply(temp, temp)
    else:
        if (n,m) == Q.shape:
            C = np.matlib.zeros((D, 1))
            for d in range(D):
                temp = np.tile(b[d,:], (n, 1)) - np.tile(a[d,:].T, (1, m))
                temp = np.multiply(temp, temp)
                temp = np.multiply(temp, Q)
                C[d] = np.sum(temp)
        else:
            raise Exception('Third argument has wrong size.')
    return C

def solve_chol(A, B):
    # solve_chol - solve linear equations from the Cholesky factorization.
    # Solve A*X = B for X, where A is square, symmetric, positive definite. The
    # input to the function is R the Cholesky decomposition of A and the matrix B.
    # Example: X = solve_chol(chol(A),B);
    #
    # NOTE: The program code is written in the C language for efficiency and is
    # contained in the file solve_chol.c, and should be compiled using matlabs mex
    # facility. However, this file also contains a (less efficient) matlab
    # implementation, supplied only as a help to people unfamiliar with mex. If
    # the C code has been properly compiled and is avaiable, it automatically
    # takes precendence over the matlab code in this file.

    if A is None or B is None:
        raise Exception('Wrong number of arguments.')

    if (A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]):
        raise Exception('Wrong sizes of matrix arguments.')

    res = np.linalg.inv(A) * (np.linalg.inv(A.T) * B)
    return res

def covSum(covfunc, logtheta=None, x=None, z=None, nargout=1):
    # covSum - compose a covariance function as the sum of other covariance
    # functions. This function doesn't actually compute very much on its own, it
    # merely does some bookkeeping, and calls other covariance functions to do the
    # actual work.
    #
    # For more help on design of covariance functions, see "covFunctions".

    j = []
    for i in range(len(covfunc)):  # iterate over covariance functions
        f = covfunc[i]
        j.append([feval([f])])

    if (logtheta is None and x is None and z is None):  # report number of parameters
        A = j[0][0]
        for i in range(1, len(covfunc)):
            A = A + '+' + j[i][0]

        return A

    n, D = x.shape

    v = []  # v vector indicates to which covariance parameters belong
    for i in range(len(covfunc)):
        for k in range(eval(j[i][0])):
            v.append(i)
    v = np.asarray(v)

    if (logtheta is not None and x is not None and z is None):  # compute covariance matrix
        A = np.matlib.zeros((n, n))  # allocate space for covariance matrix
        for i in range(len(covfunc)):  # iteration over summand functions
            f = covfunc[i]
            temp = [f]
            t = logtheta[np.where(v == i)]
            temp.append(t[0] if len(t) == 1 else t)
            temp.append(x)
            A = A + feval(temp)

    if (
            logtheta is not None and x is not None and z is not None):  # compute derivative matrix or test set covariances
        if nargout == 2:  # compute test set cavariances
            A = np.matlib.zeros((z, 1))
            B = np.matlib.zeros((x.shape[0], z))  # allocate space
            for i in range(len(covfunc)):
                f = covfunc[i]
                temp = [f]
                t = logtheta[np.where(v == i)]
                temp.append(t[0] if len(t) == 1 else t)
                temp.append(x)
                temp.append(z)
                temp.append(2)  # nargout 赋值为 2
                AA, BB = feval(temp)  # compute test covariances and accumulate
                A = A + AA
                B = B + BB
        else:  # compute derivative matrices
            i = v[z]  # which covariance function
            j = np.sum(np.where(v[0:z] == i, 1, 0))  # which parameter in that covariance
            f = covfunc[i]
            temp = [f]
            t = logtheta[np.where(v == i)]
            temp.append(t[0] if len(t) == 1 else t)
            temp.append(x)
            temp.append(j)
            A = feval(temp)

    if (nargout == 2):
        return A, B
    else:
        return A

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
    else:
        raise Exception('请选择已定义的函数')