import numpy as np
from pdb import set_trace
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2
from scipy.linalg import eigh

from .logger import LOGGER


class Chi2RankTest(object):

    def __init__(self, data, N_scaling=1):

        self.data = data
        self.data = self.data - self.data.mean(axis=0)
        self.data = self.data/self.data.std(axis=0)

        self.N = data.shape[0]
        self.N_scaling = N_scaling
        self.unnormalized_crosscovs = self.data.T@self.data # data are zero mean
        self.cca_cache_dict = {}

    def get_cachekey(self, pcols_, qcols_):

        pcols = pcols_.copy()
        qcols = qcols_.copy()
        pcols.sort()
        qcols.sort()

        key = ''
        for i in range(self.data.shape[1]):
            if i in pcols and i in qcols:
                key = key+str(3)
            elif i in pcols and not i in qcols:
                key = key+str(2)
            elif not i in pcols and i in qcols:
                key = key+str(1)
            else:
                key = key+str(0)

        return key

    def test(self, pcols, qcols, r, alpha):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''



        cachekey = self.get_cachekey(pcols, qcols)

        if cachekey in self.cca_cache_dict:
            cancorr = self.cca_cache_dict[cachekey]
        else:

            X = self.data[:, pcols]
            Y = self.data[:, qcols]

            unnormalized_crosscovs = [self.unnormalized_crosscovs[pcols,:][:,pcols], self.unnormalized_crosscovs[pcols,:][:,qcols], \
                         self.unnormalized_crosscovs[qcols,:][:,pcols], self.unnormalized_crosscovs[qcols,:][:,qcols]]

            try:
                comps = kcca_modified([X, Y], reg=0.,
                    numCC=None, kernelcca=False, ktype='linear',
                    gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                cancorr = cancorr[:,0,1]
            except:
                LOGGER.debug("calculating cancorr error %s %s, using slower implementation instead", pcols, qcols)
                X_fallback = np.atleast_2d(X).reshape(X.shape[0], -1)
                Y_fallback = np.atleast_2d(Y).reshape(Y.shape[0], -1)
                cancorr = CanCorr(X_fallback, Y_fallback, tolerance=1e-8).cancorr

            self.cca_cache_dict[cachekey] = cancorr

        testStat = 0
        p = len(pcols)
        q = len(qcols)
        
        l = cancorr[r:]
        for li in l:
            li = min(li, 1-1e-15)
            testStat += log(1)-log(1-li*li)
        ratio = 0
        for i in range(r):
            li = cancorr[i]
            ratio += 1/(li*li)-1

        ratio += self.N*self.N_scaling - r - 0.5*(p+q+1)
        testStat = testStat * ratio

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-alpha, dfreedom)
        p = 1 - chi2.cdf(testStat, dfreedom)
        if_fail_to_reject = testStat<=criticalValue

        # due to numerical errors comparing criticalValue with testStat is more accurate 

        return if_fail_to_reject


def kcca_modified(
        data, reg=0.0, numCC=None, kernelcca=False, ktype="linear", gausigma=1.0, degree=2, crosscovs=None
):
    """Set up and solve the kernel CCA eigenproblem"""
    if kernelcca:
        raise NotImplementedError
        #kernel = [
        #    _make_kernel(d, ktype=ktype, gausigma=gausigma, degree=degree) for d in data
        #]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[0] for k in kernel]) if numCC is None else numCC

    # Get the auto- and cross-covariance matrices
    if crosscovs is None:
        crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    n = sum(nFs)
    LH = np.zeros((n, n))
    RH = np.zeros((n, n))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(nDs):
        RH[
        sum(nFs[:i]): sum(nFs[: i + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
        ] = crosscovs[i * (nDs + 1)] + reg * np.eye(nFs[i])

        for j in range(nDs):
            if i != j:
                LH[
                sum(nFs[:j]): sum(nFs[: j + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
                ] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.0
    RH = (RH + RH.T) / 2.0

    maxCC = LH.shape[0]
    try:
        r, Vs = eigh(LH, RH, subset_by_index=[maxCC - numCC, maxCC - 1])
    except TypeError:
        r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]): sum(nFs[: i + 1]), :numCC])
    return comp

def _listdot(d1, d2):
    return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]

def _listcorr(a):
    """Returns pairwise row correlations for all items in array as a list of matrices"""
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [
                    np.nan_to_num(np.corrcoef(ai, aj)[0, 1])
                    for (ai, aj) in zip(a[i].T, a[j].T)
                ]
    return corrs

def recon(data, comp, corronly=False, kernelcca=False):
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp
    
