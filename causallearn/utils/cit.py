from math import sqrt, log
from scipy.stats import norm, chi2
import numpy as np
from causallearn.utils.KCI.KCI import KCI_UInd, KCI_CInd
from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel
from causallearn.utils.KCI.LinearKernel import LinearKernel
from causallearn.utils.PCUtils import Helper



def kci(data, X, Y, condition_set=None, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian',
        est_width='empirical', polyd=2, kwidthx=None, kwidthy=None, kwidthz=None):

    if condition_set is None or len(condition_set) < 1:
        return kci_ui(data[np.ix_(range(data.shape[0]), [X])], data[np.ix_(range(data.shape[0]), [Y])],
                      kernelX, kernelY, est_width, polyd, kwidthx, kwidthy)
    else:
        return kci_ci(data[np.ix_(range(data.shape[0]), [X])], data[np.ix_(range(data.shape[0]), [Y])],
                      data[np.ix_(range(data.shape[0]), list(condition_set))],
                      kernelX, kernelY, kernelZ, est_width, polyd, kwidthx, kwidthy, kwidthz)


def kci_ui(X, Y, kernelX='Gaussian', kernelY='Gaussian', est_width='empirical', polyd=2, kwidthx=None, kwidthy=None):
    '''
     To test if x and y are unconditionally independent
       Parameters
       ----------
       kernelX: kernel function for input data x
           'Gaussian': Gaussian kernel
           'Polynomial': Polynomial kernel
           'Linear': Linear kernel
       kernelY: kernel function for input data y
       est_width: set kernel width for Gaussian kernels
           'empirical': set kernel width using empirical rules (default)
           'median': set kernel width using the median trick
       polyd: polynomial kernel degrees (default=2)
       kwidthx: kernel width for data x (standard deviation sigma)
       kwidthy: kernel width for data y (standard deviation sigma)
    '''

    kci_uind = KCI_UInd(kernelX, kernelY, est_width=est_width, polyd=polyd, kwidthx=kwidthx, kwidthy=kwidthy )
    pvalue, _ = kci_uind.compute_pvalue(X, Y)
    return pvalue

def kci_ci(X, Y, Z, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', est_width='empirical', polyd=2, kwidthx=None, kwidthy=None, kwidthz=None):
    '''
     To test if x and y are conditionally independent given z
       Parameters
       ----------
       kernelX: kernel function for input data x
           'Gaussian': Gaussian kernel
           'Polynomial': Polynomial kernel
           'Linear': Linear kernel
       kernelY: kernel function for input data y
       kernelZ: kernel function for input data z
       est_width: set kernel width for Gaussian kernels
           'empirical': set kernel width using empirical rules (default)
           'median': set kernel width using the median trick
       polyd: polynomial kernel degrees (default=2)
       kwidthx: kernel width for data x (standard deviation sigma, default None)
       kwidthy: kernel width for data y (standard deviation sigma)
       kwidthz: kernel width for data y (standard deviation sigma)
    '''

    kci_cind = KCI_CInd(kernelX, kernelY, kernelZ, est_width=est_width, polyd=polyd, kwidthx=kwidthx, kwidthy=kwidthy, kwidthz=kwidthz)
    pvalue, _ = kci_cind.compute_pvalue(X, Y, Z)
    return pvalue

def mv_fisherz(mvdata, X, Y, condition_set, sample_size = None):
    '''
    Perform an independence test using Fisher-Z's test for data with missing values

    Parameters
    ----------
    mvdata : data with missing values
    X, Y and condition_set : data matrices of size number_of_samples * dimensionality

    Returns
    -------
    p : the p-value of the test
    '''
    var = list((X, Y) + condition_set)
    sub_corr_matrix, del_sample_size = get_sub_correlation_matrix(mvdata[:, var])  # the columns represent variables
    sample_size = del_sample_size
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(condition_set) - 3) * abs(Z)
    p = 1 - norm.cdf(abs(X))
    return p


def mc_fisherz(mdata, skel, prt_m, X, Y, condition_set, sample_size):
    """Perform an independent test using Fisher-Z's test with test-wise deletion and missingness correction
    If it is not the case which requires a correction, then call function mvfisherZ(...)
    :param prt_m: dictionary, with elements:
        - m: missingness indicators which are not MCAR
        - prt: parents of the missingness indicators
    """

    ## Check whether whether there is at least one common child of X and Y
    if not Helper.cond_perm_c(X, Y, condition_set, prt_m, skel):
        return mv_fisherz(mdata, X, Y, condition_set, sample_size)

    ## *********** Step 1 ***********
    # Learning generaive model for {X, Y, S} to impute X, Y, and S

    ## Get parents the {xyS} missingness indicators with parents: prt_m
    # W is the variable which can be used for missingness correction
    W_indx_ = Helper.get_prt_mvars(var=list((X, Y) + condition_set), prt_m=prt_m)

    if len(W_indx_) == 0:  # When there is no variable can be used for correction
        return mv_fisherz(mdata, X, Y, condition_set, sample_size)

    ## Get the parents of W missingness indicators
    W_indx = Helper.get_prt_mw(W_indx_, prt_m)

    ## Prepare the W for regression
    # Since the XYS will be regressed on W,
    # W will not contain any of XYS
    var = list((X, Y) + condition_set)
    W_indx = list(set(W_indx) - set(var))

    if len(W_indx) == 0:  # When there is no variable can be used for correction
        return mv_fisherz(mdata, X, Y, condition_set, sample_size)

    ## Learn regression models with test-wise deleted data
    involve_vars = var + W_indx
    tdel_data = Helper.test_wise_deletion(mdata[:, involve_vars])
    effective_sz = len(tdel_data[:, 0])
    regMs, rss = Helper.learn_regression_model(tdel_data, num_model=len(var))

    ## *********** Step 2 ***********
    # Get the data of the predictors, Ws
    Ws = Helper.get_predictor_ws(mdata[:, involve_vars], num_test_var=len(var), effective_sz=effective_sz)

    ## *********** Step 3 ***********
    # Generate the virtual data follows the full data distribution P(X, Y, S)
    data_vir = Helper.gen_vir_data(regMs, rss, Ws, len(var), effective_sz)

    if len(var) > 2:
        cond_set_bgn_0 = np.arange(2, len(var))
    else:
        cond_set_bgn_0 = []

    return mv_fisherz(data_vir, 0, 1, tuple(cond_set_bgn_0), effective_sz)

def fisherz(data, X, Y, condition_set, correlation_matrix=None):
    '''
    Perform an independence test using Fisher-Z's test

    Parameters
    ----------
    data : data matrices
    X, Y and condition_set : data matrices of size number_of_samples * dimensionality
    correlation_matrix : correlation matrix; 
                         None means without the parameter of correlation matrix

    Returns
    -------
    p : the p-value of the test
    '''
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)
    sample_size = data.shape[0]
    var = list((X, Y) + condition_set)
    sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(condition_set) - 3) * abs(Z)
    p = 1 - norm.cdf(abs(X))
    return p


def chisq(data, X, Y, conditioning_set):
    return chisq_or_gsq_test(data=data, X=X, Y=Y, conditioning_set=conditioning_set)


def gsq(data, X, Y, conditioning_set):
    return chisq_or_gsq_test(data=data, X=X, Y=Y, conditioning_set=conditioning_set, G_sq=True)


def chisq_or_gsq_test(data, X, Y, conditioning_set, G_sq=False):
    '''
    Perform an independence test using chi-square test or G-square test

    Parameters
    ----------
    data : data matrices
    X, Y and condition_set : data matrices of size number_of_samples * dimensionality
    G_sq : True means using G-square test;
           False means using chi-square test

    Returns
    -------
    p : the p-value of the test
    '''

    # Step 1: Subset the data
    categories_list = [np.unique(data[:, i]) for i in
                       list(conditioning_set)]  # Obtain the categories of each variable in conditioning_set
    value_config_list = cartesian_product(
        categories_list)  # Obtain all the possible value configurations of the conditioning_set (e.g., [[]] if categories_list == [])

    max_categories = int(
        np.max(data)) + 1  # Used to fix the size of the contingency table (before applying Fienberg's method)

    sum_of_chi_square = 0  # initialize a zero chi_square statistic
    sum_of_df = 0  # initialize a zero degree of freedom

    def recursive_and(L):
        "A helper function for subsetting the data using the conditions in L of the form [(variable, value),...]"
        if len(L) == 0:
            return data
        else:
            condition = data[:, L[0][0]] == L[0][1]
            i = 1
            while i < len(L):
                new_conjunct = data[:, L[i][0]] == L[i][1]
                condition = new_conjunct & condition
                i += 1
            return data[condition]

    for value_config in range(len(value_config_list)):
        L = list(zip(conditioning_set, value_config_list[value_config]))
        sub_data = recursive_and(L)[:, [X,
                                        Y]]  # obtain the subset dataset (containing only the X, Y columns) with only rows specifed in value_config

        # Step 2: Generate contingency table (applying Fienberg's method)
        def make_ctable(D, cat_size):
            x = np.array(D[:, 0], dtype=np.dtype(int))
            y = np.array(D[:, 1], dtype=np.dtype(int))
            bin_count = np.bincount(cat_size * x + y)  # Perform linear transformation to obtain frequencies
            diff = (cat_size ** 2) - len(bin_count)
            if diff > 0:  # The number of cells generated by bin_count can possibly be less than cat_size**2
                bin_count = np.concatenate(
                    (bin_count, np.zeros(diff)))  # In that case, we concatenate some zeros to fit cat_size**2
            ctable = bin_count.reshape(cat_size, cat_size)
            ctable = ctable[~np.all(ctable == 0, axis=1)]  # Remove rows consisted entirely of zeros
            ctable = ctable[:, ~np.all(ctable == 0, axis=0)]  # Remove columns consisted entirely of zeros
            return ctable

        ctable = make_ctable(sub_data, max_categories)

        # Step 3: Calculate chi-square statistic and degree of freedom from the contingency table
        row_sum = np.sum(ctable, axis=1)
        col_sum = np.sum(ctable, axis=0)
        expected = np.outer(row_sum, col_sum) / sub_data.shape[0]
        if G_sq == False:
            chi_sq_stat = np.sum(((ctable - expected) ** 2) / expected)
        else:
            div = np.divide(ctable, expected)
            div[div == 0] = 1  # It guarantees that taking natural log in the next step won't cause any error
            chi_sq_stat = 2 * np.sum(ctable * np.log(div))
        df = (ctable.shape[0] - 1) * (ctable.shape[1] - 1)

        sum_of_chi_square += chi_sq_stat
        sum_of_df += df

    # Step 4: Compute p-value from chi-square CDF
    if sum_of_df == 0:
        return 1
    else:
        return chi2.sf(sum_of_chi_square, sum_of_df)


def cartesian_product(lists):
    "Return the Cartesian product of lists (List of lists)"
    result = [[]]
    for pool in lists:
        result = [x + [y] for x in result for y in pool]
    return result


def get_index_mv_rows(mvdata):
    nrow, ncol = np.shape(mvdata)
    bindxRows = np.ones((nrow,), dtype=bool)
    indxRows = np.array(list(range(nrow)))
    for i in range(ncol):
        bindxRows = np.logical_and(bindxRows, ~np.isnan(mvdata[:, i]))
    indxRows = indxRows[bindxRows]
    return indxRows


def get_sub_correlation_matrix(mvdata):
    indxRows = get_index_mv_rows(mvdata)
    submatrix = np.corrcoef(mvdata[indxRows, :], rowvar=False)
    sample_size = len(indxRows)
    return submatrix, sample_size
