import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm

from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd
from causallearn.utils.PCUtils import Helper

CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
fisherz = "fisherz"
mv_fisherz = "mv_fisherz"
mc_fisherz = "mc_fisherz"
kci = "kci"
chisq = "chisq"
gsq = "gsq"


def CIT(data, method='fisherz', **kwargs):
    '''
    Parameters
    ----------
    data: numpy.ndarray of shape (n_samples, n_features)
    method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"]
    kwargs: placeholder for future arguments, or for KCI specific arguments now
    '''
    if method == fisherz:
        return FisherZ(data, **kwargs)
    elif method == kci:
        return KCI(data, **kwargs)
    elif method in [chisq, gsq]:
        return Chisq_or_Gsq(data, method_name=method, **kwargs)
    elif method == mv_fisherz:
        return MV_FisherZ(data, **kwargs)
    elif method == mc_fisherz:
        return MC_FisherZ(data, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))

class CIT_Base(object):
    # Base class for CIT, contains basic operations for input check and caching, etc.
    def __init__(self, data, cache_path=None, **kwargs):
        '''
        Parameters
        ----------
        data: data matrix, np.ndarray, in shape (n_samples, n_features)
        cache_path: str, path to save cache .json file. default as None (no io to local file).
        kwargs: for future extension.
        '''
        assert isinstance(data, np.ndarray), "Input data must be a numpy array."
        self.data = data
        self.data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
        self.sample_size, self.num_features = data.shape
        self.cache_path = cache_path
        self.SAVE_CACHE_CYCLE_SECONDS = 30
        self.last_time_cache_saved = time.time()
        self.pvalue_cache = {'data_hash': self.data_hash}
        if not cache_path is None:
            assert cache_path.endswith('.json'), "Cache must be stored as .json file."
            if os.path.exists(cache_path):
                with codecs.open(cache_path, 'r') as fin: self.pvalue_cache = json.load(fin)
                assert self.pvalue_cache['data_hash'] == self.data_hash, "Data hash mismatch."
            else: os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def check_cache_method_consistent(self, method_name, parameters_hash):
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache['method_name'] = method_name # a newly created cache
            self.pvalue_cache['parameters_hash'] = parameters_hash
        else:
            assert self.pvalue_cache['method_name'] == method_name, "CI test method name mismatch." # a loaded cache
            assert self.pvalue_cache['parameters_hash'] == parameters_hash, "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        assert allow_nan or not np.isnan(self.data).any(), "Input data contains NaN. Please check."
        assert allow_inf or not np.isinf(self.data).any(), "Input data contains Inf. Please check."

    def save_to_local_cache(self):
        if not self.cache_path is None and time.time() - self.last_time_cache_saved > self.SAVE_CACHE_CYCLE_SECONDS:
            with codecs.open(self.cache_path, 'w') as fout: fout.write(json.dumps(self.pvalue_cache, indent=2))
            self.last_time_cache_saved = time.time()

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        '''
        reformat the input X, Y and condition_set to
            1. convert to built-in types for json serialization
            2. handle multi-dim unconditional variables (for kernel-based)
            3. basic check for valid input (X, Y no overlap with condition_set)
            4. generate unique and hashable cache key

        Parameters
        ----------
        X: int, or np.*int*
        Y: int, or np.*int*
        condition_set: Iterable<int | np.*int*>

        Returns
        -------
        Xs: List<int>, sorted. may swapped with Ys for cache key uniqueness.
        Ys: List<int>, sorted.
        condition_set: List<int>
        cache_key: string. Unique for <X,Y|S> in any input type or order.
        '''
        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst  = lambda lst: '.'.join(map(str, lst))
            return f'{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}' if len(clist) > 0 else \
                   f'{_strlst(ulist1)};{_strlst(ulist2)}'

        # every time when cit is called, auto save to local cache.
        self.save_to_local_cache()

        METHODS_SUPPORTING_MULTIDIM_DATA = ["kci"]
        if condition_set is None: condition_set = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set = sorted(set(map(int, condition_set)))

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
        if self.method not in METHODS_SUPPORTING_MULTIDIM_DATA:
            X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
            assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
            return [X], [Y], condition_set, _stringize([X], [Y], condition_set)

        # also to support multi-dimensional unconditional X, Y (usually in kernel-based tests)
        Xs = sorted(set(map(int, X))) if isinstance(X, Iterable) else [int(X)]  # sorted for comparison
        Ys = sorted(set(map(int, Y))) if isinstance(Y, Iterable) else [int(Y)]
        Xs, Ys = (Xs, Ys) if (Xs < Ys) else (Ys, Xs)
        assert len(set(Xs).intersection(condition_set)) == 0 and \
               len(set(Ys).intersection(condition_set)) == 0, "X, Y cannot be in condition_set."
        return Xs, Ys, condition_set, _stringize(Xs, Ys, condition_set)

class FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('fisherz', -1)   # -1: no parameters can be specified for fisherz
        self.assert_input_data_is_valid()
        self.correlation_matrix = np.corrcoef(data.T)

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p

class KCI(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        kci_ui_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'null_ss', 'approx', 'est_width', 'polyd', 'kwidthx', 'kwidthy']}
        kci_ci_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'kernelZ', 'null_ss', 'approx', 'use_gp', 'est_width', 'polyd',
                          'kwidthx', 'kwidthy', 'kwidthz']}
        self.check_cache_method_consistent(
            'kci', hashlib.md5(json.dumps(kci_ci_kwargs, sort_keys=True).encode('utf-8')).hexdigest())
        self.assert_input_data_is_valid()
        self.kci_ui = KCI_UInd(**kci_ui_kwargs)
        self.kci_ci = KCI_CInd(**kci_ci_kwargs)

    def __call__(self, X, Y, condition_set=None):
        # Kernel-based conditional independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        p = self.kci_ui.compute_pvalue(self.data[:, Xs], self.data[:, Ys])[0] if len(condition_set) == 0 else \
            self.kci_ci.compute_pvalue(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set])[0]
        self.pvalue_cache[cache_key] = p
        return p

class Chisq_or_Gsq(CIT_Base):
    def __init__(self, data, method_name, **kwargs):
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]
        assert method_name in ['chisq', 'gsq']
        super().__init__(np.apply_along_axis(_unique, 0, data).astype(np.int64), **kwargs)
        self.check_cache_method_consistent(method_name, -1)   # -1: no parameters can be specified for chisq/gsq
        self.assert_input_data_is_valid()
        self.cardinalities = np.max(self.data, axis=0) + 1

    def chisq_or_gsq_test(self, dataSXY, cardSXY, G_sq=False):
        """by Haoyue@12/18/2021
        Parameters
        ----------
        dataSXY: numpy.ndarray, in shape (|S|+2, n), where |S| is size of conditioning set (can be 0), n is sample size
                 dataSXY.dtype = np.int64, and each row has values [0, 1, 2, ..., card_of_this_row-1]
        cardSXY: cardinalities of each row (each variable)
        G_sq: True if use G-sq, otherwise (False by default), use Chi_sq
        """
        def _Fill2DCountTable(dataXY, cardXY):
            """
            e.g. dataXY: the observed dataset contains 5 samples, on variable x and y they're
                x: 0 1 2 3 0
                y: 1 0 1 2 1
            cardXY: [4, 3]
            fill in the counts by index, we have the joint count table in 4 * 3:
                xy| 0 1 2
                --|-------
                0 | 0 2 0
                1 | 1 0 0
                2 | 0 1 0
                3 | 0 0 1
            note: if sample size is large enough, in theory:
                    min(dataXY[i]) == 0 && max(dataXY[i]) == cardXY[i] - 1
                however some values may be missed.
                also in joint count, not every value in [0, cardX * cardY - 1] occurs.
                that's why we pass cardinalities in, and use `minlength=...` in bincount
            """
            cardX, cardY = cardXY
            xyIndexed = dataXY[0] * cardY + dataXY[1]
            xyJointCounts = np.bincount(xyIndexed, minlength=cardX * cardY).reshape(cardXY)
            xMarginalCounts = np.sum(xyJointCounts, axis=1)
            yMarginalCounts = np.sum(xyJointCounts, axis=0)
            return xyJointCounts, xMarginalCounts, yMarginalCounts

        def _Fill3DCountTableByBincount(dataSXY, cardSXY):
            cardX, cardY = cardSXY[-2:]
            cardS = np.prod(cardSXY[:-2])
            cardCumProd = np.ones_like(cardSXY)
            cardCumProd[:-1] = np.cumprod(cardSXY[1:][::-1])[::-1]
            SxyIndexed = np.dot(cardCumProd[None], dataSXY)[0]

            SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS * cardX * cardY).reshape((cardS, cardX, cardY))
            SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
            SMarginalCountsNonZero = SMarginalCounts != 0
            SMarginalCounts = SMarginalCounts[SMarginalCountsNonZero]
            SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTableByUnique(dataSXY, cardSXY):
            # Sometimes when the conditioning set contains many variables and each variable's cardinality is large
            # e.g. consider an extreme case where
            # S contains 7 variables and each's cardinality=20, then cardS = np.prod(cardSXY[:-2]) would be 1280000000
            # i.e., there are 1280000000 different possible combinations of S,
            #    so the SxyJointCounts array would be of size 1280000000 * cardX * cardY * np.int64,
            #    i.e., ~3.73TB memory! (suppose cardX, cardX are also 20)
            # However, samplesize is usually in 1k-100k scale, far less than cardS,
            # i.e., not all (and actually only a very small portion of combinations of S appeared in data)
            #    i.e., SMarginalCountsNonZero in _Fill3DCountTable_by_bincount is a very sparse array
            # So when cardSXY is large, we first re-index S (skip the absent combinations) and then count XY table for each.
            # See https://github.com/cmu-phil/causal-learn/pull/37.
            cardX, cardY = cardSXY[-2:]
            cardSs = cardSXY[:-2]

            cardSsCumProd = np.ones_like(cardSs)
            cardSsCumProd[:-1] = np.cumprod(cardSs[1:][::-1])[::-1]
            SIndexed = np.dot(cardSsCumProd[None], dataSXY[:-2])[0]

            uniqSIndices, inverseSIndices, SMarginalCounts = np.unique(SIndexed, return_counts=True,
                                                                       return_inverse=True)
            cardS_reduced = len(uniqSIndices)
            SxyIndexed = inverseSIndices * cardX * cardY + dataSXY[-2] * cardY + dataSXY[-1]
            SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS_reduced * cardX * cardY).reshape(
                (cardS_reduced, cardX, cardY))

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTable(dataSXY, cardSXY):
            # about the threshold 1e5, see a rough performance example at:
            # https://gist.github.com/MarkDana/e7d9663a26091585eb6882170108485e#file-count-unique-in-array-performance-md
            if np.prod(cardSXY) < CONST_BINCOUNT_UNIQUE_THRESHOLD: return _Fill3DCountTableByBincount(dataSXY, cardSXY)
            return _Fill3DCountTableByUnique(dataSXY, cardSXY)

        def _CalculatePValue(cTables, eTables):
            """
            calculate the rareness (pValue) of an observation from a given distribution with certain sample size.

            Let k, m, n be respectively the cardinality of S, x, y. if S=empty, k==1.
            Parameters
            ----------
            cTables: tensor, (k, m, n) the [c]ounted tables (reflect joint P_XY)
            eTables: tensor, (k, m, n) the [e]xpected tables (reflect product of marginal P_X*P_Y)
              if there are zero entires in eTables, zero must occur in whole rows or columns.
              e.g. w.l.o.g., row eTables[w, i, :] == 0, iff np.sum(cTables[w], axis=1)[i] == 0, i.e. cTables[w, i, :] == 0,
                   i.e. in configuration of conditioning set == w, no X can be in value i.

            Returns: pValue (float in range (0, 1)), the larger pValue is (>alpha), the more independent.
            -------
            """
            eTables_zero_inds = eTables == 0
            eTables_zero_to_one = np.copy(eTables)
            eTables_zero_to_one[eTables_zero_inds] = 1  # for legal division

            if G_sq == False:
                sum_of_chi_square = np.sum(((cTables - eTables) ** 2) / eTables_zero_to_one)
            else:
                div = np.divide(cTables, eTables_zero_to_one)
                div[div == 0] = 1  # It guarantees that taking natural log in the next step won't cause any error
                sum_of_chi_square = 2 * np.sum(cTables * np.log(div))

            # array in shape (k,), zero_counts_rows[w]=c (0<=c<m) means layer w has c all-zero rows
            zero_counts_rows = eTables_zero_inds.all(axis=2).sum(axis=1)
            zero_counts_cols = eTables_zero_inds.all(axis=1).sum(axis=1)
            sum_of_df = np.sum((cTables.shape[1] - 1 - zero_counts_rows) * (cTables.shape[2] - 1 - zero_counts_cols))
            return 1 if sum_of_df == 0 else chi2.sf(sum_of_chi_square, sum_of_df)

        if len(cardSXY) == 2:  # S is empty
            xyJointCounts, xMarginalCounts, yMarginalCounts = _Fill2DCountTable(dataSXY, cardSXY)
            xyExpectedCounts = np.outer(xMarginalCounts, yMarginalCounts) / dataSXY.shape[1]  # divide by sample size
            return _CalculatePValue(xyJointCounts[None], xyExpectedCounts[None])

        # else, S is not empty: conditioning
        SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts = _Fill3DCountTable(dataSXY, cardSXY)
        SxyExpectedCounts = SxJointCounts[:, :, None] * SyJointCounts[:, None, :] / SMarginalCounts[:, None, None]
        return _CalculatePValue(SxyJointCounts, SxyExpectedCounts)

    def __call__(self, X, Y, condition_set=None):
        # Chi-square (or G-square) independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        indexs = condition_set + Xs + Ys
        p = self.chisq_or_gsq_test(self.data[:, indexs].T, self.cardinalities[indexs], G_sq=self.method == 'gsq')
        self.pvalue_cache[cache_key] = p
        return p

class MV_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('mv_fisherz', -1)   # -1: no parameters can be specified for mv_fisherz
        self.assert_input_data_is_valid(allow_nan=True)

    def _get_index_no_mv_rows(self, mvdata):
        nrow, ncol = np.shape(mvdata)
        bindxRows = np.ones((nrow,), dtype=bool)
        indxRows = np.array(list(range(nrow)))
        for i in range(ncol):
            bindxRows = np.logical_and(bindxRows, ~np.isnan(mvdata[:, i]))
        indxRows = indxRows[bindxRows]
        return indxRows

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test for data with missing values.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(self.data[:, var])
        assert len(test_wise_deletion_XYcond_rows_index) != 0, \
            "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data."
        test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][:, var]
        sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(len(test_wise_deletion_XYcond_rows_index) - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p

class MC_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        # no cache for MC_FisherZ, since skel and prt_m is provided for each test.
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('mc_fisherz', -1)   # -1: no parameters can be specified for mc_fisherz
        self.assert_input_data_is_valid(allow_nan=True)
        self.mv_fisherz = MV_FisherZ(data, **kwargs)

    def __call__(self, X, Y, condition_set, skel, prt_m):
        """Perform an independent test using Fisher-Z's test with test-wise deletion and missingness correction
        If it is not the case which requires a correction, then call function mvfisherZ(...)
        :param prt_m: dictionary, with elements:
            - m: missingness indicators which are not MCAR
            - prt: parents of the missingness indicators
        """

        ## Check whether whether there is at least one common child of X and Y
        if not Helper.cond_perm_c(X, Y, condition_set, prt_m, skel):
            return self.mv_fisherz(X, Y, condition_set)

        ## *********** Step 1 ***********
        # Learning generaive model for {X, Y, S} to impute X, Y, and S

        ## Get parents the {xyS} missingness indicators with parents: prt_m
        # W is the variable which can be used for missingness correction
        W_indx_ = Helper.get_prt_mvars(var=list((X, Y) + condition_set), prt_m=prt_m)

        if len(W_indx_) == 0:  # When there is no variable can be used for correction
            return self.mv_fisherz(X, Y, condition_set)

        ## Get the parents of W missingness indicators
        W_indx = Helper.get_prt_mw(W_indx_, prt_m)

        ## Prepare the W for regression
        # Since the XYS will be regressed on W,
        # W will not contain any of XYS
        var = list((X, Y) + condition_set)
        W_indx = list(set(W_indx) - set(var))

        if len(W_indx) == 0:  # When there is no variable can be used for correction
            return self.mv_fisherz(X, Y, condition_set)

        ## Learn regression models with test-wise deleted data
        involve_vars = var + W_indx
        tdel_data = Helper.test_wise_deletion(self.data[:, involve_vars])
        effective_sz = len(tdel_data[:, 0])
        regMs, rss = Helper.learn_regression_model(tdel_data, num_model=len(var))

        ## *********** Step 2 ***********
        # Get the data of the predictors, Ws
        # The sample size of Ws is the same as the effective sample size
        Ws = Helper.get_predictor_ws(self.data[:, involve_vars], num_test_var=len(var), effective_sz=effective_sz)

        ## *********** Step 3 ***********
        # Generate the virtual data follows the full data distribution P(X, Y, S)
        # The sample size of data_vir is the same as the effective sample size
        data_vir = Helper.gen_vir_data(regMs, rss, Ws, len(var), effective_sz)

        if len(var) > 2:
            cond_set_bgn_0 = np.arange(2, len(var))
        else:
            cond_set_bgn_0 = []

        virtual_cit = MV_FisherZ(data_vir)
        return virtual_cit(0, 1, tuple(cond_set_bgn_0))




#
#
# ######## below we save the original test (which is slower but easier-to-read) ###########
# ######## logic of new test is exactly the same as old, so returns exactly same result ###
# def chisq_notoptimized(data, X, Y, conditioning_set):
#     return chisq_or_gsq_test_notoptimized(data=data, X=X, Y=Y, conditioning_set=conditioning_set)
#
#
# def gsq_notoptimized(data, X, Y, conditioning_set):
#     return chisq_or_gsq_test_notoptimized(data=data, X=X, Y=Y, conditioning_set=conditioning_set, G_sq=True)
#
#
# def chisq_or_gsq_test_notoptimized(data, X, Y, conditioning_set, G_sq=False):
#     """
#     Perform an independence test using chi-square test or G-square test
#
#     Parameters
#     ----------
#     data : data matrices
#     X, Y and condition_set : column indices of data
#     G_sq : True means using G-square test;
#            False means using chi-square test
#
#     Returns
#     -------
#     p : the p-value of the test
#     """
#
#     # Step 1: Subset the data
#     categories_list = [np.unique(data[:, i]) for i in
#                        list(conditioning_set)]  # Obtain the categories of each variable in conditioning_set
#     value_config_list = cartesian_product(
#         categories_list)  # Obtain all the possible value configurations of the conditioning_set (e.g., [[]] if categories_list == [])
#
#     max_categories = int(
#         np.max(data)) + 1  # Used to fix the size of the contingency table (before applying Fienberg's method)
#
#     sum_of_chi_square = 0  # initialize a zero chi_square statistic
#     sum_of_df = 0  # initialize a zero degree of freedom
#
#     def recursive_and(L):
#         "A helper function for subsetting the data using the conditions in L of the form [(variable, value),...]"
#         if len(L) == 0:
#             return data
#         else:
#             condition = data[:, L[0][0]] == L[0][1]
#             i = 1
#             while i < len(L):
#                 new_conjunct = data[:, L[i][0]] == L[i][1]
#                 condition = new_conjunct & condition
#                 i += 1
#             return data[condition]
#
#     for value_config in range(len(value_config_list)):
#         L = list(zip(conditioning_set, value_config_list[value_config]))
#         sub_data = recursive_and(L)[:, [X,
#                                         Y]]  # obtain the subset dataset (containing only the X, Y columns) with only rows specifed in value_config
#
#         ############# Haoyue@12/18/2021  DEBUG: this line is a must:  #####################
#         ########### not all value_config in cartesian product occurs in data ##############
#         # e.g. S=(S0,S1), where S0 has categories {0,1}, S1 has {2,3}. But in combination,#
#         ##### (S0,S1) only shows up with value pair (0,2), (0,3), (1,2) -> no (1,3). ######
#         ########### otherwise #degree_of_freedom will add a spurious 1: (0-1)*(0-1) #######
#         if len(sub_data) == 0: continue  #################################################
#
#         ###################################################################################
#
#         # Step 2: Generate contingency table (applying Fienberg's method)
#         def make_ctable(D, cat_size):
#             x = np.array(D[:, 0], dtype=np.dtype(int))
#             y = np.array(D[:, 1], dtype=np.dtype(int))
#             bin_count = np.bincount(cat_size * x + y)  # Perform linear transformation to obtain frequencies
#             diff = (cat_size ** 2) - len(bin_count)
#             if diff > 0:  # The number of cells generated by bin_count can possibly be less than cat_size**2
#                 bin_count = np.concatenate(
#                     (bin_count, np.zeros(diff)))  # In that case, we concatenate some zeros to fit cat_size**2
#             ctable = bin_count.reshape(cat_size, cat_size)
#             ctable = ctable[~np.all(ctable == 0, axis=1)]  # Remove rows consisted entirely of zeros
#             ctable = ctable[:, ~np.all(ctable == 0, axis=0)]  # Remove columns consisted entirely of zeros
#
#             return ctable
#
#         ctable = make_ctable(sub_data, max_categories)
#
#         # Step 3: Calculate chi-square statistic and degree of freedom from the contingency table
#         row_sum = np.sum(ctable, axis=1)
#         col_sum = np.sum(ctable, axis=0)
#         expected = np.outer(row_sum, col_sum) / sub_data.shape[0]
#         if G_sq == False:
#             chi_sq_stat = np.sum(((ctable - expected) ** 2) / expected)
#         else:
#             div = np.divide(ctable, expected)
#             div[div == 0] = 1  # It guarantees that taking natural log in the next step won't cause any error
#             chi_sq_stat = 2 * np.sum(ctable * np.log(div))
#         df = (ctable.shape[0] - 1) * (ctable.shape[1] - 1)
#
#         sum_of_chi_square += chi_sq_stat
#         sum_of_df += df
#
#     # Step 4: Compute p-value from chi-square CDF
#     if sum_of_df == 0:
#         return 1
#     else:
#         return chi2.sf(sum_of_chi_square, sum_of_df)
#
#
# def cartesian_product(lists):
#     "Return the Cartesian product of lists (List of lists)"
#     result = [[]]
#     for pool in lists:
#         result = [x + [y] for x in result for y in pool]
#     return result