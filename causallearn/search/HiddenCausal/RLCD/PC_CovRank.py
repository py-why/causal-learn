from __future__ import annotations
import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm

from causallearn.utils.cit import CIT_Base

CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge


class CovRank(CIT_Base):
    def __init__(self, data, alpha, rescale_rank_test, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('CovRank', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()

        self.data = (data-data.mean())/data.std()
        self.alpha = alpha

        try:
            from .CCARankTester import CCARankTester
        except ImportError as exc:
            raise ImportError("CCARankTester is required for pc_cov_rank but is not included in scm-identify's StructureLearning/RLCD package.") from exc
        alpha_dict = {i:self.alpha for i in range(data.shape[1])}
        self.RankTester = CCARankTester(self.data, alpha_dict=alpha_dict, rescale_rank_test=rescale_rank_test)

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
        
        _, p = self.RankTester.test(Xs+condition_set, Ys+condition_set, r=len(condition_set))
        self.pvalue_cache[cache_key] = p

        return p
    

def pc_true_cov_rank(
    fake_data: ndarray, 
    independence_test_method, 
    alpha=0.05, 
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    mvpc: bool = False, 
    correction_name: str = 'MV_Crtn_Fisher_Z',
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):

    return pc_alg_true_cov_rank(fake_data=fake_data, independence_test_method=independence_test_method, node_names=node_names, alpha=alpha, stable=stable, uc_rule=uc_rule,
                    uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                    show_progress=show_progress, **kwargs)



def pc_alg_true_cov_rank(
    fake_data: ndarray, 
    independence_test_method,
    node_names: List[str] | None,
    alpha: float,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()

    indep_test = independence_test_method

    cg_1 = SkeletonDiscovery.skeleton_discovery(fake_data, alpha, indep_test, stable,
                                                background_knowledge=background_knowledge, verbose=verbose,
                                                show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


def pc_cov_rank(
    data: ndarray, 
    alpha=0.05, 
    rescale_rank_test=1,
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    mvpc: bool = False, 
    correction_name: str = 'MV_Crtn_Fisher_Z',
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")


    return pc_alg_cov_rank(data=data, node_names=node_names, alpha=alpha, rescale_rank_test=rescale_rank_test, stable=stable, uc_rule=uc_rule,
                    uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                    show_progress=show_progress, **kwargs)



def pc_alg_cov_rank(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    rescale_rank_test: float,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    indep_test = CovRank(data, alpha, rescale_rank_test, **kwargs)
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                background_knowledge=background_knowledge, verbose=verbose,
                                                show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


class FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('fisherz', NO_SPECIFIED_PARAMETERS_MSG)
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
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r) # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p