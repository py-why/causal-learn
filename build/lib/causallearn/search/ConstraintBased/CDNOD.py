import time
from itertools import permutations, combinations
from typing import Dict, List, Optional

import networkx as nx
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.utils.cit import *
from causallearn.search.ConstraintBased.PC import get_parent_missingness_pairs, skeleton_correction


def cdnod(data: ndarray, c_indx: ndarray, alpha: float=0.05, indep_test: str=fisherz, stable: bool=True,
          uc_rule: int=0, uc_priority: int=2, mvcdnod: bool=False, correction_name: str='MV_Crtn_Fisher_Z',
          background_knowledge: Optional[BackgroundKnowledge]=None, verbose: bool=False,
          show_progress: bool = True, **kwargs) -> CausalGraph:
    """
    Causal discovery from nonstationary/heterogeneous data
    phase 1: learning causal skeleton,
    phase 2: identifying causal directions with generalization of invariance, V-structure. Meek rule
    phase 3: identifying directions with independent change principle, and (TODO: under development)
    phase 4: recovering the nonstationarity driving force (TODO: under development)

    Parameters
    ----------
     c_indx: time index or domain index that captures the unobserved changing factors

    Returns
    -------
    cg : a CausalGraph object over the augmented dataset that includes c_indx
    """
    # augment the variable set by involving c_indx to capture the distribution shift
    data_aug = np.concatenate((data, c_indx), axis=1)
    if mvcdnod:
        return mvcdnod_alg(data=data_aug, alpha=alpha, indep_test=indep_test, correction_name=correction_name,
                           stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, verbose=verbose,
                           show_progress=show_progress, **kwargs)
    else:
        return cdnod_alg(data=data_aug, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                         uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                         show_progress=show_progress, **kwargs)


def cdnod_alg(data: ndarray, alpha: float, indep_test: str, stable: bool, uc_rule: int, uc_priority: int,
              background_knowledge: Optional[BackgroundKnowledge] = None, verbose: bool = False,
              show_progress: bool = True, **kwargs) -> CausalGraph:
    """
    Perform Peter-Clark algorithm for causal discovery on the augmented data set that captures the unobserved changing factors

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    alpha : desired significance level (float) in (0, 1)
    indep_test : name of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
           - "MV_Fisher_Z": Missing-value Fishers'Z conditional independence test
           - "kci": kernel-based conditional independence test (If C is time index, KCI test is recommended)
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
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable)

    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)

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


def mvcdnod_alg(data: ndarray, alpha: float, indep_test: str, correction_name: str, stable: bool, uc_rule: int,
                uc_priority: int, verbose: bool, show_progress: bool, **kwargs) -> CausalGraph:
    """
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level (float) in (0, 1)
    :param indep_test: name of the test-wise deletion independence test being used
           - "MV_Fisher_Z": Fisher's Z conditional independence test
           - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    : param correction_name: name of the missingness correction
            - "MV_Crtn_Fisher_Z": Permutation based correction method
            - "MV_Crtn_G_sq": G-squared conditional independence test (TODO: under development)
            - "MV_DRW_Fisher_Z": density ratio weighting based correction method (TODO: under development)
            - "MV_DRW_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :param uc_rule: how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    :param uc_priority: rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg: a CausalGraph object
    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable, verbose=verbose,
                                                  show_progress=show_progress)
    cg_pre.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    # print('Finish missingness correction.')

    ## Step 3: Orient the edges
    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data.shape[1] - 1
    for i in cg_corr.G.get_adjacent_nodes(cg_corr.G.nodes[c_indx_id]):
        cg_corr.G.add_directed_edge(i, cg_corr.G.nodes[c_indx_id])

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_corr, uc_priority)
        else:
            cg_2 = UCSepset.uc_sepset(cg_corr)
        cg = Meek.meek(cg_2)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_corr, uc_priority)
        else:
            cg_2 = UCSepset.maxp(cg_corr)
        cg = Meek.meek(cg_2)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, uc_priority)
        else:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha)
        cg_before = Meek.definite_meek(cg_2)
        cg = Meek.meek(cg_before)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg
