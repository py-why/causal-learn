import time
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek, Helper
from itertools import permutations, combinations
from causallearn.graph.GraphClass import CausalGraph
import numpy as np
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci, mc_fisherz
import networkx as nx
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge


def pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc=False, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None):
    if mvpc:
        return mvpc_alg(data=data, alpha=alpha, indep_test=indep_test, correction_name=correction_name, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority)
    else:
        return pc_alg(data=data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge)



def pc_alg(data, alpha, indep_test, stable, uc_rule, uc_priority, background_knowledge=None):
    '''
    Perform Peter-Clark algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray)
    alpha : desired significance level (float) in (0, 1)
    indep_test : name of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
           - "MV_Fisher_Z": Missing-value Fishers'Z conditional independence test
           - "kci": kernel-based conditional independence test
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

    Returns
    -------
    cg : a CausalGraph object

    '''

    start = time.time()
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable, background_knowledge=background_knowledge)

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
    end = time.time()

    cg.PC_elapsed = end - start

    return cg




def mvpc_alg(data, alpha, indep_test, correction_name, stable, uc_rule, uc_priority):
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

    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_prt_mpairs(data, alpha, indep_test, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable)
    cg_pre.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    # print('Finish missingness correction.')

    ## Step 3: Orient the edges
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
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


#######################################################################################################################
## *********** Functions for Step 1 ***********
def get_prt_mpairs(data, alpha, indep_test, stable=True):
    """
    Detect the parents of missingness indicators
    If a missingness indicator has no parent, it will not be included in the result
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    prt_m = {'prt': [], 'm': []}

    ## Get the index of missingness indicators
    m_indx = get_mindx(data)

    ## Get the index of parents of missingness indicators
    # If the missingness indicator has no parent, then it will not be collected in prt_m
    for r in m_indx:
        prt_r = detect_parent(r, data, alpha, indep_test, stable)
        if isempty(prt_r):
            pass
        else:
            prt_m['prt'].append(prt_r)
            prt_m['m'].append(r)
    return prt_m


def isempty(prt_r):
    """Test whether the parent of a missingness indicator is empty"""
    return len(prt_r) == 0


def get_mindx(data):
    """Detect the parents of missingness indicators
    :param data: data set (numpy ndarray)
    :return:
    m_indx: list, the index of missingness indicators
    """

    m_indx = []
    _, ncol = np.shape(data)
    for i in range(ncol):
        if np.isnan(data[:, i]).any():
            m_indx.append(i)
    return m_indx


def detect_parent(r, data_, alpha, indep_test, stable=True):
    """Detect the parents of a missingness indicator
    :param r: the missingness indicator
    :param data_: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    : return:
    prt: parent of the missingness indicator, r
    """
    ## TODO: in the test-wise deletion CI test, if test between a binary and a continuous variable,
    #  there can be the case where the binary variable only take one value after deletion.
    #  It is because the assumption is violated.

    ## *********** Adaptation 0 ***********
    # For avoid changing the original data
    data = data_.copy()
    ## *********** End ***********

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    ## *********** Adaptation 1 ***********
    # data
    ## Replace the variable r with its missingness indicator
    ## If r is not a missingness indicator, return [].
    data[:, r] = np.isnan(data[:, r]).astype(float)  # True is missing; false is not missing
    if sum(data[:, r]) == 0 or sum(data[:, r]) == len(data[:, r]):
        return []
    ## *********** End ***********

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.data = data
    cg.set_ind_test(indep_test)
    cg.corr_mat = np.corrcoef(data, rowvar=False) if indep_test == fisherz else []

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:

            ## *********** Adaptation 2 ***********
            # the skeleton search
            ## Only test which variable is the neighbor of r
            if x != r:
                continue
            ## *********** End ***********

            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    ## *********** Adaptation 3 ***********
    ## extract the parent of r from the graph
    cg.to_nx_skeleton()
    cg_skel_adj = nx.to_numpy_array(cg.nx_skel).astype(int)
    prt = get_parent(r, cg_skel_adj)
    ## *********** End ***********

    return prt


def get_parent(r, cg_skel_adj):
    """Get the neighbors of missingness indicators which are the parents
    :param r: the missingness indicator index
    :param cg_skel_adj: adjacancy matrix of a causal skeleton
    :return:
    prt: list, parents of the missingness indicator r
    """
    num_var = len(cg_skel_adj[0, :])
    indx = np.array([i for i in range(num_var)])
    prt = indx[cg_skel_adj[r, :] == 1]
    return prt


## *********** END ***********
#######################################################################################################################

def skeleton_correction(data, alpha, test_with_correction_name, init_cg, prt_m, stable=True):
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_with_correction_name: name of the independence test being used
           - "MV_Crtn_Fisher_Z": Fisher's Z conditional independence test
           - "MV_Crtn_G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]

    ## *********** Adaption 1 ***********
    no_of_var = data.shape[1]

    ## Initialize the graph with the result of test-wise deletion skeletion search
    cg = init_cg

    cg.data = data
    if test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]:
        cg.set_ind_test(mc_fisherz, True)
    # No need of the correlation matrix if using test-wise deletion test
    cg.corr_mat = np.corrcoef(data, rowvar=False) if test_with_correction_name == "MV_Crtn_Fisher_Z" else []
    cg.prt_m = prt_m
    ## *********** Adaption 1 ***********

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg


#######################################################################################################################

# *********** Evaluation util ***********

def get_adjacancy_matrix(g):
    return nx.to_numpy_array(g.nx_graph).astype(int)


def matrix_diff(cg1, cg2):
    adj1 = get_adjacancy_matrix(cg1)
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count / 2, diff_ls


