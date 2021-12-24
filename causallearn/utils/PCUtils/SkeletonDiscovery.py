from itertools import combinations
from causallearn.utils.Fas import fas
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import chisq, gsq
from tqdm.auto import tqdm

def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    alpha: desired significance level in (0, 1) (float)
    indep_test : the function of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    if indep_test == chisq or indep_test == gsq:
        # if dealing with discrete data, data is numpy.ndarray with n rows m columns,
        # for each column, translate the discrete values to int indexs starting from 0,
        #   e.g. [45, 45, 6, 7, 6, 7] -> [2, 2, 0, 1, 0, 1]
        #        ['apple', 'apple', 'pear', 'peach', 'pear'] -> [0, 0, 2, 1, 2]
        # in old code, its presumed that discrete `data` is already indexed,
        # but here we make sure it's in indexed form, so allow more user input e.g. 'apple' ..
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]
        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress: pbar.reset()
        for x in range(no_of_var):
            if show_progress: pbar.update()
            if show_progress: pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    if background_knowledge is not None and (
                            background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                            and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                        if verbose: print('%d ind %d | %s with background background_knowledge\n' % (x, y, S))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                        break
                    else:
                        p = cg.ci_test(x, y, S)
                        if p > alpha:
                            if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                            if not stable:
                                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                                if edge1 is not None:
                                    cg.G.remove_edge(edge1)
                                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                                if edge2 is not None:
                                    cg.G.remove_edge(edge2)
                            else:
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                            break
                        else:
                            if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
        if show_progress: pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress: pbar.close()

    return cg


def skeleton_discovery_using_fas(data, alpha, indep_test, stable=True, background_knowledge=None, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    alpha: desired significance level in (0, 1) (float)
    indep_test : the function of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    if indep_test == chisq or indep_test == gsq:
        # if dealing with discrete data, data is numpy.ndarray with n rows m columns,
        # for each column, translate the discrete values to int indexs starting from 0,
        #   e.g. [45, 45, 6, 7, 6, 7] -> [2, 2, 0, 1, 0, 1]
        #        ['apple', 'apple', 'pear', 'peach', 'pear'] -> [0, 0, 2, 1, 2]
        # in old code, its presumed that discrete `data` is already indexed,
        # but here we make sure it's in indexed form, so allow more user input e.g. 'apple' ..
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data
        cg.cardinalities = None

    cache_variables_map = {"data_hash_key": cg.data_hash_key,
                          "ci_test_hash_key": cg.ci_test_hash_key,
                          "cardinalities": cg.cardinalities}
    graph, sep_sets = fas(cg.data, cg.G.nodes, independence_test_method=indep_test, alpha=alpha,
                          knowledge=background_knowledge, depth=-1, verbose=verbose, stable=stable,
                          show_progress=show_progress, cache_variables_map=cache_variables_map)

    for (x, y) in sep_sets.keys():
        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge1 is not None:
            cg.G.remove_edge(edge1)
        append_value(cg.sepset, x, y, list(sep_sets[(x, y)]))
        append_value(cg.sepset, y, x, list(sep_sets[(x, y)]))

    return cg