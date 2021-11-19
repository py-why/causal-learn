from itertools import permutations, combinations

import numpy as np

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.Helper import append_value


def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray)
    alpha: desired significance level in (0, 1) (float)
    indep_test : name of the independence test being used
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)

    Returns
    -------
    cg : a CausalGraph object

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.data = data
    cg.set_ind_test(indep_test)

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for x in range(no_of_var):
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    if background_knowledge is not None and (
                            background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                            and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                        print('%d ind %d | %s with background knowledge\n' % (x, y, S))
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
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
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
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg
