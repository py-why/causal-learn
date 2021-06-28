#######################################################################################################################
import numpy as np
from graph import GeneralGraph
from graph import GraphUtils
from graph import Edge
from graph import GraphNode
from GraphClass import CausalGraph
from Helper import appendValue
from itertools import permutations, combinations
#######################################################################################################################


def skeletonDiscovery(data, alpha, test_name, stable=True):
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_name: name of the independence test being used
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_name in ["Fisher_Z", "Chi_sq", "G_sq"]

    sepset = np.empty((no_of_var, no_of_var), object)

    utils = GraphUtils()
    no_of_var = data.shape[1]
    nodes = []
    for i in range(no_of_var):
        name = "V", str(i)
        node = GraphNode(name)
        nodes.append(node)

    cg = utils.fully_directed_graph(nodes)

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(nodes, 2))

    depth = -1
    while cg.get_max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.get_adjacent_nodes(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x.remove(y)

            if len(Neigh_x) >= depth:
                if test_name == "Fisher_Z":
                    test = fisherZTest(data, alpha)
                elif test_name == "Chi_sq":

                elif test_name == "G_sq":
                for S in combinations(Neigh_x, depth):
                    ind = test.is_independent(x, y, S)
                    if ind:
                        if not stable:  # Unstable: Remove x---y right away
                            cg.remove_edges(x, y)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            appendValue(sepset, x, y, S)
                            appendValue(sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            cg.remove_edges(x, y)

    return cg, sepset
#######################################################################################################################
