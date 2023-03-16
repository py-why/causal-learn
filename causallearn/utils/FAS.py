from __future__ import annotations

from itertools import combinations
from typing import List, Dict, Tuple, Set

from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def fas(data: ndarray, nodes: List[Node], independence_test_method: CIT_Base, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = True, show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    nodes: The search nodes.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    show_progress: whether to use tqdm to show progress bar
    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    sep_sets: Separated sets of graph
    test_results: Results of conditional independence tests
    """
    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not all(isinstance(node, Node) for node in nodes):
        raise TypeError("'nodes' must be 'List[Node]' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if knowledge is not None and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'knowledge' must be 'BackgroundKnowledge' type!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    if depth == -1:
        depth = float('inf')

    no_of_var = data.shape[1]
    node_names = [node.get_name() for node in nodes]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    test_results: Dict[Tuple[int, int, Set[int]], float] = {}

    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)

    var_range = tqdm(range(no_of_var), leave=True) if show_progress \
        else range(no_of_var)
    current_depth: int = -1
    while cg.max_degree() - 1 > current_depth and current_depth < depth:
        current_depth += 1
        edge_removal = set()
        for x in var_range:
            if show_progress:
                var_range.set_description(f'Depth={current_depth}, working on node {x}')
                var_range.update()
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < current_depth - 1:
                continue
            for y in Neigh_x:
                sepsets = set()
                if (knowledge is not None and
                    knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    if not stable:
                        remove_if_exists(x, y)
                        remove_if_exists(y, x)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        sep_sets[(x, y)] = set()
                        sep_sets[(y, x)] = set()
                        break
                    else:
                        edge_removal.add((x, y))  # after all conditioning sets at
                        edge_removal.add((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, current_depth):
                    p = cg.ci_test(x, y, S)
                    test_results[(x, y, S)] = p
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            remove_if_exists(x, y)
                            remove_if_exists(y, x)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            sep_sets[(x, y)] = set(S)
                            sep_sets[(y, x)] = set(S)
                            break
                        else:
                            edge_removal.add((x, y))  # after all conditioning sets at
                            edge_removal.add((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        for (x, y) in edge_removal:
            remove_if_exists(x, y)
            if cg.sepset[x, y] is not None:
                origin_set = set(l_in for l_out in cg.sepset[x, y]
                                 for l_in in l_out)
                sep_sets[(x, y)] = origin_set
                sep_sets[(y, x)] = origin_set

    return cg.G, sep_sets, test_results
