"""
Code modified from:
https://github.com/jmschrei/pomegranate/blob/master/pomegranate/BayesianNetwork.pyx

Several tricks to save memory of parent_graphs and prune edges in order_graph are based on:
https://arxiv.org/abs/1608.02682

Ignavier Ng, Yujia Zheng, Jiji Zhang, Kun Zhang
"""
import itertools as it
import logging
from bisect import bisect_left
from collections import OrderedDict

import networkx as nx
import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.utils.PriorityQueue import PriorityQueue

_logger = logging.getLogger(__name__)
INF = float("inf")
NEGINF = float("-inf")


def bic_exact_search(X, super_graph=None, search_method='astar',
                     use_path_extension=True, use_k_cycle_heuristic=False,
                     k=3, verbose=False, include_graph=None, max_parents=None):
    """
    Search for the optimal graph using DP or A star.
    Parameters
    ----------
    X : numpy.ndarray, shape=(n, d)
        The data to fit the structure too, where each row is a sample and
        each column corresponds to the associated variable.
    super_graph : numpy.ndarray, shape=(d, d)
        Super-structure to restrict search space (binary matrix).
        If None, no super-structure is used. Default is None.
    search_method : str
        Method of of exact search.
        Default is astar.
    use_path_extension : bool
        Whether to use optimal path extension for order graph. Note that
        this trick will not affect the correctness of search procedure.
        Default is True.
    use_k_cycle_heuristic : bool
        Whether to use k-cycle conflict heuristic for astar.
        Default is False.
    k : int
        Parameter used by k-cycle conflict heuristic for astar.
        Default is 3.
    verbose : bool
        Whether to log messages related to search procedure.
    max_parents : int
        The maximum number of parents a node can have. If used, this means
        using the k-learn procedure. Can drastically speed up algorithms.
        If None, no max on parents. Default is None.
    Returns
    -------
    dag_est :  numpy.ndarray, shape=(d, d)
        Estimated DAG.
    search_stats :  dict
        Some statistics related to the seach procedure.
    """
    n, d = X.shape
    if super_graph is None:
        super_graph = np.ones((d, d))
        super_graph[np.diag_indices_from(super_graph)] = 0

    if include_graph is None:
        include_graph = np.zeros((d, d))
    else:
        assert Dag.is_dag(include_graph)

    assert set(super_graph.diagonal()) == {0}  # Diagonals must be zeros
    if max_parents is None:
        max_parents = d

    # To store statistics related to the seach procedure
    search_stats = {}

    # Generate parent graphs (without parallel computing)
    parent_graphs = tuple([
        generate_parent_graph(X, i, max_parents,
                              parent_set=tuple(np.where(super_graph[:, i])[0]),
                              include_parents=tuple(np.where(include_graph[:, i])[0]))
        for i in range(d)])
    search_stats['n_parent_graphs_entries'] = sum([len(l) for l in parent_graphs])
    if verbose:
        _logger.info("Finished generating parent graphs.")

    # Shortest path search
    if search_method == 'dp':
        structures, shortest_path_stats = dp_shortest_path(parent_graphs, use_path_extension, verbose)
    elif search_method == 'astar':
        structures, shortest_path_stats = astar_shortest_path(parent_graphs, use_path_extension,
                                                              use_k_cycle_heuristic, k, verbose)
    else:
        raise ValueError("Unknown search method.")

    search_stats.update(shortest_path_stats)  # Store in search_stats
    if verbose:
        _logger.info("Finished searching for shortest path.")

    # Covnert structures to adjacency matrix
    dag_est = np.zeros((d, d))
    for i, parents in enumerate(structures):
        dag_est[parents, i] = 1

    return dag_est, search_stats


def astar_shortest_path(parent_graphs, use_path_extension=True,
                        use_k_cycle_heuristic=False, k=3, verbose=False):
    """
    Search for the shortest path in the order graph using A star.
    Parameters
    ----------
    parent_graphs : tuple, shape=(d,)
        The parent graph for each variable.
    use_path_extension : bool
        Whether to use optimal path extension for order graph. Note that
        this trick will not affect the correctness of search procedure.
        Default is True.
    use_k_cycle_heuristic : bool
        Whether to use k-cycle conflict heuristic for astar.
        Default is False.
    k : int
        Parameter used by k-cycle conflict heuristic for astar.
        Default is 3.
    verbose : bool
        Whether to log messages related to search procedure.
    Returns
    -------
    structures :  tuple, shape=(d,)
        Optimal parents for each variable.
    shortest_path_stats :  dict
        Some statistics related to the shortest path seach.
    """
    d = len(parent_graphs)
    opened = PriorityQueue()
    closed = set()

    if use_k_cycle_heuristic:
        # Create pattern databse
        PD = create_dynamic_pd(parent_graphs, k)
        if verbose:
            _logger.info('Finished creating pattern database.')

    score = {(): 0}
    h = sum(parent_graphs[i][0][1] for i in range(d))
    opened.push(((), [() for i in range(d)]), h)
    max_n_opened = 1  # For counting the maximum length of open list
    while_iter = 0  # For counting the number of iterations of the while loop
    for_iter = 0  # For counting the number of iterations of the for loop
    while not opened.empty():
        while_iter += 1
        _, (U, structures) = opened.pop()
        U = tuple(sorted(U))  # Ensure consistency of variable order

        if U in closed:
            continue
        else:
            closed.add(U)

        if len(U) == d:
            break

        out_set = tuple(i for i in range(d) if i not in U)
        for i in out_set:
            for_iter += 1
            parents, best_score = query_best_structure(parent_graphs[i], U)

            g = best_score + score[U]
            new_U = tuple(sorted(U + (i,)))
            new_structures = structures[:]
            new_structures[i] = parents

            if use_path_extension:
                new_U, new_structures, g = path_extension(new_U, new_structures, parent_graphs, g)

            if use_k_cycle_heuristic:
                h = compute_dynamic_h(new_U, PD)
            else:
                h = sum(parent_graphs[j][0][1] for j in range(d) if j not in new_U)

            f = g + h
            new_entry = (new_U, new_structures)

            if new_U in closed:
                if g < score[new_U]:
                    closed.remove(new_U)
                    opened.push(new_entry, f)
                    score[new_U] = g
            else:
                if opened.get(new_U) is not None:
                    if g < score[new_U]:
                        opened.delete(new_U)
                        opened.push(new_entry, f)
                        score[new_U] = g
                else:
                    opened.push(new_entry, f)
                    score[new_U] = g

        if len(opened) > max_n_opened:
            max_n_opened = len(opened)

    shortest_path_stats = {
        'while_iter': while_iter,
        'for_iter': for_iter,
        'n_closed': len(closed),
        'max_n_opened': max_n_opened
    }
    if use_k_cycle_heuristic:
        shortest_path_stats['n_pattern_database'] = len(PD)
    return tuple(structures), shortest_path_stats


def dp_shortest_path(parent_graphs, use_path_extension=True, verbose=False):
    """
    Search for the shortest path in the order graph using DP (Bellman-Ford algorithm).
    Parameters
    ----------
    parent_graphs : tuple, shape=(d,)
        The parent graph for each variable.
    use_path_extension : bool
        Whether to use optimal path extension for order graph. Note that
        this trick will not affect the correctness of search procedure.
        Default is True.
    verbose : bool
        Whether to log messages related to search procedure.
    Returns
    -------
    structures :  tuple, shape=(d,)
        Optimal parents for each variable.
    shortest_path_stats :  dict
        Some statistics related to the shortest path seach.
    """
    d = len(parent_graphs)
    order_graph = nx.DiGraph()
    for i in range(d + 1):
        optimal_child = {}
        for subset in it.combinations(range(d), i):
            order_graph.add_node(subset)

            for variable in subset:
                parent = tuple(v for v in subset if (v != variable))
                if use_path_extension:
                    if parent in optimal_child:
                        continue
                    if set(parent).issuperset(parent_graphs[variable][0][0]):
                        optimal_child[parent] = subset

                structure, weight = query_best_structure(parent_graphs[variable], parent)
                order_graph.add_edge(parent, subset, weight=weight,
                                     structure=structure)

        if use_path_extension:
            # Remove some edges indicated by optimal path extension
            for parent, child in optimal_child.items():
                edges_to_remove = [edge for edge in nx.edges(order_graph, parent) if edge[1] != child]
                for edge in edges_to_remove:
                    order_graph.remove_edge(*edge)

    path = nx.shortest_path(order_graph, source=(), target=tuple(range(d)),
                            weight='weight', method='bellman-ford')

    score, structures = 0, list(None for i in range(d))
    for u, v in zip(path[:-1], path[1:]):
        idx = list(set(v) - set(u))[0]
        parents = order_graph.get_edge_data(u, v)['structure']
        structures[idx] = parents
        score -= order_graph.get_edge_data(u, v)['weight']

    shortest_path_stats = {
        'n_order_graph_nodes': order_graph.number_of_nodes(),
        'n_order_graph_edges': order_graph.number_of_edges()
    }
    return structures, shortest_path_stats


def generate_parent_graph(X, i, max_parents=None, parent_set=None, include_parents=None):
    """
    Generate a parent graph for a single variable over its parents.
    This will generate the parent graph for a single parents given the data.
    A parent graph is the dynamically generated best parent set and respective
    score for each combination of parent variables. For example, if we are
    generating a parent graph for x1 over x2, x3, and x4, we may calculate that
    having x2 as a parent is better than x2,x3 and so store the value
    of x2 in the node for x2,x3.
    Parameters
    ----------
    X : numpy.ndarray, shape=(n, d)
        The data to fit the structure too, where each row is a sample and
        each column corresponds to the associated variable.
    i : int
        The column index to build the parent graph for.
    max_parents : int
        The maximum number of parents a node can have. If used, this means
        using the k-learn procedure. Can drastically speed up algorithms.
        If None, no max on parents. Default is None.
    parent_set : tuple, default None
        The variables which are possible parents for this variable. If nothing
        is passed in then it defaults to all other variables, as one would
        expect in the naive case. This allows for cases where we want to build
        a parent graph over only a subset of the variables.
    Returns
    -------
    parent_graph : tuple, shape=(d,)
        The parents for each variable in this SCC
    """
    n, d = X.shape
    if max_parents is None:
        max_parents = d

    if parent_set is None:
        parent_set = tuple(set(range(d)) - set([i]))
    else:
        # Remove possible duplicates elements
        parent_set = tuple(set(parent_set))

    if include_parents is None:
        include_parents = ()  # Emptry tuple

    parent_graph = []
    for j in range(len(parent_set) + 1):
        if j == 0:
            if len(include_parents) > 0:
                continue
            structure = ()
            score = bic_score_node(X, i, structure)
            insort(parent_graph, structure, score)
        elif j <= max_parents:
            for structure in it.combinations(parent_set, j):
                if not set(structure).issuperset(include_parents):
                    # Skip if structure does not contain all parents to be included
                    continue
                score = bic_score_node(X, i, structure)

                for variable in structure:
                    curr_structure = tuple(l for l in structure if l != variable)
                    _, curr_best_score = query_best_structure(parent_graph, curr_structure)

                    if curr_best_score < score:
                        # A subset of the structure has better score
                        # This indicates that the structure is not a maximal candidate parents set
                        # So we do not save it to parent_graph
                        break
                else:
                    # No subset of structure is found to have better score
                    # This indicates that the structure is a maximal candidate parents set
                    # So we save it to parent_graph
                    insort(parent_graph, structure, score)

    return parent_graph


def bic_score_node(X, i, structure):
    structure = list(structure)
    n, d = X.shape
    if len(structure) == 0:
        residual = np.sum(X[:, i] ** 2)
    else:
        _, residual, _, _ = np.linalg.lstsq(a=X[:, structure],
                                            b=X[:, i],
                                            rcond=None)
    bic = n * np.log(residual / n) + len(structure) * np.log(n)
    return bic.item()


def insort(parent_graph, structure, score):
    """
    parent_graph is a list of tuples with the form (structure, score) and is 
    sorted based on score. This function inserts the structure and score
    at the corresponding position such that the list remains sorted.
    Referred from https://stackoverflow.com/a/39501468
    """

    class KeyWrapper:
        def __init__(self, iterable, key):
            self.it = iterable
            self.key = key

        def __getitem__(self, i):
            return self.key(self.it[i])

        def __len__(self):
            return len(self.it)

    index = bisect_left(KeyWrapper(parent_graph, key=lambda c: c[1]), score)
    parent_graph.insert(index, (structure, score))


def query_best_structure(parent_graph, target_structure):
    """
    This function returns the first structure and corresponding score that is
    a subset of target_structure. Since parent_graph is sorted based on score,
    the first structure that is a subset is guaranteed to be the best subset
    """
    target_structure = set(target_structure)
    for curr_structure, curr_score in parent_graph:
        if set(curr_structure).issubset(target_structure):
            return curr_structure, curr_score
    # If include_graph is not used, the for loop is guaranteed to return
    # the curr_structure and curr_score
    # i.e., the target_structure is guaranteed to exist in parent_graph
    # However, if include_graph is used, in some cases target_structure
    # may not exist in parent_graph, so we need to return some default values
    return None, INF


def path_extension(U, structures, parent_graphs, g):
    d = len(structures)
    while True:
        extended = False
        out_set = tuple(i for i in range(d) if i not in U)
        for i in out_set:
            parents, best_score = query_best_structure(parent_graphs[i], U)
            if best_score == parent_graphs[i][0][1]:
                g += parent_graphs[i][0][1]
                U = tuple(sorted(U + (i,)))
                structures = structures[:]
                structures[i] = parents
                extended = True
                break
        if not extended:
            break

    return U, structures, g


def create_dynamic_pd(parent_graphs, k=2):
    d = len(parent_graphs)
    V = tuple(range(d))

    PD_final = OrderedDict()
    PD_prev = {V: 0}  # Working variable
    delta_h = {V: 0}  # Working variable
    save = set()
    # Perform BFS for k levels
    for l in range(1, k + 1):
        PD_curr = {}  # Working variable
        for U in PD_prev:
            # TODO: Handle edge cases if length of U is 1
            expand(U, l, PD_prev, PD_curr, parent_graphs)
            check_save(U, V, delta_h, PD_prev, parent_graphs, save)
            PD_final[tuple_diff(V, U)] = PD_prev[U]

        # Update working variable
        PD_prev = PD_curr

    # Remove superset patterns with no improvement
    for i in tuple_diff(PD_final.keys(), save):
        del PD_final[i]

    # Sort patterns in decreasing costs
    PD_final = OrderedDict(sorted(PD_final.items(),
                                  key=lambda tup: delta_h[tuple_diff(V, tup[0])],
                                  reverse=True))
    return PD_final


def expand(U, l, PD_prev, PD_curr, parent_graphs):
    for i in U:
        out_set = tuple_diff(U, [i])
        g = PD_prev[U] + query_best_structure(parent_graphs[i], out_set)[1]
        if out_set in PD_curr:
            if g < PD_curr[out_set]:
                # Duplicate detection
                PD_curr[out_set] = g
        else:
            PD_curr[out_set] = g


def check_save(U, V, delta_h, PD_prev, parent_graphs, save):
    d = len(parent_graphs)

    # Get g and h to compute delta_h
    g = PD_prev[U]
    h_simple = sum(parent_graphs[j][0][1] for j in range(d) if j not in U)
    delta_h[U] = g - h_simple

    # Check improvement over subset patterns
    for i in tuple_diff(V, U):
        if delta_h[U] > delta_h[tuple_union(U, [i])]:
            save.add(tuple_diff(V, U))


def compute_dynamic_h(U, PD):
    h = 0
    R = U
    for S in PD:
        if set(S).issubset(R):
            R = tuple_diff(R, S)
            h += PD[S]
    return h


def tuple_diff(A, B):
    # A and B are two different tuples/lists
    # Return A - B in a sorted way
    A = set(A)
    B = set(B)
    return tuple(sorted(set(A) - set(B)))


def tuple_union(A, B):
    # A and B are two different tuples/lists
    # Return A + B in a sorted way
    A = list(A)
    B = list(B)
    return tuple(sorted(A + B))
