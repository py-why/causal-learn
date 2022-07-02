from __future__ import annotations

from copy import deepcopy
from typing import List, Dict, Tuple, Set

from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.Edges import Edges
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge



def possible_parents(node_x: Node, adjx: List[Node], knowledge: BackgroundKnowledge | None = None) -> List[Node]:
    possible_parents: List[Node] = []

    for node_z in adjx:
        if (knowledge is None) or \
                (not knowledge.is_forbidden(node_z, node_x) and not knowledge.is_required(node_x, node_z)):
            possible_parents.append(node_z)

    return possible_parents


def freeDegree(nodes: List[Node], adjacencies) -> int:
    max_degree = 0
    for node_x in nodes:
        opposites = adjacencies[node_x]
        for node_y in opposites:
            adjx = set(opposites)
            adjx.remove(node_y)

            if len(adjx) > max_degree:
                max_degree = len(adjx)
    return max_degree


def forbiddenEdge(node_x: Node, node_y: Node, knowledge: BackgroundKnowledge | None) -> bool:
    if knowledge is None:
        return False
    elif knowledge.is_forbidden(node_x, node_y) and knowledge.is_forbidden(node_y, node_x):
        print(node_x.get_name() + " --- " + node_y.get_name() +
              " because it was forbidden by background background_knowledge.")
        return True
    return False


def searchAtDepth0(data: ndarray, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                   sep_sets: Dict[Tuple[int, int], Set[int]],
                   independence_test_method: CIT | None=None, alpha: float = 0.05,
                   verbose: bool = False, knowledge: BackgroundKnowledge | None = None, pbar=None) -> bool:
    empty = []

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()
    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth=0, working on node {i}')
        if verbose and (i + 1) % 100 == 0:
            print(nodes[i + 1].get_name())

        for j in range(i + 1, len(nodes)):
            p_value = independence_test_method(i, j, tuple(empty))
            independent = p_value > alpha
            no_edge_required = True if knowledge is None else \
                ((not knowledge.is_required(nodes[i], nodes[j])) or knowledge.is_required(nodes[j], nodes[i]))
            if independent and no_edge_required:
                sep_sets[(i, j)] = set()

                if verbose:
                    print(nodes[i].get_name() + " _||_ " + nodes[j].get_name() + " | (),  score = " + str(p_value))
            elif not forbiddenEdge(nodes[i], nodes[j], knowledge):
                adjacencies[nodes[i]].add(nodes[j])
                adjacencies[nodes[j]].add(nodes[i])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > 0


def searchAtDepth(data: ndarray, depth: int, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                  sep_sets: Dict[Tuple[int, int], Set[int]],
                  independence_test_method: CIT | None = None,
                  alpha: float = 0.05,
                  verbose: bool = False, knowledge: BackgroundKnowledge | None = None, pbar=None) -> bool:
    def edge(adjx: List[Node], i: int, adjacencies_completed_edge: Dict[Node, Set[Node]]) -> bool:
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)
            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()
                flag = False
                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    p_value = independence_test_method(i, Y, tuple(cond_set))
                    independent = p_value > alpha

                    no_edge_required = True if knowledge is None else (
                            not knowledge.is_required(nodes[i], adjx[j]) or knowledge.is_required(adjx[j],
                                                                                                  nodes[i]))
                    if independent and no_edge_required:

                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        if verbose:
                            message = "Independence accepted: " + nodes[i].get_name() + " _||_ " + adjx[
                                j].get_name() + " | "
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "\tp = " + str(p_value)
                            print(message)
                        flag = True
                if flag:
                    return False
        return True

    count = 0

    adjacencies_completed = deepcopy(adjacencies)

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth={depth}, working on node {i}')
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies_completed)
            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def searchAtDepth_not_stable(data: ndarray, depth: int, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                             sep_sets: Dict[Tuple[int, int], Set[int]],
                             independence_test_method: CIT | None=None, alpha: float = 0.05, verbose: bool = False,
                             knowledge: BackgroundKnowledge | None = None,
                             pbar=None) -> bool:
    def edge(adjx, i, adjacencies_completed_edge):
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)
            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()

                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    p_value = independence_test_method(i, Y, tuple(cond_set))
                    independent = p_value > alpha

                    no_edge_required = True if knowledge is None else \
                        (not knowledge.is_required(nodes[i], adjx[j]) or knowledge.is_required(adjx[j], nodes[i]))
                    if independent and no_edge_required:

                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        if verbose:
                            message = "Independence accepted: " + nodes[i].get_name() + " _||_ " + adjx[
                                j].get_name() + " | "
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "\tp = " + str(p_value)
                            print(message)
                        return False
        return True

    count = 0

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth={depth}, working on node {i}')
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies)

            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def fas(data: ndarray, nodes: List[Node], independence_test_method: CIT | None=None, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = True, show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]]]:
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
    sep_sets: separated sets of graph
    """

    # --------check parameter -----------
    if (depth is not None) and type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (knowledge is not None) and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")

    # --------end check parameter -----------

    # ------- initial variable -----------
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    adjacencies: Dict[Node, Set[Node]] = {node: set() for node in nodes}
    if depth is None or depth < 0:
        depth = 1000

    # ------- end initial variable ---------
    print('Starting Fast Adjacency Search.')

    # use tqdm to show progress bar
    pbar = tqdm(total=len(nodes)) if show_progress else None
    for d in range(depth):
        if d == 0:
            more = searchAtDepth0(data, nodes, adjacencies, sep_sets, independence_test_method, alpha, verbose,
                                  knowledge, pbar=pbar)
        else:
            if stable:
                more = searchAtDepth(data, d, nodes, adjacencies, sep_sets, independence_test_method, alpha, verbose,
                                     knowledge, pbar=pbar)
            else:
                more = searchAtDepth_not_stable(data, d, nodes, adjacencies, sep_sets, independence_test_method, alpha,
                                                verbose, knowledge, pbar=pbar)
        if not more:
            break
    if show_progress:
        pbar.close()

    graph = GeneralGraph(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_x = nodes[i]
            node_y = nodes[j]
            if adjacencies[node_x].__contains__(node_y):
                graph.add_edge(Edges().undirected_edge(node_x, node_y))

    print("Finishing Fast Adjacency Search.")
    return graph, sep_sets
