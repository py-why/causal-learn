from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Set, Callable

import networkx as nx
from networkx.algorithms.components.connected import connected_components
import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import *


def prefix_set(nodes: Set[int], ci_test: Callable[[int, int, set[int]], bool], pset: Set[int], verbose: bool = False) -> \
Set[int]:
    # d
    d_set = set()
    for w in nodes - pset:
        w_is_indept = False
        for u in nodes:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                if ci_test(u, v, pset - {u}) and not ci_test(u, v, pset.union({w}) - {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    d_set.add(w)
                    w_is_indept = True
                    break

    # e
    e_set = set()
    for w in nodes - pset - d_set:
        w_is_indept = False
        for u in pset - {w}:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                for v_p in nodes - pset - {w, u, v}:
                    if ci_test(u, v_p, pset.union({v}) - {u}) and not ci_test(u, v_p, pset.union({w, v}) - {u}):
                        if verbose: print(f"Removing {w} from the prefix set")
                        e_set.add(w)
                        w_is_indept = True
                        break

    # f
    f_set = set()
    for w in nodes - pset - d_set - e_set:
        w_is_indept = False
        for u in pset - {w}:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                if not ci_test(u, v, pset - {u}) and not ci_test(v, w, pset) and ci_test(u, w, pset.union({v}) - {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    f_set.add(w)
                    w_is_indept = True
                    break

    return nodes - d_set - e_set - f_set


def set_ci(ci_test: Callable[[int, int, set[int]], bool], set1: Set[int], set2: Set[int], cond_set: Set[int]):
    for u in set1:
        for v in set2:
            if not ci_test(u, v, cond_set):
                return False
    return True


def ccpg_alg(nodes: Set[int], ci_test: Callable[[int, int, set[int]], bool], verbose=False):
    # Step 1: learn prefix subsets
    p_set: Set[int] = set()
    S: List[Set[int]] = []
    while p_set != nodes:
        p_set = prefix_set(nodes, ci_test, p_set)
        # enforce termination when ci test are not perfect
        if len(S):
            if p_set == S[-1] and p_set != nodes:
                S.append(nodes)
                break
        if verbose: print(f"Prefix set: {p_set}")
        S.append(p_set)

    # Step 2: determine connected components of the graph
    components: List[Set[int]] = []
    for i, s_i in enumerate(S):
        cond_set = S[i - 1] if i > 0 else set()
        edges = set()
        for u, v in combinations(s_i - cond_set, 2):
            if not ci_test(u, v, cond_set):
                edges.add(frozenset({u, v}))

        ug = nx.Graph()
        ug.add_nodes_from(s_i - cond_set)
        ug.add_edges_from(edges)
        cc = connected_components(ug)
        if verbose: print(f"Connected components: {list(cc)}")
        components.extend([set(c) for c in cc])

    # Step 3: determine outer component edges
    edges = set()
    # edges: Set[{int, int}] = set()
    for i, j in combinations(range(len(components)), 2):
        cond_set = set().union(*components[:i - 1]) if i > 0 else set()
        if not set_ci(ci_test, components[i], components[j], cond_set):
            edges.add((i, j))

    return components, edges


def ccpg(
        data: ndarray,
        alpha: float = 0.05,
        ci_test_name: str = "fisherz",
        verbose: bool = False,
        **kwargs
) -> CausalGraph:
    # Setup ci_test:
    # ci = CIT(data, ci_test_name, **kwargs)
    ci = MemoizedCIT(data, ci_test_name, **kwargs)

    # def ci_test(i: int, j: int, cond: Set[int]) -> bool:
    #     return ci(i, j, list(cond)) > alpha

    # Discover CCPG nodes and edges
    n, d = data.shape
    components, edges = ccpg_alg(set(range(d)), ci.is_ci, verbose)

    # build graph from edges
    k = len(edges)
    # make names like "{x,y}"
    names = ["{" + ",".join(map(str, comp)) + "}" for comp in components]
    cg = CausalGraph(k, node_names=names)
    cg.G.remove_edges(cg.G.get_graph_edges())
    # add edges between components
    for (i, j) in edges:
        cg.G.add_directed_edge(cg.G.nodes[i], cg.G.nodes[j])

    return cg


# A memoized version of CIT to match the memoizing nature of the Author's CCPG CI_Tester
class MemoizedCIT:
    def __init__(self, data, ci_test_name, alpha: float = 0.05, **kwargs):
        self.cit = CIT(data, ci_test_name, **kwargs)
        self.cache: dict[tuple[int, int, tuple[int,...]], float] = {}
        self.alpha = alpha

    def pvalue(self, i: int, j: int, cond_set: set[int]) -> float:
        cache_key = (i, j, tuple(sorted(cond_set)))
        if cache_key not in self.cache:
            self.cache[cache_key] = self.cit(i, j, list(cond_set))
        return self.cache[cache_key]

    def is_ci(self, i: int, j: int, cond_set: set[int]) -> bool:
        return self.pvalue(i, j, cond_set) > self.alpha
