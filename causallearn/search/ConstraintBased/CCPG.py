from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Set, Callable

import networkx as nx
import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge

def prefix_set(nodes: Set[int], ci_test: Callable[[int, int, set[int]], bool], pset: Set[int], verbose: bool = False) -> Set[int]:
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

# TODO: Pull in the combinations() tool, setup connected components steps correctly, add more type info (better alignment with causal-learn)
def ccpg_alg(nodes: Set[int], ci_test: Callable[[int, int, set[int]], bool], verbose = False) ->
    # Step 1: learn prefix subsets
    s = set()
    S = []
    while s != nodes:
        s = prefix_set(nodes, ci_test, s)
        # enforce termination when ci test are not perfect
        if len(S):
            if s == S[-1] and s != nodes:
                S.append(nodes)
                break
        if verbose: print(f"Prefix set: {s}")
        S.append(s)

    # Step 2: determine connected components of the graph
    components = []
    for i in range(len(S)):
        edges = set()
        cond_set = S[i-1] if i > 0 else set()
        for u, v in itertools.combinations(S[i]-cond_set, 2):
            if not ci_test(u, v, cond_set):
                edges.add(frozenset({u, v}))

        ug = nx.Graph()
        ug.add_nodes_from(S[i]-cond_set)
        ug.add_edges_from(edges)
        cc = connected_components(ug)
        if verbose: print(f"Connected components: {list(cc)}")
        components.extend([set(c) for c in connected_components(ug)])

    # Step 3: determine outer component edges
    edges = set()
    for i, j in itertools.combinations(range(len(components)), 2):
        cond_set = set().union(*components[:i-1]) if i > 0 else set()
        if not set_ci(ci_test, components[i], components[j], cond_set):
            edges.add((i,j))

    return components, edges

# TODO: use the below function as the entrypoint for the above functions. This function should put the CausalGraph together, and setup a lambda for the ci_test being passed into above functions.
def ccpg():
    return
