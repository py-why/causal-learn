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

