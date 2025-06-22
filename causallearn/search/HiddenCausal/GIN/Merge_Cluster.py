import numpy as np
import pandas as pd
from itertools import combinations
import itertools
import networkx as nx

'''Merge overlap causal cluster'''
def merge_list(L2):
    l = L2.copy()

    edges = []
    s = list(map(set, l))
    for i, j in combinations(range(len(s)), r=2):

        if s[i].intersection(s[j]):
            edges.append((i, j))

    G = graph(edges)

    result = []
    unassigned = list(range(len(s)))

    for component in connected_components(G):
        union = set().union(*(s[i] for i in component))
        result.append(sorted(union))
        unassigned = [i for i in unassigned if i not in component]

    result.extend(map(sorted, (s[i] for i in unassigned)))

    return result


def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

def connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            c = set(bfs(G, v))
            yield c
            seen.update(c)

def graph(edge_list):
    result = {}
    for source, target in edge_list:
        result.setdefault(source, set()).add(target)
        result.setdefault(target, set()).add(source)
    return result







