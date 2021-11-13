from copy import deepcopy

import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


def pdag2dag(G):
    '''
    Covert a PDAG to its corresponding DAG

    Parameters
    ----------
    G : Partially Direct Acyclic Graph

    Returns
    -------
    Gd : Direct Acyclic Graph
    '''
    nodes = G.get_nodes()
    # first create a DAG that contains all the directed edges in PDAG
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()
    for edge in edges:
        if not ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or (
                edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW)):
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int))  # index whether the ith node has been removed. 1:removed; 0: not
    while 0 in inde:
        for i in range(Gp.num_vars):
            if (inde[i] == 0):
                sign = 0
                if (len(np.intersect1d(np.where(Gp.graph[:, i] == 1)[0],
                                       np.where(inde == 0)[0])) == 0):  # Xi has no out-going edges
                    sign = sign + 1
                    Nx = np.intersect1d(
                        np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]),
                        np.where(inde == 0)[0])  # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where(Gp.graph[i, :] == 1)[0], np.where(Gp.graph[:, i] == 1)[0]),
                                        np.where(inde == 0)[0])  # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if (len(Nx) > 0):
                        if check2(Gp, Nx, Ax):  # according to the original paper
                            sign = sign + 1
                    else:
                        sign = sign + 1
                if (sign == 2):
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]):
                        Gd.add_edge(Edge(nodes[index], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                    inde[i] = 1

    return Gd


def check2(G, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if (len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0):
            s = 0
            break
    return s
