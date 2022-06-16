import os
import sys

import numpy as np

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.DAG2CPDAG import dag2cpdag


def random_dag(p, d):
    """
    Randomly generates an Erdos-Renyi direct acyclic graph given an ordering.

    p = |variables|
    d = |edges| / |possible edges|
    """

    # npe = |possible edges|
    pe = int(p * (p - 1) / 2)

    # e = |edges|
    ne = int(d * pe)

    # generate edges
    e = np.append(np.zeros(pe - ne, "uint8"), np.ones(ne, "uint8"))
    np.random.shuffle(e)

    # generate graph
    g = np.zeros([p, p], "uint8")
    g.T[np.triu_indices(p, 1)] = e

    return g


def parameterize_dag(g):
    """
    Randomly parameterize a directed acyclic graph.

    g = directed acyclic graph (adjacency matrix)
    """

    # p = |variables|
    p = g.shape[0]

    # e = |edges|
    e = np.sum(g)

    # generate variance terms (variance parameters uniformly sampled [1.0, 2.0])
    o = np.diag(np.ones(p))
    # o = np.diag(np.random.uniform(0.3,2.0,p))

    # generate edge weights (edge parameters uniformly sampled +/-[0.3, 0.7])
    b = np.zeros([p, p])
    b[np.where(g == 1)] = np.random.uniform(-1, 1, e)
    # b[np.where(g==1)] = np.random.choice([-1,1], e) * np.random.uniform(0.2,1.0, e)

    # calculate covariance
    s = np.dot(np.dot(np.linalg.inv(np.eye(p) - b), o), np.linalg.inv(np.eye(p) - b).T)

    return s


ps = [60]
ds = [0.17]
n = 1000
reps = 5

for p in ps:
    for d in ds:
        stats = [0, 0, 0, 0]
        for rep in range(1, reps + 1):
            g0 = random_dag(p, d)
            print(
                "Nodes:",
                p,
                "| Edges:",
                np.sum(g0),
                "| Avg Degree:",
                2 * np.sum(g0) / p,
                "| Rep:",
                rep,
            )
            cov = parameterize_dag(g0)
            X = np.random.multivariate_normal(np.zeros(p), cov, n)

            node_names = [("x%d" % i) for i in range(p)]
            nodes = []

            for name in node_names:
                node = GraphNode(name)
                nodes.append(node)

            G0 = GeneralGraph(nodes)
            for y in range(p):
                for x in np.where(g0[y] == 1)[0]:
                    G0.add_directed_edge(nodes[x], nodes[y])

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                G0 = dag2cpdag(G0)
                sys.stdout = old_stdout

            G = grasp(X)
            print()

            AdjC = AdjacencyConfusion(G0, G)
            stats[0] += AdjC.get_adj_precision() / reps
            stats[1] += AdjC.get_adj_recall() / reps

            ArrC = ArrowConfusion(G0, G)
            stats[2] += ArrC.get_arrows_precision() / reps
            stats[3] += ArrC.get_arrows_recall() / reps

        print(
            [
                ["AP", "AR", "AHP", "AHR"][i] + ": " + str(round(stats[i], 2))
                for i in range(4)
            ]
        )
