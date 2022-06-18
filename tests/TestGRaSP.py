import unittest

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

    # generate variance terms
    o = np.diag(np.ones(p))

    # generate edge weights (edge parameters uniformly sampled [-1.0, 1.0])
    b = np.zeros([p, p])
    b[np.where(g == 1)] = np.random.uniform(-1, 1, e)

    # calculate covariance
    s = np.dot(np.dot(np.linalg.inv(np.eye(p) - b), o), np.linalg.inv(np.eye(p) - b).T)

    return s


class TestGRaSP(unittest.TestCase):
    def test_grasp(self):
        ps = [30, 60]
        ds = [0.1, 0.15]
        n = 1000
        reps = 5

        for p in ps:
            for d in ds:
                stats = [[], [], [], []]
                for rep in range(1, reps + 1):
                    g0 = random_dag(p, d)
                    print(
                        "\nNodes:",
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

                    G0 = dag2cpdag(G0)

                    G = grasp(X)

                    AdjC = AdjacencyConfusion(G0, G)
                    stats[0].append(AdjC.get_adj_precision())
                    stats[1].append(AdjC.get_adj_recall())

                    ArrC = ArrowConfusion(G0, G)
                    stats[2].append(ArrC.get_arrows_precision())
                    stats[3].append(ArrC.get_arrows_recall())

                    print(
                        [
                            ["AP", "AR", "AHP", "AHR"][i]
                            + ": "
                            + str(round(stats[i][-1], 2))
                            for i in range(4)
                        ]
                    )

                print(
                    "\nOverall Stats: \n",
                    [
                        ["AP", "AR", "AHP", "AHR"][i]
                        + ": "
                        + str(round(np.sum(stats[i]) / reps, 2))
                        for i in range(4)
                    ],
                )

        print("finish")
