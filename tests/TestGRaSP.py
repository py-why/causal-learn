import unittest

import numpy as np
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.DAG2CPDAG import dag2cpdag

import gc


def simulate_data(p, d, n):
    """
    Randomly generates an Erdos-Renyi direct acyclic graph given an ordering 
    and randomly simulates data with the provided parameters.

    p = |variables|
    d = |edges| / |possible edges|
    n = sample size
    """

    # npe = |possible edges|
    pe = int(p * (p - 1) / 2)

    # ne = |edges|
    ne = int(d * pe)

    # generate edges
    e = np.append(np.zeros(pe - ne), 0.5 * np.random.uniform(-1, 1, ne))
    np.random.shuffle(e)
    B = np.zeros([p, p])
    B.T[np.triu_indices(p, 1)] = e

    # generate variance terms
    O = np.diag(np.ones(p))

    # simulate data
    X = np.random.multivariate_normal(np.zeros(p), O, n)
    for i in range(p):
        J = np.where(B[i])[0]
        for j in J: X[:, i] += B[i, j] * X[:, j]

    pi = [i for i in range(p)]
    np.random.shuffle(pi)

    return (B != 0)[pi][:, pi], X[:, pi]


class TestGRaSP(unittest.TestCase):
    def test_grasp(self):
        ps = [30, 60]
        ds = [0.1, 0.15]
        n = 1000
        reps = 5

        gc.set_threshold(20000, 50, 50)
        # gc.set_debug(gc.DEBUG_STATS)

        for p in ps:
            for d in ds:
                stats = [[], [], [], []]
                for rep in range(1, reps + 1):
                    g0, X = simulate_data(p, d, n)
                    print(
                        "\nNodes:", p,
                        "| Edges:", np.sum(g0),
                        "| Avg Degree:", 2 * round(np.sum(g0) / p, 2),
                        "| Rep:", rep,
                    )

                    node_names = [("X%d" % (i + 1)) for i in range(p)]
                    nodes = []

                    for name in node_names:
                        node = GraphNode(name)
                        nodes.append(node)

                    G0 = GeneralGraph(nodes)
                    for y in range(p):
                        for x in np.where(g0[y] == 1)[0]:
                            G0.add_directed_edge(nodes[x], nodes[y])

                    G0 = dag2cpdag(G0)

                    G = grasp(X, depth=1, parameters={'lambda_value': 4})
                    gc.collect()

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
