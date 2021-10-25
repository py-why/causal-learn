import sys

sys.path.append("")

import unittest
import numpy as np
import pandas as pd
from pytrad.search.ConstraintBased.FCI import fci, mod_endpoint
from pytrad.utils.cit import fisherz, kci
from itertools import combinations

from pytrad.graph.GeneralGraph import GeneralGraph
from pytrad.graph.GraphNode import GraphNode
from pytrad.graph.Edge import Edge
from pytrad.graph.Endpoint import Endpoint
from pytrad.utils.GraphUtils import GraphUtils


def gen_coef():
    return np.random.uniform(1, 3)


class TestFCI(unittest.TestCase):

    def test_simple_test(self):
        np.random.seed(0)
        sample_size, loc, scale = 200, 0.0, 1.0
        X1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        X2 = X1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X3 = X1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X4 = X2 * gen_coef() + X3 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        data = np.array([X1, X2, X3, X4]).T
        G = fci(data, fisherz, 0.05, verbose=True)
        pgv_g = GraphUtils.to_pgv(G)
        pgv_g.draw('simple_test.png', prog='dot', format='png')

    def test_fritl(self):
        np.random.seed(0)
        sample_size, loc, scale = 1000, 0.0, 1.0
        L1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        L2 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        L3 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        X1 = gen_coef() * L1 + gen_coef() * L2 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X2 = gen_coef() * X1 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X3 = gen_coef() * X2 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X4 = gen_coef() * X1 + gen_coef() * L3 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X5 = gen_coef() * X3 + gen_coef() * L3 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X6 = gen_coef() * L1 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X7 = gen_coef() * L2 + gen_coef() * L3 + gen_coef() * X6 + np.random.normal(loc=loc, scale=scale,
                                                                                    size=sample_size)
        data = np.array([X1, X2, X3, X4, X5, X6, X7]).T
        G = fci(data, fisherz, 0.05, verbose=True)
        pgv_g = GraphUtils.to_pgv(G)
        pgv_g.draw('fritl.png', prog='dot', format='png')

    def test_causation_p185(self):
        np.random.seed(0)
        sample_size, loc, scale = 2000, 0.0, 1.0
        T1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        T2 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        C = np.random.normal(loc=loc, scale=scale, size=sample_size)
        F = C * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        H = C * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        B = F * gen_coef() + T1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        D = H * gen_coef() + T2 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        A = D * gen_coef() + T1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        E = B * gen_coef() + T2 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        data = np.array([A, B, C, D, E, F, H]).T
        print(fci(data, fisherz, 0.05, verbose=True))

    def test_from_txt(self):
        for i in range(1, 11):
            df = pd.read_csv(f'fci-test-data/data-{i}.txt', delimiter='\t')
            data = df.to_numpy()
            v_labels = df.columns.to_list()

            resultG = fci(data, fisherz, 0.01, verbose=False)
            resultGnodes = resultG.get_nodes()
            nodes = []
            for v in v_labels:
                nodes.append(GraphNode(v))
            G = GeneralGraph(nodes)

            for x, y in combinations(resultGnodes, 2):
                edge = resultG.get_edge(x, y)
                if edge:
                    x = edge.get_node1()
                    y = edge.get_node2()
                    xend = edge.get_endpoint1()
                    yend = edge.get_endpoint2()
                    edge = Edge(nodes[resultGnodes.index(x)], nodes[resultGnodes.index(y)], Endpoint.CIRCLE,
                                Endpoint.CIRCLE)
                    mod_endpoint(edge, nodes[resultGnodes.index(x)], xend)
                    mod_endpoint(edge, nodes[resultGnodes.index(y)], yend)

                    print(edge)
                    G.add_edge(edge)

            result = str(G)
            print(result)
            with open(f"fci-test-data/result-py-FCI-{i}.txt", "w") as result_file:
                result_file.write(result)
