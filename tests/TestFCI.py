import hashlib
import os
import random
import sys
import time
import unittest

from networkx import DiGraph, erdos_renyi_graph, is_directed_acyclic_graph
import numpy as np
import pandas as pd

from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import fci, ruleR5
from causallearn.utils.cit import chisq, fisherz, kci, d_separation
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

######################################### Test Notes ###########################################
# All the benchmark results of loaded files (e.g. "./TestData/benchmark_returned_results/")    #
# are obtained from the code of causal-learn as of commit                                      #
# https://github.com/py-why/causal-learn/commit/5918419 (02-03-2022).                          #
#                                                                                              #
# We are not sure if the results are completely "correct" (reflect ground truth graph) or not. #
# So if you find your tests failed, it means that your modified code is logically inconsistent #
# with the code as of 5918419, but not necessarily means that your code is "wrong".            #
# If you are sure that your modification is "correct" (e.g. fixed some bugs in 5918419),       #
# please report it to us. We will then modify these benchmark results accordingly. Thanks :)   #
######################################### Test Notes ###########################################

BENCHMARK_TXTFILE_TO_MD5 = {
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_asia_fci_chisq_0.05.txt': '4b014aa90aee456bc02dddd008d8be1f',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_cancer_fci_chisq_0.05.txt': '0312381641cb3b4818e0c8539f74e802',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_earthquake_fci_chisq_0.05.txt': 'a1160b92ce15a700858552f08e43b7de',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_sachs_fci_chisq_0.05.txt': 'dced4a202fc32eceb75f53159fc81f3b',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_survey_fci_chisq_0.05.txt': 'b1a28eee1e0c6ea8a64ac1624585c3f4',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_alarm_fci_chisq_0.05.txt': '4b06f6ea7fefe66255a3d16cb8a3711f',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_barley_fci_chisq_0.05.txt': 'a926e329b2bf2b397d3676e9d2020483',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_child_fci_chisq_0.05.txt': '5eff3ea6ce2e2daa53d6b649bd9c31fe',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_insurance_fci_chisq_0.05.txt': '55eed065b8dfc6a21610d3d6e18423c6',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_water_fci_chisq_0.05.txt': '48917f914c8f80684ae2abde87417d5c',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_hailfinder_fci_chisq_0.05.txt': 'bcffe793866db6ddfe9cc602081789cd',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_hepar2_fci_chisq_0.05.txt': '3558683a485588a41493b1a2f2c1d370',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_win95pts_fci_chisq_0.05.txt': '52596ace09c45656f61c70d0e66bdd6a',
    'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_andes_fci_chisq_0.05.txt': '4b7e2ecba5eb50c7266f24670d0d0ae9',
    'tests/TestData/benchmark_returned_results/linear_10_fci_fisherz_0.05.txt': 'eaabd1fbb16bc152b45e456c30166c55'
}
#
INCONSISTENT_RESULT_GRAPH_ERRMSG = "Returned graph is inconsistent with the benchmark. Please check your code with the commit 5918419."
INCONSISTENT_RESULT_GRAPH_WITH_PAG_ERRMSG = "Returned graph is inconsistent with the truth PAG."

# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == expected_MD5, \
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/5918419/tests/TestData'


def gen_coef():
    return np.random.uniform(1, 3)


class TestFCI(unittest.TestCase):
    def test_simple_test(self):
        data = np.empty(shape=(0, 4))
        true_dag = DiGraph()
        ground_truth_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        true_dag.add_edges_from(ground_truth_edges)
        G, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag)

        ground_truth_nodes = []
        for i in range(4):
            ground_truth_nodes.append(GraphNode(f'X{i + 1}'))
        ground_truth_dag = Dag(ground_truth_nodes)
        for u, v in ground_truth_edges:
            ground_truth_dag.add_directed_edge(ground_truth_nodes[u], ground_truth_nodes[v])
        pag = dag2pag(ground_truth_dag, [])

        print(f'fci(data, d_separation, 0.05):')
        self.run_simulate_data_test(pag, G)

        nodes = G.get_nodes()
        assert G.is_adjacent_to(nodes[0], nodes[1])

        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[0], nodes[1]).add_forbidden_by_node(nodes[1], nodes[0])
        G_with_background_knowledge, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag,
                                                 background_knowledge=bk)
        assert not G_with_background_knowledge.is_adjacent_to(nodes[0], nodes[1])

    def test_simple_test2(self):
        data = np.empty(shape=(0, 7))
        true_dag = DiGraph()
        ground_truth_edges = [(7, 0), (7, 1), (8, 3), (8, 4), (2, 5), (2, 6), (5, 1), (6, 3), (3, 0), (1, 4)]
        true_dag.add_edges_from(ground_truth_edges)
        G, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag)
        ground_truth_nodes = []
        for i in range(9):
            ground_truth_nodes.append(GraphNode(f'X{i + 1}'))
        ground_truth_dag = Dag(ground_truth_nodes)
        for u, v in ground_truth_edges:
            ground_truth_dag.add_directed_edge(ground_truth_nodes[u], ground_truth_nodes[v])

        pag = dag2pag(ground_truth_dag, ground_truth_nodes[7: 9])

        print(f'fci(data, d_separation, 0.05):')
        self.run_simulate_data_test(pag, G)

    def test_simple_test3(self):

        data = np.empty(shape=(0, 5))
        true_dag = DiGraph()
        ground_truth_edges = [(0, 2), (1, 2), (2, 3), (2, 4)]
        true_dag.add_edges_from(ground_truth_edges)
        G, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag)

        ground_truth_nodes = []
        for i in range(5):
            ground_truth_nodes.append(GraphNode(f'X{i + 1}'))
        ground_truth_dag = Dag(ground_truth_nodes)
        for u, v in ground_truth_edges:
            ground_truth_dag.add_directed_edge(ground_truth_nodes[u], ground_truth_nodes[v])

        pag = dag2pag(ground_truth_dag, [])

        print(f'fci(data, d_separation, 0.05):')
        self.run_simulate_data_test(pag, G)

    def test_fritl(self):
        data = np.empty(shape=(0, 7))
        true_dag = DiGraph()
        ground_truth_edges = [(7, 0), (7, 5), (8, 0), (8, 6), (9, 3), (9, 4), (9, 6),
                              (0, 1), (0, 2), (1, 2), (2, 4), (5, 6)]
        true_dag.add_edges_from(ground_truth_edges)
        G, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag)

        ground_truth_nodes = []
        for i in range(10):
            ground_truth_nodes.append(GraphNode(f'X{i + 1}'))
        ground_truth_dag = Dag(ground_truth_nodes)
        for u, v in ground_truth_edges:
            ground_truth_dag.add_directed_edge(ground_truth_nodes[u], ground_truth_nodes[v])

        pag = dag2pag(ground_truth_dag, ground_truth_nodes[7: 10])

        print(f'fci(data, d_separation, 0.05):')
        self.run_simulate_data_test(pag, G)

    @staticmethod
    def run_simulate_data_test(truth, est):
        graph_utils = GraphUtils()
        adj_precision = graph_utils.adj_precision(truth, est)
        adj_recall = graph_utils.adj_recall(truth, est)
        arrow_precision = graph_utils.arrow_precision(truth, est)
        arrow_recall = graph_utils.adj_precision(truth, est)

        print(f'adj_precision: {adj_precision}')
        print(f'adj_recall: {adj_recall}')
        print(f'arrow_precision: {arrow_precision}')
        print(f'arrow_recall: {arrow_recall}')
        print()
        assert np.isclose([adj_precision, adj_recall, arrow_precision, arrow_recall], [1.0, 1.0, 1.0, 1.0]).all()

    def test_bnlearn_discrete_datasets(self):
        benchmark_names = [
            "asia", "cancer", "earthquake", "sachs", "survey",
            "alarm", "barley", "child", "insurance", "water",
            "hailfinder", "hepar2", "win95pts",
            "andes"
        ]

        bnlearn_path = 'tests/TestData/bnlearn_discrete_10000/data'
        for bname in benchmark_names:
            data = np.loadtxt(os.path.join(bnlearn_path, f'{bname}.txt'), skiprows=1)
            G, edges = fci(data, chisq, 0.05, verbose=False)
            benchmark_returned_graph = np.loadtxt(
                f'tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_{bname}_fci_chisq_0.05.txt')
            assert np.all(G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG

    def test_continuous_dataset(self):
        data = np.loadtxt('tests/data_linear_10.txt', skiprows=1)
        G, edges = fci(data, fisherz, 0.05, verbose=False)
        benchmark_returned_graph = np.loadtxt(
            f'tests/TestData/benchmark_returned_results/linear_10_fci_fisherz_0.05.txt')
        assert np.all(G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG

    def test_er_graph(self):
        random.seed(42)
        np.random.seed(42)
        p = 0.1
        for _ in range(5):
            data = np.empty(shape=(0, 10))
            true_dag = erdos_renyi_graph(15, p, directed=True)  # The last 5 variables are latent variables
            while not is_directed_acyclic_graph(true_dag):
                true_dag = erdos_renyi_graph(15, p, directed=True)
            ground_truth_edges = list(true_dag.edges)
            print(ground_truth_edges)
            G, edges = fci(data, d_separation, 0.05, verbose=False, true_dag=true_dag)

            ground_truth_nodes = []
            for i in range(15):
                ground_truth_nodes.append(GraphNode(f'X{i + 1}'))
            ground_truth_dag = Dag(ground_truth_nodes)
            for u, v in ground_truth_edges:
                ground_truth_dag.add_directed_edge(ground_truth_nodes[u], ground_truth_nodes[v])
            print(ground_truth_dag)
            pag = dag2pag(ground_truth_dag, ground_truth_nodes[10:])
            print('pag:')
            print(pag)
            print('fci graph:')
            print(G)
            print(f'fci(data, d_separation, 0.05):')
            self.run_simulate_data_test(pag, G)

    def test_rule5(self):
        nodes = []
        for i in range(7):
            nodes.append(GraphNode(str(i)))
        g = GeneralGraph(nodes)
        g.add_edge(Edge(nodes[0], nodes[1], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[0], nodes[2], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[0], nodes[5], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[0], nodes[6], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[1], nodes[3], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[2], nodes[4], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[3], nodes[5], Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(nodes[4], nodes[6], Endpoint.CIRCLE, Endpoint.CIRCLE))

        ruleR5(g, changeFlag=True, verbose=True)

        for edge in g.get_graph_edges():
            assert edge.get_endpoint1() == Endpoint.TAIL
            assert edge.get_endpoint2() == Endpoint.TAIL
