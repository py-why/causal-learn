import hashlib
import os
import random
import unittest

import numpy as np

from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import CIT, chisq, fisherz, kci, d_separation
from causallearn.utils.FAS import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

BENCHMARK_TXTFILE_TO_MD5 = {
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_asia_fci_chisq_0.05.txt": "65f54932a9d8224459e56c40129e6d8b",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_cancer_fci_chisq_0.05.txt": "0312381641cb3b4818e0c8539f74e802",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_earthquake_fci_chisq_0.05.txt": "a1160b92ce15a700858552f08e43b7de",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_sachs_fci_chisq_0.05.txt": "dced4a202fc32eceb75f53159fc81f3b",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_survey_fci_chisq_0.05.txt": "b1a28eee1e0c6ea8a64ac1624585c3f4",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_alarm_fci_chisq_0.05.txt": "c3bbc2b8aba456a4258dd071a42085bc",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_barley_fci_chisq_0.05.txt": "4a5000e7a582083859ee6aef15073676",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_child_fci_chisq_0.05.txt": "6b7858589e12f04b0f489ba4589a1254",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_insurance_fci_chisq_0.05.txt": "9975942b936aa2b1fc90c09318ca2d08",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_water_fci_chisq_0.05.txt": "48eee804d59526187b7ecd0519556ee5",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_hailfinder_fci_chisq_0.05.txt": "6b9a6b95b6474f8530e85c022f4e749c",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_hepar2_fci_chisq_0.05.txt": "4aae21ff3d9aa2435515ed2ee402294c",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_win95pts_fci_chisq_0.05.txt": "648fdf271e1440c06ca2b31b55ef1f3f",
    "tests/TestData/benchmark_returned_results/bnlearn_discrete_10000_andes_fci_chisq_0.05.txt": "04092ae93e54c727579f08bf5dc34c77",
    "tests/TestData/benchmark_returned_results/linear_10_fci_fisherz_0.05.txt": "289c86f9c665bf82bbcc4c9e1dcec3e7"
}

# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == expected_MD5, \
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/5918419/tests/TestData'


class TestFAS(unittest.TestCase):
    def test_inputs(self):
        data = np.loadtxt('tests/data_linear_10.txt', skiprows=1)
        alpha = 0.05
        cit = CIT(data, fisherz, alpha=alpha)
        nodes = [GraphNode(f"X{i + 1}") for i in range(data.shape[1])]
        bgk = BackgroundKnowledge()
        self.assertRaises(TypeError, fas, data=None, nodes=nodes, independence_test_method=cit, alpha=alpha, knowledge=bgk, verbose=False)
        self.assertRaises(TypeError, fas, data=data, nodes=None, independence_test_method=cit, alpha=alpha, knowledge=bgk, verbose=False)
        self.assertRaises(TypeError, fas, data=data, nodes=nodes, independence_test_method=None, alpha=alpha, knowledge=bgk, verbose=False)
        self.assertRaises(TypeError, fas, data=data, nodes=nodes, independence_test_method=cit, alpha=1, knowledge=bgk, verbose=False)
        self.assertRaises(TypeError, fas, data=data, nodes=nodes, independence_test_method=cit, alpha=0, knowledge=bgk, verbose=False)
        self.assertRaises(TypeError, fas, data=data, nodes=nodes, independence_test_method=cit, alpha=alpha, knowledge=data, verbose=False)

    @staticmethod
    def run_test_with_random_background(data, cit, alpha):
        random.seed(42)

        nodes = [GraphNode(f"X{i + 1}") for i in range(data.shape[1])]
        bgk = BackgroundKnowledge()
        for _ in range(5):
            node1, node2 = random.sample(nodes, 2)
            bgk.add_forbidden_by_node(node1, node2)
            bgk.add_forbidden_by_node(node2, node1)
            G, edges, test_results = fas(data, nodes, cit, alpha, knowledge=bgk, verbose=False)
            assert G.num_vars == data.shape[1], 'Graph should contain the same number of nodes as variables.'
            assert all(G.get_edge(x, y) is None for x, y in bgk.forbidden_rules_specs), 'Graph contains forbidden edges.'

    @staticmethod
    def run_test_at_depths(data, cit, alpha):
        random.seed(42)

        nodes = [GraphNode(f"X{i + 1}") for i in range(data.shape[1])]
        for _ in range(3):
            depth = random.randint(1, min(data.shape[1], 5))
            G, edges, test_results = fas(data, nodes, cit, alpha, depth=depth, verbose=False)
            assert max(len(S) for _, _, S in test_results.keys()) <= depth, 'Tests performed with depth greater than maximum depth.'

    def test_bnlearn_discrete_datasets(self):
        benchmark_names = [
            "asia", "cancer", "earthquake", "sachs", "survey",
            "alarm", "barley", "child", "insurance", "water",
            "hailfinder", "hepar2", "win95pts",
            "andes"
        ]

        bnlearn_path = 'tests/TestData/bnlearn_discrete_10000/data'
        alpha = 0.05
        for bname in benchmark_names:
            print(f'Testing discrete dataset "{bname}...')
            data = np.loadtxt(os.path.join(bnlearn_path, f'{bname}.txt'), skiprows=1)
            cit = CIT(data, chisq, alpha=alpha)
            TestFAS.run_test_with_random_background(data, cit, alpha)
            TestFAS.run_test_at_depths(data, cit, alpha)

    def test_continuous_dataset(self):
        print('Testing continuous dataset...')
        data = np.loadtxt('tests/data_linear_10.txt', skiprows=1)
        alpha = 0.05
        cit = CIT(data, fisherz, alpha=alpha)
        TestFAS.run_test_with_random_background(data, cit, alpha)
        TestFAS.run_test_at_depths(data, cit, alpha)
