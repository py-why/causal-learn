import os
import sys
sys.path.append("")
import unittest
import hashlib
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data



######################################### Test Notes ###########################################
# All the benchmark results of loaded files (e.g. "./TestData/benchmark_returned_results/")    #
# are obtained from the code of causal-learn as of commit                                      #
# https://github.com/cmu-phil/causal-learn/commit/94d1536 (06-29-2022).                        #
#                                                                                              #
# We are not sure if the results are completely "correct" (reflect ground truth graph) or not. #
# So if you find your tests failed, it means that your modified code is logically inconsistent #
# with the code as of 94d1536, but not necessarily means that your code is "wrong".            #
# If you are sure that your modification is "correct" (e.g. fixed some bugs in 94d1536),       #
# please report it to us. We will then modify these benchmark results accordingly. Thanks :)   #
######################################### Test Notes ###########################################


BENCHMARK_TXTFILE_TO_MD5 = {
    "./TestData/data_linear_10.txt": "95a17e15038d4cade0845140b67c05a6",
    "./TestData/data_discrete_10.txt": "ccb51c6c1946d8524a8b29a49aef2cc4",
    "./TestData/data_linear_missing_10.txt": "4e3ee59becd0fbe5fdb818154457a558",
    "./TestData/test_pc_simulated_linear_gaussian_data.txt": "ac1f99453f7e038857b692b1b3c42f3c",
    "./TestData/graph.10.txt": "4970d4ecb8be999a82a665e5f5e0825b",
    "./TestData/benchmark_returned_results/discrete_10_pc_chisq_0.05_stable_0_-1.txt": "87ebf9d830d75a5161b3a3a34ad6921f",
    "./TestData/benchmark_returned_results/discrete_10_pc_gsq_0.05_stable_0_-1.txt": "87ebf9d830d75a5161b3a3a34ad6921f",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_0.txt": "bfd3827bfa2a61bd201807925ad67d2b",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_1.txt": "8d8b6ee83725c723e480d463239b73f6",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_2.txt": "e713b5fab6a6a0e7eaeb7014b51ee63a",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_3.txt": "6ef587a2a477b5993182a64a3521a836",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_4.txt": "a9aced4cbec93970b4fe116c6c13198c",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_1_-1.txt": "5e5196b6b03094d6a5d4e3a569a95678",
    "./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_2_-1.txt": "ab84e149289d3a66afb14f06a0930444",
    "./TestData/benchmark_returned_results/linear_missing_10_mvpc_fisherz_0.05_stable_0_4.txt": "ad4f7b51bf5605f1b7a948352f4348b0",
    "./TestData/bnlearn_discrete_10000/data/alarm.txt": "234731f9e9d07cf26c2cdf50324fbd41",
    "./TestData/bnlearn_discrete_10000/data/andes.txt": "2179cb6c4da6f41d7982c5201c4812d6",
    "./TestData/bnlearn_discrete_10000/data/asia.txt": "2cc5019dada850685851046f5651216d",
    "./TestData/bnlearn_discrete_10000/data/barley.txt": "a11648ef79247b44f755de12bf8af655",
    "./TestData/bnlearn_discrete_10000/data/cancer.txt": "ce82b4f74df4046ec5a10b56cb3666ba",
    "./TestData/bnlearn_discrete_10000/data/child.txt": "1c494aef579eeff5bd4f273c5eb8e8ce",
    "./TestData/bnlearn_discrete_10000/data/earthquake.txt": "aae36bc780a74f679f4fe6f047a727fe",
    "./TestData/bnlearn_discrete_10000/data/hailfinder.txt": "566b42b5e572ba193a84559fb69bcd05",
    "./TestData/bnlearn_discrete_10000/data/hepar2.txt": "adeba165828084938998a0258f472c41",
    "./TestData/bnlearn_discrete_10000/data/insurance.txt": "c99fe6f55bba87c7d472b21293238c17",
    "./TestData/bnlearn_discrete_10000/data/sachs.txt": "b941ab1f186a6bbd15a87e1348254a39",
    "./TestData/bnlearn_discrete_10000/data/survey.txt": "0a91ac89655693f1de0535459cc43e0f",
    "./TestData/bnlearn_discrete_10000/data/water.txt": "a244e5c89070d6e35a80428383ef4225",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/alarm.graph.txt": "d6d7d0148729f3c1531f1e1c7ca5ae31",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/andes.graph.txt": "6639621629d39489ac296c50341bd6f6",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/asia.graph.txt": "c5dc87ff17dcb3d0f9b8400809e86675",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/barley.graph.txt": "f11cd8986397cfe497e94185bb94ab13",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/cancer.graph.txt": "ced3dc3128ad168b56fd94ce96500075",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/child.graph.txt": "54ebd690a78783e3dc97b41f0b407d2c",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/earthquake.graph.txt": "ced3dc3128ad168b56fd94ce96500075",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/hailfinder.graph.txt": "be4ef7093faf10ccece6bdfd25f5a16e",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/hepar2.graph.txt": "4fc4821d7697157fee1dbdae6bd0618b",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/insurance.graph.txt": "4dc73d0965f960c1e91b2c7308036e9d",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/sachs.graph.txt": "27e24b01f7b57a5c55f8919bf5f465a1",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/survey.graph.txt": "1a58f049d68aea68440897fc5fbf3d7d",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/water.graph.txt": "aecd0ce7de6adc905ec28a6cc94e72f1",
    "./TestData/bnlearn_discrete_10000/truth_dag_graph/win95pts.graph.txt": "a582c579f926d5f7aef2a1d3a9491670",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/alarm_pc_chisq_0.05_stable_0_-1.txt": "6db3f3c1f1e4eaf6efbdfbe76f80fd3c",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/asia_pc_chisq_0.05_stable_0_-1.txt": "cf20415c8e2edbfca29dc5f052e2f26c",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/barley_pc_chisq_0.05_stable_0_-1.txt": "d06e7b3c442420cc08361d008aae665c",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/cancer_pc_chisq_0.05_stable_0_-1.txt": "e72fb8c9e87ba69752425c5735f6745d",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/child_pc_chisq_0.05_stable_0_-1.txt": "2a9463e955a4c2f7c4fc0030bf1b36c2",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/earthquake_pc_chisq_0.05_stable_0_-1.txt": "36a1ff0ad26a60f3149b7a09485cf192",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/hailfinder_pc_chisq_0.05_stable_0_-1.txt": "3f125e9b13f3e5620ef1a97a3179e2ad",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/hepar2_pc_chisq_0.05_stable_0_-1.txt": "71d746e4ebfd038c0757c72e3e49a1c3",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/insurance_pc_chisq_0.05_stable_0_-1.txt": "0d3698f052e0876ef2910949b8c67b5a",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/sachs_pc_chisq_0.05_stable_0_-1.txt": "1c6ab417e2dd970304d32bfec98dc369",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/survey_pc_chisq_0.05_stable_0_-1.txt": "aa86bae4be714cdaf381772e59b18f92",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/water_pc_chisq_0.05_stable_0_-1.txt": "ca5e1f716cc8e0e205c61f403e282fdf",
    "./TestData/bnlearn_discrete_10000/benchmark_returned_results/win95pts_pc_chisq_0.05_stable_0_-1.txt": "1168e7c6795df8063298fc2f727566be",
}
INCONSISTENT_RESULT_GRAPH_ERRMSG = "Returned graph is inconsistent with the benchmark. Please check your code with the commit 94d1536."
UNROBUST_RESULT_GRAPH_ERRMSG = "Returned graph is much too different from the benchmark. Please check the randomness in your algorithm."
# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == expected_MD5,\
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/94d1536/tests/TestData'


class TestPC(unittest.TestCase):
    # Load data from file "data_linear_10.txt". Run PC with fisherz test, and different uc_rule and uc_priority.
    def test_pc_load_linear_10_with_fisher_z(self):
        print('Now start test_pc_load_linear_10_with_fisher_z ...')
        data_path = "./TestData/data_linear_10.txt"
        truth_graph_path = "./TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path) # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
        cg = pc(data, 0.05, fisherz)  # Run PC and obtain the estimated graph (cg is CausalGraph object)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_2.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=0 (overwrite)
        cg = pc(data, 0.05, fisherz, True, 0, 0)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_0.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 0)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=1 (orient bi-directed)
        cg = pc(data, 0.05, fisherz, True, 0, 1)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=3 (prioritize stronger colliders)
        cg = pc(data, 0.05, fisherz, True, 0, 3)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_3.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 3)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=4 (prioritize stronger* colliders)
        cg = pc(data, 0.05, fisherz, True, 0, 4)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_4.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 4)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=1 (maxP), uc_priority=-1 (whatever is default in uc_rule
        cg = pc(data, 0.05, fisherz, True, 1, -1)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_1_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 1, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=2 (definiteMaxP), uc_priority=-1 (whatever is default in uc_rule
        cg = pc(data, 0.05, fisherz, True, 2, -1)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_2_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 2, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('test_pc_load_linear_10_with_fisher_z passed!\n')

    # Simulate linear Gaussian data. Run PC with fisherz test with default parameters.
    def test_pc_simulate_linear_gaussian_with_fisher_z(self):
        print('Now start test_pc_simulate_linear_gaussian_with_fisher_z ...')
        # Graph specification.
        num_of_nodes = 5
        truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_directed_edges = {(0, 3), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_undirected_edges = {(0, 1), (1, 2), (2, 1), (1, 0)}
        # After the skeleton is discovered, the edges are oriented in the following way:
        # Unshilded triples:
        #   2 -- 1 -- 0: not v-structure.
        #   1 -- 2 -- 4: not v-structure.
        #   0 -- 3 -- 2: v-structure, oritented as 0 -> 3 <- 2.
        #   0 -- 3 -- 4: not v-structure.
        #   1 -- 3 -- 4: not v-structure.
        # Then by Meek rule 1: 3 -> 4.
        # Then by Meek rule 2: 2 -> 4.
        # Then by Meek rule 3: 1 -> 3.

        ###### Simulation configuration: code to generate "./TestData/test_pc_simulated_linear_gaussian_data.txt" ######
        # data = simulate_linear_continuous_data(num_of_nodes, 10000, truth_DAG_directed_edges, "gaussian", 42)
        ###### Simulation configuration: code to generate "./TestData/test_pc_simulated_linear_gaussian_data.txt" ######

        data = np.loadtxt("./TestData/test_pc_simulated_linear_gaussian_data.txt", skiprows=1)

        # Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
        cg = pc(data, 0.05, fisherz)
        returned_directed_edges = set(cg.find_fully_directed())
        returned_undirected_edges = set(cg.find_undirected())
        returned_bidirected_edges = set(cg.find_bi_directed())
        self.assertEqual(truth_CPDAG_directed_edges, returned_directed_edges, "Directed edges are not correct.")
        self.assertEqual(truth_CPDAG_undirected_edges, returned_undirected_edges, "Undirected edges are not correct.")
        self.assertEqual(0, len(returned_bidirected_edges), "There should be no bi-directed edges.")
        print(f"    pc(data, 0.05, fisherz)\treturns exactly the same CPDAG as the truth.")
        # cg.draw_pydot_graph(labels=list(map(str, range(num_of_nodes))))
        print('test_pc_simulate_linear_gaussian_with_fisher_z passed!\n')

    # Simulate linear non-Gaussian data. Run PC with kci test with default parameters.
    def test_pc_simulate_linear_nongaussian_with_kci(self):
        print('Now start test_pc_simulate_linear_nongaussian_with_kci ...')
        print('!! It will take around 17 mins to run this test (on M1 Max chip) ... !!')
        print('!! You may also reduce the sample size (<2500), but the result will then not be totally correct ... !!')

        # Graph specification.
        num_of_nodes = 5
        truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_directed_edges = {(0, 3), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_undirected_edges = {(0, 1), (1, 2), (2, 1), (1, 0)}
        # this simple graph is the same as in test_pc_simulate_linear_gaussian_with_fisher_z.

        data = simulate_linear_continuous_data(num_of_nodes, 2500, truth_DAG_directed_edges, "exponential", 42)
        # there is no randomness in data generation (with seed fixed for simulate_data).
        # however, there still exists randomness in KCI (null_sample_spectral).
        # for this simple test, we can assume that KCI always returns the correct result (despite randomness).

        # Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
        cg = pc(data, 0.05, kci)
        returned_directed_edges = set(cg.find_fully_directed())
        returned_undirected_edges = set(cg.find_undirected())
        returned_bidirected_edges = set(cg.find_bi_directed())
        self.assertEqual(truth_CPDAG_directed_edges, returned_directed_edges, "Directed edges are not correct.")
        self.assertEqual(truth_CPDAG_undirected_edges, returned_undirected_edges, "Undirected edges are not correct.")
        self.assertEqual(0, len(returned_bidirected_edges), "There should be no bi-directed edges.")
        print(f"    pc(data, 0.05, kci)\treturns exactly the same CPDAG as the truth.")
        # cg.draw_pydot_graph(labels=list(map(str, range(num_of_nodes))))
        print('test_pc_simulate_linear_nongaussian_with_kci passed!\n')

    # Simulate discrete data using forward sampling. Run PC with chisq test with default parameters.
    def test_pc_simulate_discrete_with_chisq(self):
        print('Now start test_pc_simulate_discrete_with_chisq ...')

        # Graph specification.
        num_of_nodes = 5
        truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_directed_edges = {(0, 3), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_undirected_edges = {(0, 1), (1, 2), (2, 1), (1, 0)}
        # this simple graph is the same as in test_pc_simulate_linear_gaussian_with_fisher_z.

        data = simulate_discrete_data(num_of_nodes, 10000, truth_DAG_directed_edges, 42)

        # Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
        cg = pc(data, 0.05, chisq)
        returned_directed_edges = set(cg.find_fully_directed())
        returned_undirected_edges = set(cg.find_undirected())
        returned_bidirected_edges = set(cg.find_bi_directed())
        self.assertEqual(truth_CPDAG_directed_edges, returned_directed_edges, "Directed edges are not correct.")
        self.assertEqual(truth_CPDAG_undirected_edges, returned_undirected_edges, "Undirected edges are not correct.")
        self.assertEqual(0, len(returned_bidirected_edges), "There should be no bi-directed edges.")
        print(f"    pc(data, 0.05, chisq)\treturns exactly the same CPDAG as the truth.")
        # cg.draw_pydot_graph(labels=list(map(str, range(num_of_nodes))))
        print('test_pc_simulate_discrete_with_chisq passed!\n')

    # Load data from file "data_discrete_10.txt". Run PC with gsq or chisq test.
    def test_pc_load_discrete_10_with_gsq_chisq(self):
        print('Now start test_pc_load_discrete_10_with_gsq_chisq ...')
        data_path = "./TestData/data_discrete_10.txt"
        truth_graph_path = "./TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Run PC with gsq test.
        cg = pc(data, 0.05, gsq, True, 0, -1)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/discrete_10_pc_gsq_0.05_stable_0_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, gsq, True, 0, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with chisq test.
        cg = pc(data, 0.05, chisq, True, 0, -1)
        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/discrete_10_pc_chisq_0.05_stable_0_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, chisq, True, 0, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('test_pc_load_discrete_10_with_gsq_chisq passed!\n')

    # Load data from file "data_linear_missing_10.txt". Run Missing-Value PC with mv_fisherz.
    def test_pc_load_linear_missing_10_with_mv_fisher_z(self):
        print('Now start test_pc_load_linear_10_with_fisher_z ...')
        data_path = "./TestData/data_linear_missing_10.txt"
        truth_graph_path = "./TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path)
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # since there is randomness in mvpc (np.random.shuffle in get_predictor_ws of utils/PCUtils/Helper.py),
        # we need to get two results respectively:
        #  - one with randomness to ensure that randomness is not a big problem for robustness of the algorithm end-to-end
        #  - one with no randomness (deterministic) to ensure that logic of the algorithm is consistent after any further changes
        #    (i.e., to ensure that the little difference in the results is caused by randomness, not by the logic change).
        cg_with_randomness = pc(data, 0.05, mv_fisherz, True, 0, 4, mvpc=True)
        state = np.random.get_state() # save the current random state
        np.random.seed(42) # set the random state to 42 temporarily, just for the following line
        cg_without_randomness = pc(data, 0.05, mv_fisherz, True, 0, 4, mvpc=True)
        np.random.set_state(state) # restore the random state

        benchmark_returned_graph = np.loadtxt("./TestData/benchmark_returned_results/linear_missing_10_mvpc_fisherz_0.05_stable_0_4.txt")
        assert np.all(cg_without_randomness.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        assert np.all(cg_with_randomness.G.graph != benchmark_returned_graph) / benchmark_returned_graph.size < 0.02,\
                UNROBUST_RESULT_GRAPH_ERRMSG # 0.05 is an empiric value we find here
        shd = SHD(truth_cpdag, cg_with_randomness.G)
        print(f"    pc(data, 0.05, mv_fisherz, True, 0, 4, mvpc=True)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('test_pc_load_linear_missing_10_with_mv_fisher_z passed!\n')

    # Load data from data in bnlearn repository. Run PC with chisq. Test speed.
    def test_pc_load_bnlearn_discrete_datasets(self):
        print('Now start test_pc_load_bnlearn_discrete_datasets ...')
        print('Please check SHD with truth graph and time cost with https://github.com/cmu-phil/causal-learn/pull/6.')
        benchmark_names = [
            "asia", "cancer", "earthquake", "sachs", "survey",
            "alarm", "barley", "child", "insurance", "water",
            "hailfinder", "hepar2", "win95pts",
            # "andes",
        ]

        bnlearn_data_dir = './TestData/bnlearn_discrete_10000/data'
        bnlearn_truth_dag_graph_dir = './TestData/bnlearn_discrete_10000/truth_dag_graph'
        bnlearn_benchmark_returned_results_dir = './TestData/bnlearn_discrete_10000/benchmark_returned_results'
        for bname in benchmark_names:
            data = np.loadtxt(os.path.join(bnlearn_data_dir, f'{bname}.txt'), skiprows=1)
            truth_dag = txt2generalgraph(os.path.join(bnlearn_truth_dag_graph_dir, f'{bname}.graph.txt'))
            truth_cpdag = dag2cpdag(truth_dag)
            num_edges_in_truth = truth_dag.get_num_edges()
            num_nodes_in_truth = truth_dag.get_num_nodes()
            cg = pc(data, 0.05, chisq, True, 0, -1)
            benchmark_returned_graph = np.loadtxt(
                os.path.join(bnlearn_benchmark_returned_results_dir, f'{bname}_pc_chisq_0.05_stable_0_-1.txt'))
            assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
            shd = SHD(truth_cpdag, cg.G)
            print(f'{bname} ({num_nodes_in_truth} nodes/{num_edges_in_truth} edges): used {cg.PC_elapsed:.5f}s, SHD: {shd.get_shd()}')

        print('test_pc_load_bnlearn_discrete_datasets passed!\n')
