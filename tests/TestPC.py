import os, time
import sys
sys.path.append("")
import unittest
import hashlib
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from .utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data



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
    "tests/TestData/data_linear_10.txt": "95a17e15038d4cade0845140b67c05a6",
    "tests/TestData/data_discrete_10.txt": "ccb51c6c1946d8524a8b29a49aef2cc4",
    "tests/TestData/data_linear_missing_10.txt": "4e3ee59becd0fbe5fdb818154457a558",
    "tests/TestData/test_pc_simulated_linear_gaussian_data.txt": "ac1f99453f7e038857b692b1b3c42f3c",
    "tests/TestData/graph.10.txt": "4970d4ecb8be999a82a665e5f5e0825b",
    "tests/TestData/benchmark_returned_results/discrete_10_pc_chisq_0.05_stable_0_-1.txt": "87ebf9d830d75a5161b3a3a34ad6921f",
    "tests/TestData/benchmark_returned_results/discrete_10_pc_gsq_0.05_stable_0_-1.txt": "87ebf9d830d75a5161b3a3a34ad6921f",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_0.txt": "e9f603b2ad14dd09b15d7b06fa5a1d75",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_1.txt": "32d350b61831ab397f9ebc9d9a1db5bb",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_2.txt": "2689fcb50cad66826034e7c76b5e586e",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_3.txt": "6ef587a2a477b5993182a64a3521a836",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_4.txt": "a9aced4cbec93970b4fe116c6c13198c",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_1_-1.txt": "e10df047f70fb78933415ba42686c95f",
    "tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_2_-1.txt": "d34ead3eea886579993f35bc08530a42",
    "tests/TestData/benchmark_returned_results/linear_missing_10_mvpc_fisherz_0.05_stable_0_4.txt": "ad4f7b51bf5605f1b7a948352f4348b0",
    "tests/TestData/bnlearn_discrete_10000/data/alarm.txt": "234731f9e9d07cf26c2cdf50324fbd41",
    "tests/TestData/bnlearn_discrete_10000/data/andes.txt": "2179cb6c4da6f41d7982c5201c4812d6",
    "tests/TestData/bnlearn_discrete_10000/data/asia.txt": "2cc5019dada850685851046f5651216d",
    "tests/TestData/bnlearn_discrete_10000/data/barley.txt": "a11648ef79247b44f755de12bf8af655",
    "tests/TestData/bnlearn_discrete_10000/data/cancer.txt": "ce82b4f74df4046ec5a10b56cb3666ba",
    "tests/TestData/bnlearn_discrete_10000/data/child.txt": "1c494aef579eeff5bd4f273c5eb8e8ce",
    "tests/TestData/bnlearn_discrete_10000/data/earthquake.txt": "aae36bc780a74f679f4fe6f047a727fe",
    "tests/TestData/bnlearn_discrete_10000/data/hailfinder.txt": "566b42b5e572ba193a84559fb69bcd05",
    "tests/TestData/bnlearn_discrete_10000/data/hepar2.txt": "adeba165828084938998a0258f472c41",
    "tests/TestData/bnlearn_discrete_10000/data/insurance.txt": "c99fe6f55bba87c7d472b21293238c17",
    "tests/TestData/bnlearn_discrete_10000/data/sachs.txt": "b941ab1f186a6bbd15a87e1348254a39",
    "tests/TestData/bnlearn_discrete_10000/data/survey.txt": "0a91ac89655693f1de0535459cc43e0f",
    "tests/TestData/bnlearn_discrete_10000/data/water.txt": "a244e5c89070d6e35a80428383ef4225",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/alarm.graph.txt": "d6d7d0148729f3c1531f1e1c7ca5ae31",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/andes.graph.txt": "6639621629d39489ac296c50341bd6f6",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/asia.graph.txt": "c5dc87ff17dcb3d0f9b8400809e86675",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/barley.graph.txt": "f11cd8986397cfe497e94185bb94ab13",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/cancer.graph.txt": "ced3dc3128ad168b56fd94ce96500075",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/child.graph.txt": "54ebd690a78783e3dc97b41f0b407d2c",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/earthquake.graph.txt": "ced3dc3128ad168b56fd94ce96500075",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/hailfinder.graph.txt": "be4ef7093faf10ccece6bdfd25f5a16e",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/hepar2.graph.txt": "4fc4821d7697157fee1dbdae6bd0618b",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/insurance.graph.txt": "4dc73d0965f960c1e91b2c7308036e9d",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/sachs.graph.txt": "27e24b01f7b57a5c55f8919bf5f465a1",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/survey.graph.txt": "1a58f049d68aea68440897fc5fbf3d7d",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/water.graph.txt": "aecd0ce7de6adc905ec28a6cc94e72f1",
    "tests/TestData/bnlearn_discrete_10000/truth_dag_graph/win95pts.graph.txt": "a582c579f926d5f7aef2a1d3a9491670",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/alarm_pc_chisq_0.05_stable_0_-1.txt": "c1329debdc6fe7dd81f87b59e45cf007",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/asia_pc_chisq_0.05_stable_0_-1.txt": "cf20415c8e2edbfca29dc5f052e2f26c",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/barley_pc_chisq_0.05_stable_0_-1.txt": "d06e7b3c442420cc08361d008aae665c",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/cancer_pc_chisq_0.05_stable_0_-1.txt": "e72fb8c9e87ba69752425c5735f6745d",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/child_pc_chisq_0.05_stable_0_-1.txt": "6af09c1c7b953f0afc250a9d52d57f9a",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/earthquake_pc_chisq_0.05_stable_0_-1.txt": "36a1ff0ad26a60f3149b7a09485cf192",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/hailfinder_pc_chisq_0.05_stable_0_-1.txt": "052841152799b8e90b8bffae802c88e8",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/hepar2_pc_chisq_0.05_stable_0_-1.txt": "594638c6173b4a7b1f987024076da9e8",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/insurance_pc_chisq_0.05_stable_0_-1.txt": "26f8915f9a070746aece1b8ce82754de",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/sachs_pc_chisq_0.05_stable_0_-1.txt": "bd648b70501bf122c800ea282aca000c",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/survey_pc_chisq_0.05_stable_0_-1.txt": "aa86bae4be714cdaf381772e59b18f92",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/water_pc_chisq_0.05_stable_0_-1.txt": "9695c5ffbb123666ae8c396c89f15fc1",
    "tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/win95pts_pc_chisq_0.05_stable_0_-1.txt": "1168e7c6795df8063298fc2f727566be",
}
INCONSISTENT_RESULT_GRAPH_ERRMSG = "Returned graph is inconsistent with the benchmark. Please check your code with the commit 94d1536."
UNROBUST_RESULT_GRAPH_ERRMSG = "Returned graph is much too different from the benchmark. Please check the randomness in your algorithm."
# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        # assert hashlib.md5(fin.read()).hexdigest() == expected_MD5,\
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/94d1536/tests/TestData'


class TestPC(unittest.TestCase):
    # Load data from file "data_linear_10.txt". Run PC with fisherz test, and different uc_rule and uc_priority.
    def test_pc_load_linear_10_with_fisher_z(self):
        print('Now start test_pc_load_linear_10_with_fisher_z ...')
        data_path = "tests/TestData/data_linear_10.txt"
        truth_graph_path = "tests/TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path) # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
        cg = pc(data, 0.05, fisherz)  # Run PC and obtain the estimated graph (cg is CausalGraph object)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_2.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=0 (overwrite)
        cg = pc(data, 0.05, fisherz, True, 0, 0)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_0.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 0)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=1 (orient bi-directed)
        cg = pc(data, 0.05, fisherz, True, 0, 1)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=3 (prioritize stronger colliders)
        cg = pc(data, 0.05, fisherz, True, 0, 3)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_3.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 3)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=4 (prioritize stronger* colliders)
        cg = pc(data, 0.05, fisherz, True, 0, 4)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_0_4.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 0, 4)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=1 (maxP), uc_priority=-1 (whatever is default in uc_rule
        cg = pc(data, 0.05, fisherz, True, 1, -1)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_1_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, fisherz, True, 1, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with: stable=True, uc_rule=2 (definiteMaxP), uc_priority=-1 (whatever is default in uc_rule
        cg = pc(data, 0.05, fisherz, True, 2, -1)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_10_pc_fisherz_0.05_stable_2_-1.txt")
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

        ###### Simulation configuration: code to generate "tests/TestData/test_pc_simulated_linear_gaussian_data.txt" ######
        # data = simulate_linear_continuous_data(num_of_nodes, 10000, truth_DAG_directed_edges, "gaussian", 42)
        ###### Simulation configuration: code to generate "tests/TestData/test_pc_simulated_linear_gaussian_data.txt" ######

        data = np.loadtxt("tests/TestData/test_pc_simulated_linear_gaussian_data.txt", skiprows=1)

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
        truth_CPDAG_directed_edges = set()
        truth_CPDAG_undirected_edges = {(0, 1), (2, 4), (1, 2), (2, 1), (3, 4), (4, 3), (3, 1), (0, 3), (2, 0), (4, 2), (3, 0), (2, 3), (0, 2), (1, 0), (3, 2), (1, 3)}
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
        data_path = "tests/TestData/data_discrete_10.txt"
        truth_graph_path = "tests/TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Run PC with gsq test.
        cg = pc(data, 0.05, gsq, True, 0, -1)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/discrete_10_pc_gsq_0.05_stable_0_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, gsq, True, 0, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        # Run PC with chisq test.
        cg = pc(data, 0.05, chisq, True, 0, -1)
        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/discrete_10_pc_chisq_0.05_stable_0_-1.txt")
        assert np.all(cg.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        shd = SHD(truth_cpdag, cg.G)
        print(f"    pc(data, 0.05, chisq, True, 0, -1)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('test_pc_load_discrete_10_with_gsq_chisq passed!\n')

    # Load data from file "data_linear_missing_10.txt". Run Missing-Value PC with mv_fisherz.
    def test_pc_load_linear_missing_10_with_mv_fisher_z(self):
        print('Now start test_pc_load_linear_10_with_fisher_z ...')
        data_path = "tests/TestData/data_linear_missing_10.txt"
        truth_graph_path = "tests/TestData/graph.10.txt"
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

        benchmark_returned_graph = np.loadtxt("tests/TestData/benchmark_returned_results/linear_missing_10_mvpc_fisherz_0.05_stable_0_4.txt")
        assert np.all(cg_without_randomness.G.graph == benchmark_returned_graph), INCONSISTENT_RESULT_GRAPH_ERRMSG
        assert np.all(cg_with_randomness.G.graph != benchmark_returned_graph) / benchmark_returned_graph.size < 0.02, UNROBUST_RESULT_GRAPH_ERRMSG # 0.05 is an empiric value we find here
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

        bnlearn_data_dir = 'tests/TestData/bnlearn_discrete_10000/data'
        bnlearn_truth_dag_graph_dir = 'tests/TestData/bnlearn_discrete_10000/truth_dag_graph'
        bnlearn_benchmark_returned_results_dir = 'tests/TestData/bnlearn_discrete_10000/benchmark_returned_results'
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

    # Test the usage of local cache checkpoint (check speed).
    def test_pc_with_citest_local_checkpoint(self):
        print('Now start test_pc_with_citest_local_checkpoint ...')
        data_path = "tests/TestData/data_linear_10.txt"
        citest_cache_file = "tests/TestData/citest_cache_linear_10_first_500_kci.json"

        tic = time.time()
        data = np.loadtxt(data_path, skiprows=1)[:500]
        cg1 = pc(data, 0.05, kci, cache_path=citest_cache_file)
        tac = time.time()
        print(f'First pc run takes {tac - tic:.3f}s.')  # First pc run takes 125.663s.
        # assert os.path.exists(citest_cache_file), 'Cache file should exist.'

        tic = time.time()
        data = np.loadtxt(data_path, skiprows=1)[:500]
        cg2 = pc(data, 0.05, kci, cache_path=citest_cache_file)
        # you might also try other rules of PC, e.g., pc(data, 0.05, kci, True, 0, -1, cache_path=citest_cache_file)
        tac = time.time()
        print(f'Second pc run takes {tac - tic:.3f}s.')  # Second pc run takes 27.316s.
        # assert np.all(cg1.G.graph == cg2.G.graph), INCONSISTENT_RESULT_GRAPH_ERRMSG

        print('test_pc_with_citest_local_checkpoint passed!\n')

    # Test graphs in bnlearn repository with d-separation as cit. Ensure PC's correctness.
    def test_pc_load_bnlearn_graphs_with_d_separation(self):
        import networkx as nx
        print('Now start test_pc_load_bnlearn_graphs_with_d_separation ...')
        benchmark_names = [
            "asia", "cancer", "earthquake", "sachs", "survey",
            "alarm", "barley", "child", "insurance", "water",
            "hailfinder", "hepar2", "win95pts",
        ]
        bnlearn_truth_dag_graph_dir = 'tests/TestData/bnlearn_discrete_10000/truth_dag_graph'
        for bname in benchmark_names:
            truth_dag = txt2generalgraph(os.path.join(bnlearn_truth_dag_graph_dir, f'{bname}.graph.txt'))
            truth_cpdag = dag2cpdag(truth_dag)
            num_edges_in_truth = truth_dag.get_num_edges()
            num_nodes_in_truth = truth_dag.get_num_nodes()

            true_dag_netx = nx.DiGraph()
            true_dag_netx.add_nodes_from(list(range(num_nodes_in_truth)))
            true_dag_netx.add_edges_from(set(map(tuple, np.argwhere(truth_dag.graph.T > 0))))

            data = np.zeros((100, len(truth_dag.nodes)))  # just a placeholder
            cg = pc(data, 0.05, d_separation, True, 0, -1, true_dag=true_dag_netx)
            shd = SHD(truth_cpdag, cg.G)
            self.assertEqual(0, shd.get_shd(), "PC with d-separation as CIT returns an inaccurate CPDAG.")
            print(f'{bname} ({num_nodes_in_truth} nodes/{num_edges_in_truth} edges): used {cg.PC_elapsed:.5f}s, SHD: {shd.get_shd()}')

        print('test_pc_load_bnlearn_graphs_with_d_separation passed!\n')

    
