import sys
sys.path.append("")
import itertools
import unittest
import hashlib
import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.DAG2CPDAG import dag2cpdag



######################################### Test Notes ###########################################
# All the benchmark results of loaded files (e.g. "./TestData/benchmark_returned_results/")    #
# are obtained from the code of causal-learn as of commit                                      #
# https://github.com/cmu-phil/causal-learn/commit/8badb41 (07-08-2022).                        #
#                                                                                              #
# We are not sure if the results are completely "correct" (reflect ground truth graph) or not. #
# So if you find your tests failed, it means that your modified code is logically inconsistent #
# with the code as of 8badb41, but not necessarily means that your code is "wrong".            #
# If you are sure that your modification is "correct" (e.g. fixed some bugs in 8badb41),       #
# please report it to us. We will then modify these benchmark results accordingly. Thanks :)   #
######################################### Test Notes ###########################################


BENCHMARK_TXTFILE_TO_MD5 = {
    "./TestData/test_exact_search_simulated_linear_gaussian_data.txt": "1ec70464e4fc68c312adfb7143bd240b",
    "./TestData/test_exact_search_simulated_linear_gaussian_CPDAG.txt": "52a6d3c5db269d5e212edcbb8283aca9",
}
# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == expected_MD5,\
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/8badb41/tests/TestData'


class TestAstar(unittest.TestCase):
    # Load data and run Astar with default parameters.
    def test_astar_simulate_linear_gaussian_with_local_score_BIC(self):
        print('Now start test_astar_simulate_linear_gaussian_with_local_score_BIC ...')
        truth_CPDAG_matrix = np.loadtxt("./TestData/test_exact_search_simulated_linear_gaussian_CPDAG.txt")
        data = np.loadtxt("./TestData/test_exact_search_simulated_linear_gaussian_data.txt")
        assert truth_CPDAG_matrix.shape[0] == truth_CPDAG_matrix.shape[1], "Should be a square numpy matrix"
        num_of_nodes = len(truth_CPDAG_matrix)
        assert data.shape[1] == num_of_nodes, "The second dimension of data should be same as number of nodes"
        data = data - data.mean(axis=0, keepdims=True)    # Center the data
        # Iterate over different configurations of path extension and k-cycle heuristic
        # to make sure they are working fine
        for use_path_extension, use_k_cycle_heuristic in itertools.product([False, True], repeat=2):
            DAG_matrix, _ = bic_exact_search(data, search_method='astar', use_path_extension=use_path_extension,
                                             use_k_cycle_heuristic=use_k_cycle_heuristic, k=3)
            # Convert DAG adjacency matrix to Dag object
            nodes = [GraphNode(str(i)) for i in range(num_of_nodes)]
            DAG = Dag(nodes)
            for i, j in zip(*np.where(DAG_matrix == 1)):
                DAG.add_directed_edge(nodes[i], nodes[j])
            CPDAG = dag2cpdag(DAG)    # Convert DAG to CPDAG
            self.assertTrue(np.all(CPDAG.graph == truth_CPDAG_matrix))
        print('test_astar_simulate_linear_gaussian_with_local_score_BIC passed!\n')