import os
import sys
import timeit
# get current directory
path = os.getcwd()
# get parent directory
path=os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(path)

import unittest
import numpy as np
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.search.ConstraintBased.PC import (get_adjacancy_matrix,pc)

def causal_graph_diff(cg1, cg2):
    ''' Compare the differences between the causal graphs'''
    adj1 = get_adjacancy_matrix(cg1)
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count

#####################

class Test_test_wise_deletion_PC(unittest.TestCase):

    # example1
    def test_pc_with_mv_fisherz_full_data(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data

        print('Running PC')
        start = timeit.default_timer()
        cg_pc = pc(data, 0.05, fisherz, True, 0,-1)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_pc = stop - start

        print('Running test-wise deletion PC')
        start = timeit.default_timer()
        cg_test_wise_pc = pc(data, 0.05, mv_fisherz, True, 0,-1)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_test_wise_pc = stop - start

        print('Running MVPC')
        start = timeit.default_timer()
        cg_mvpc = pc(data, 0.05, mv_fisherz, True, 0,-1, True)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_mvpc = stop - start

        print(f'Time comparison:\n pc: {t_pc}s, test-wise pc: {t_test_wise_pc}, mvpc: {t_mvpc}')
        print(f'Result differences (0 means no difference between the results of different methods):\n pc and test-wise pc: {causal_graph_diff(cg_pc, cg_test_wise_pc)}')
        print(f'Result differences (0 means no difference between the results of different methods):\n pc and mvpc: {causal_graph_diff(cg_pc, cg_mvpc)}')

    def test_pc_with_mv_fisherz_MCAR_data(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        
        #**************** baseline methods: PC on full data ****************#
        print('Running PC')
        start = timeit.default_timer()
        cg_pc = pc(data, 0.05, fisherz, True, 0,-1)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_pc = stop - start

        #**************** Test-wise deletion PC on MCAR data (Missing less than a half) ****************#
        nrow,ncol = data.shape        
        random_mask = np.random.rand(nrow,ncol) > 1
        mdata = data
        mdata[random_mask] = None
        
        print('Running test-wise deletion PC')
        start = timeit.default_timer()
        cg_test_wise_pc = pc(mdata, 0.05, mv_fisherz, True, 0,-1)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_test_wise_pc = stop - start

        print('Running MVPC')
        start = timeit.default_timer()
        cg_mvpc = pc(mdata, 0.05, mv_fisherz, True, 0,-1, True)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_mvpc = stop - start

        print(f'Time comparison:\n pc: {t_pc}s, test-wise pc: {t_test_wise_pc}, mvpc: {t_mvpc}')
        print(f'Result differences (0 means no difference between the results of different methods):\n pc and test-wise pc: {causal_graph_diff(cg_pc, cg_test_wise_pc)}')
        print(f'Result differences (0 means no difference between the results of different methods):\n pc and mvpc: {causal_graph_diff(cg_pc, cg_mvpc)}')
        
    
    def test_pc_with_mv_fisherz_MCAR_data_assertion(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data

        #**************** Test-wise deletion PC on MCAR data (About a half of the data is missing) ****************#
        print('Running test-wise deletion PC: Expect an assertion due to too many missing data ')
        nrow,ncol = data.shape        
        random_mask = np.random.rand(nrow,ncol) > 0
        mdata = data
        mdata[random_mask] = None

        start = timeit.default_timer()
        cg_test_wise_pc = pc(mdata, 0.05, mv_fisherz, True, 0,-1)  # Run PC and obtain the estimated graph (CausalGraph object)
        stop = timeit.default_timer()
        t_test_wise_pc = stop - start

################################################################

if __name__ == '__main__':
    test = Test_test_wise_deletion_PC()
    print('------------------------------')
    print('Test test-wise deletion PC and MVPC on full datasets.')
    test.test_pc_with_mv_fisherz_full_data()
    print('------------------------------')
    print('Test test-wise deletion PC and MVPC on MCAR datasets.')
    test.test_pc_with_mv_fisherz_MCAR_data()
    print('------------------------------')
    print('Test test-wise deletion PC on MCAR datasets where most values are missing.')
    test.test_pc_with_mv_fisherz_MCAR_data_assertion()
    