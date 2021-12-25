import sys

sys.path.append("")
import unittest
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci


class TestCDNOD(unittest.TestCase):

    # example1: time-varying data
    def test_cdnod_timevary(self):
        data_path = "data_cdnod.txt"
        data = np.loadtxt(data_path, skiprows=1)[:500, :]  # Import the file at data_path as data
        c_indx = np.reshape(np.asarray(list(range(data.shape[0]))), (data.shape[0],1)) # used to capture the unobserved time-varying factors
        # kci test is recommended for time-varying distributions
        cg = cdnod(data, c_indx, 0.05, kci, True, 0,
                -1)  # Run CDNOD and obtain the estimated augmented graph (CausalGraph object)

        # visualization using pydot
        # note that the last node is the c_indx
        cg.draw_pydot_graph()

        print('finish')


    # example2: domain-varying data
    def test_cdnod_domainvary(self):
        data1_path = "TestData/data_linear_1.txt"
        data1 = np.loadtxt(data1_path, skiprows=1)[:100, :]  # Import the file at data_path as data
        data2_path = "TestData/data_linear_2.txt"
        data2 = np.loadtxt(data2_path, skiprows=1)[:100, :]  # Import the file at data_path as data
        data3_path = "TestData/data_linear_3.txt"
        data3 = np.loadtxt(data3_path, skiprows=1)[:100, :]  # Import the file at data_path as data
        data = np.concatenate((data1, data2, data3))
        # c_indx is used to capture the unobserved domain-varying factors
        c_indx = np.concatenate((np.ones((data1.shape[0], 1)), 2*np.ones((data2.shape[0], 1)), 3*np.ones((data3.shape[0], 1))))
        cg = cdnod(data, c_indx, 0.05, kci, True, 0,
                -1)  # Run CDNOD and obtain the estimated augmented graph (CausalGraph object)

        # visualization using pydot
        # note that the last node is the c_indx
        cg.draw_pydot_graph()

        print('finish')

