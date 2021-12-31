import sys
import time

sys.path.append("")
import unittest

import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz


class TestPC(unittest.TestCase):

    # example1
    def test_pc_with_fisher_z(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, False, 0,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example2
    def test_pc_with_g_sq(self):
        data_path = "data_discrete_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, gsq, True, 0,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example3
    def test_pc_with_chi_sq(self):
        data_path = "data_discrete_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, chisq, True, 0,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example4
    def test_pc_with_fisher_z_maxp(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 1,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example5
    def test_pc_with_fisher_z_definite_maxp(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 2,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example6
    def test_pc_with_fisher_z_with_uc_priority0(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                0)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example7
    def test_pc_with_fisher_z_with_uc_priority1(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example8
    def test_pc_with_fisher_z_with_uc_priority2(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                2)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example9
    def test_pc_with_fisher_z_with_uc_priority3(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                3)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example10
    def test_pc_with_fisher_z_with_uc_priority4(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                4)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example11
    def test_pc_with_mv_fisher_z_with_uc_priority4(self):
        data_path = "data_linear_missing_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data

        cg = pc(data, 0.05, mv_fisherz, True, 0,
                4, mvpc=True)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example12
    def test_pc_with_kci(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)[:50, :]  # Import the file at data_path as data
        cg = pc(data, 0.05, kci, True, 0,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)

        # visualization using pydot
        # cg.draw_pydot_graph()

        # visualization using networkx
        # cg.to_nx_graph()
        # cg.draw_nx_graph(skel=False)

        print('finish')

    # example13
    def test_bnlearn_discrete_datasets(self):
        import os
        benchmark_names = [
            "asia", "cancer", "earthquake", "sachs", "survey",
            "alarm", "barley", "child", "insurance", "water",
            "hailfinder", "hepar2", "win95pts",
            "andes",
        ]

        bnlearn_path = './TestData/bnlearn_discrete_10000'
        for bname in benchmark_names:
            data = np.loadtxt(os.path.join(bnlearn_path, f'{bname}.txt'), skiprows=1)
            cg = pc(data, 0.05, chisq, True, 0, -1)  # Run PC and obtain the estimated graph (CausalGraph object)
            print(f'{bname}: used {cg.PC_elapsed:.5f}s')
            # visualization using pydot
            # cg.draw_pydot_graph()

            # visualization using networkx
            # cg.to_nx_graph()
            # cg.draw_nx_graph(skel=False)
