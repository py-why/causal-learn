import sys

sys.path.append("")

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.PC import get_adjacancy_matrix, pc
from causallearn.utils.cit import fisherz, mv_fisherz


def load(filename):
    return pd.read_csv(filename, index_col=0).to_numpy()


def matrix_diff(adj1, cg2):
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count


class TestMVPC(unittest.TestCase):
    def test_mar_syn(self):
        res_full = []
        res_ref = []
        res_mar = []
        res_mvpc = []

        data_ref = load(f'mdata/mar_ref_10_5.csv')
        data_mar = load(f'mdata/mar_mv_10_5.csv')
        data_full = load(f'mdata/mar_full_10_5.csv')

        cpdag = pd.read_csv(f'mdata/cpdag_matrix.csv', index_col=0).to_numpy()

        cg_full = pc(data_full, 0.01, fisherz, True, 0,
                     -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        cg_full.to_nx_graph()

        cg_ref = pc(data_ref, 0.01, mv_fisherz, True, 0,
                    -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        cg_ref.to_nx_graph()

        cg_mar = pc(data_mar, 0.01, mv_fisherz, True, 0,
                    -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        cg_mar.to_nx_graph()

        mvpc_cg_mar = pc(data_mar, 0.01, mv_fisherz, True, 0,
                         -1, True)  # Run PC and obtain the estimated graph (CausalGraph object)
        mvpc_cg_mar.to_nx_graph()

        res_full.append(matrix_diff(cpdag, cg_full))
        res_ref.append(matrix_diff(cpdag, cg_ref))
        res_mar.append(matrix_diff(cpdag, cg_mar))
        res_mvpc.append(matrix_diff(cpdag, mvpc_cg_mar))

        #
        # for seed in range(50):
        #     seed = seed + 1
        #     np.random.seed(0)
        #
        #     print('seed', seed)
        #     data_ref = load(f'mdata/mar_ref_10_5_{seed}.csv')
        #     data_mar = load(f'mdata/mar_mv_10_5_{seed}.csv')
        #     data_full = load(f'mdata/mar_full_10_5_{seed}.csv')
        #
        #     cpdag = pd.read_csv(f'mdata/cpdag_mar_{seed}.csv', index_col=0).to_numpy()
        #
        #     cg_full = pc(data_full, 0.01, "Fisher_Z", True, 0,
        #                           -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        #     cg_full.toNxGraph()
        #
        #     cg_ref = pc(data_ref, 0.01, "MV_Fisher_Z", True, 0,
        #                          -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        #     cg_ref.toNxGraph()
        #
        #     cg_mar = pc(data_mar, 0.01, "MV_Fisher_Z", True, 0,
        #                          -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        #     cg_mar.toNxGraph()
        #
        #     mvpc_cg_mar = pc(data_mar, 0.01, "MV_Fisher_Z", True, 0,
        #                                 -1, True)  # Run PC and obtain the estimated graph (CausalGraph object)
        #     mvpc_cg_mar.toNxGraph()
        #
        #     res_full.append(matrix_diff(cpdag, cg_full))
        #     res_ref.append(matrix_diff(cpdag, cg_ref))
        #     res_mar.append(matrix_diff(cpdag, cg_mar))
        #     res_mvpc.append(matrix_diff(cpdag, mvpc_cg_mar))

        # 0，01 Full:  2.58 2.52 ; Ref:  3.02 2.51 ; Mar:  9.86 4.52 ; MVPC 4.4 3.58
        # 0，05 Full:  8.8 4.11 ; Ref:  3.5 2.7 ; Mar:  11.92 4.60 ; MVPC 5.28 3.96
        print('Full: ', np.mean(res_full), np.std(res_full), "; Ref: ", np.mean(res_ref), np.std(res_ref), "; Mar: ",
              np.mean(res_mar), np.std(res_mar), "; MVPC", np.mean(res_mvpc), np.std(res_mvpc))

    def test_5var(self):
        sz = 100000
        data = np.zeros((sz, 4))

        X = np.random.normal(0, 1.0, size=sz)
        Z = 2 * X + 0.5 * np.random.normal(0, 1.0, size=sz)
        Y = 0.5 * Z + 0.5 * np.random.normal(0, 1.0, size=sz)
        W = 0.2 * X + 0.8 * Y + 0.5 * np.random.normal(0, 1.0, size=sz)
        U = np.random.normal(0, 1.0, size=sz)
        data[:, 0], data[:, 1], data[:, 2], data[:, 3] = X, Y, Z, W
        mdata = data.copy()

        # X--> Z -->Y
        # X--> W <--Y
        # X--> W2<--Y
        # W --> Rx
        # W2 --> Ry
        # U --> Rw2

        mdata[W > 0, 0] = np.nan
        mdata[U > 0, 3] = np.nan

        cg_full = pc(data, 0.05, fisherz, True, 0,
                     -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        cg_full.to_nx_graph()
        cg_full.to_nx_skeleton()

        cg_mar = pc(mdata, 0.05, mv_fisherz, True, 0,
                    -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        cg_mar.to_nx_graph()
        cg_mar.to_nx_skeleton()

        mvpc_cg_mar = pc(mdata, 0.05, mv_fisherz, True, 0,
                         -1, True)  # Run PC and obtain the estimated graph (CausalGraph object)
        mvpc_cg_mar.to_nx_graph()
        mvpc_cg_mar.to_nx_skeleton()
        plt.subplot(1, 3, 1)
        plt.title('complete')
        cg_full.draw_nx_graph(skel=False)  # Draw the estimated graph (or its skeleton)
        plt.subplot(1, 3, 2)
        plt.title('test-wise deletion')
        cg_mar.draw_nx_graph(skel=False)  # Draw the estimated graph (or its skeleton)
        plt.subplot(1, 3, 3)
        plt.title('test-wise MVPC')
        mvpc_cg_mar.draw_nx_graph(skel=False)  # Draw the estimated graph (or its skeleton)
        plt.show()
