import sys

sys.path.append("")
import unittest

import numpy as np

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph


class TestConfusion(unittest.TestCase):
    def test_confusion_case1(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        cg = pc(data, 0.05, fisherz, True, 0, 4)
        est = cg.G

        truth_dag = txt2generalgraph("TestData/graph.10.txt")
        truth_cpdag = dag2cpdag(truth_dag)

        adj = AdjacencyConfusion(truth_cpdag, est)
        arrow = ArrowConfusion(truth_cpdag, est)

        adjTp = adj.get_adj_tp()
        adjFp = adj.get_adj_fp()
        adjFn = adj.get_adj_fn()
        adjTn = adj.get_adj_tn()

        arrowsTp = arrow.get_arrows_tp()
        arrowsFp = arrow.get_arrows_fp()
        arrowsFn = arrow.get_arrows_fn()
        arrowsTn = arrow.get_arrows_tn()
        arrowsTpCE = arrow.get_arrows_tp_ce()
        arrowsFpCE = arrow.get_arrows_fp_ce()
        arrowsFnCE = arrow.get_arrows_fn_ce()
        arrowsTnCE = arrow.get_arrows_tn_ce()

        adjPrec = adj.get_adj_precision()
        adjRec = adj.get_adj_recall()
        arrowPrec = arrow.get_arrows_precision()
        arrowRec = arrow.get_arrows_recall()
        arrowPrecCE = arrow.get_arrows_precision_ce()
        arrowRecCE = arrow.get_arrows_recall_ce()

        print(f"AdjTp: {adjTp}")
        print(f"AdjFp: {adjFp}")
        print(f"AdjFn: {adjFn}")
        print(f"AdjTn: {adjTn}")

        print(f"ArrowsTp: {arrowsTp}")
        print(f"ArrowsFp: {arrowsFp}")
        print(f"ArrowsFn: {arrowsFn}")
        print(f"ArrowsTn: {arrowsTn}")
        print(f"ArrowsTpCE: {arrowsTpCE}")
        print(f"ArrowsFpCE: {arrowsFpCE}")
        print(f"ArrowsFnCE: {arrowsFnCE}")
        print(f"ArrowsTnCE: {arrowsTnCE}")

        print(f"AdjPrec: {adjPrec}")
        print(f"AdjRec: {adjRec}")
        print(f"ArrowPrec: {arrowPrec}")
        print(f"ArrowRec: {arrowRec}")
        print(f"ArrowPrecCE: {arrowPrecCE}")
        print(f"ArrowRecCE: {arrowRecCE}")

        shd = SHD(truth_cpdag, est)
        print(f"SHD: {shd.get_shd()}")

    def test_confusion_case2(self):
        data_path = "TestData/data_linear_1.txt"
        data = np.loadtxt(data_path, skiprows=1)
        cg = pc(data, 0.05, fisherz, True, 0, 4)
        est = cg.G

        truth_dag = txt2generalgraph("TestData/graph.1.txt")
        truth_cpdag = dag2cpdag(truth_dag)

        adj = AdjacencyConfusion(truth_cpdag, est)
        arrow = ArrowConfusion(truth_cpdag, est)

        adjTp = adj.get_adj_tp()
        adjFp = adj.get_adj_fp()
        adjFn = adj.get_adj_fn()
        adjTn = adj.get_adj_tn()

        arrowsTp = arrow.get_arrows_tp()
        arrowsFp = arrow.get_arrows_fp()
        arrowsFn = arrow.get_arrows_fn()
        arrowsTn = arrow.get_arrows_tn()
        arrowsTpCE = arrow.get_arrows_tp_ce()
        arrowsFpCE = arrow.get_arrows_fp_ce()
        arrowsFnCE = arrow.get_arrows_fn_ce()
        arrowsTnCE = arrow.get_arrows_tn_ce()

        adjPrec = adj.get_adj_precision()
        adjRec = adj.get_adj_recall()
        arrowPrec = arrow.get_arrows_precision()
        arrowRec = arrow.get_arrows_recall()
        arrowPrecCE = arrow.get_arrows_precision_ce()
        arrowRecCE = arrow.get_arrows_recall_ce()

        print(f"AdjTp: {adjTp}")
        print(f"AdjFp: {adjFp}")
        print(f"AdjFn: {adjFn}")
        print(f"AdjTn: {adjTn}")

        print(f"ArrowsTp: {arrowsTp}")
        print(f"ArrowsFp: {arrowsFp}")
        print(f"ArrowsFn: {arrowsFn}")
        print(f"ArrowsTn: {arrowsTn}")
        print(f"ArrowsTpCE: {arrowsTpCE}")
        print(f"ArrowsFpCE: {arrowsFpCE}")
        print(f"ArrowsFnCE: {arrowsFnCE}")
        print(f"ArrowsTnCE: {arrowsTnCE}")

        print(f"AdjPrec: {adjPrec}")
        print(f"AdjRec: {adjRec}")
        print(f"ArrowPrec: {arrowPrec}")
        print(f"ArrowRec: {arrowRec}")
        print(f"ArrowPrecCE: {arrowPrecCE}")
        print(f"ArrowRecCE: {arrowRecCE}")

        shd = SHD(truth_cpdag, est)
        print(f"SHD: {shd.get_shd()}")
