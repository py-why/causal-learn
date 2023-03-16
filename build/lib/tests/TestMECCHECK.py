import sys

sys.path.append("")
import unittest

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.MECCheck import mec_check


class TestMECCHECK(unittest.TestCase):

    def test_case1(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        dag1 = Dag(nodes)
        dag2 = Dag(nodes)

        dag1.add_directed_edge(nodes[0], nodes[1])
        dag1.add_directed_edge(nodes[0], nodes[2])
        dag1.add_directed_edge(nodes[1], nodes[2])

        dag2.add_directed_edge(nodes[0], nodes[1])
        dag2.add_directed_edge(nodes[0], nodes[2])
        dag2.add_directed_edge(nodes[2], nodes[1])
        assert mec_check(dag1, dag2)

    def test_case2(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        dag1 = Dag(nodes)
        dag2 = Dag(nodes)

        dag1.add_directed_edge(nodes[0], nodes[1])
        dag1.add_directed_edge(nodes[2], nodes[1])

        dag2.add_directed_edge(nodes[1], nodes[0])
        dag2.add_directed_edge(nodes[1], nodes[2])

        assert mec_check(dag1, dag2) is False
