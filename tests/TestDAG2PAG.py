import sys
import unittest

sys.path.append("")
import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils


class TestDAG2PAG(unittest.TestCase):
    # Unit test for DAG2PAG
    def test_case1(self):
        nodes = []
        for i in range(4):
            nodes.append(GraphNode(str(i)))
        dag = Dag(nodes)
        dag.add_directed_edge(nodes[0], nodes[1])
        dag.add_directed_edge(nodes[0], nodes[2])
        dag.add_directed_edge(nodes[1], nodes[3])
        dag.add_directed_edge(nodes[2], nodes[3])
        pag = dag2pag(dag, [])
        print(pag)

    def test_case2(self):
        nodes = []
        for i in range(5):
            nodes.append(GraphNode(str(i)))
        dag = Dag(nodes)
        dag.add_directed_edge(nodes[0], nodes[1])
        dag.add_directed_edge(nodes[0], nodes[2])
        dag.add_directed_edge(nodes[1], nodes[3])
        dag.add_directed_edge(nodes[2], nodes[4])
        dag.add_directed_edge(nodes[3], nodes[4])
        pag = dag2pag(dag, [nodes[0], nodes[2]])
        print(pag)

    def test_case3(self):
        nodes = []
        X = {}
        L = {}
        for i in range(7):
            node = GraphNode(f"X{i + 1}")
            nodes.append(node)
            X[i + 1] = node
        for i in range(3):
            node = GraphNode(f"L{i + 1}")
            nodes.append(node)
            L[i + 1] = node
        dag = Dag(nodes)
        dag.add_directed_edge(L[1], X[1])
        dag.add_directed_edge(L[1], X[6])

        dag.add_directed_edge(L[2], X[1])
        dag.add_directed_edge(L[2], X[7])

        dag.add_directed_edge(L[3], X[4])
        dag.add_directed_edge(L[3], X[5])
        dag.add_directed_edge(L[3], X[7])

        dag.add_directed_edge(X[1], X[2])
        dag.add_directed_edge(X[1], X[4])

        dag.add_directed_edge(X[2], X[3])

        dag.add_directed_edge(X[3], X[5])

        dag.add_directed_edge(X[6], X[7])
        pag = dag2pag(dag, [L[1], L[2], L[3]])
        print(pag)
        graphviz_pag = GraphUtils.to_pgv(pag)
        graphviz_pag.draw("pag.png", prog='dot', format='png')

    def test_case_selection(self):
        nodes = []
        for i in range(5):
            nodes.append(GraphNode(str(i)))
        dag = Dag(nodes)
        dag.add_directed_edge(nodes[0], nodes[1])
        dag.add_directed_edge(nodes[1], nodes[2])
        dag.add_directed_edge(nodes[2], nodes[3])
        # Selection nodes
        dag.add_directed_edge(nodes[3], nodes[4])
        dag.add_directed_edge(nodes[0], nodes[4])
        pag = dag2pag(dag, islatent=[], isselection=[nodes[4]])
        print(pag)
