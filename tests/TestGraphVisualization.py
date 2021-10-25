import sys

sys.path.append("")

import unittest
from pytrad.graph.Dag import Dag
from pytrad.graph.GraphNode import GraphNode
from pytrad.utils.GraphUtils import GraphUtils
from pytrad.utils.DAG2PAG import dag2pag

class testGraphVisualization(unittest.TestCase):

    def test_draw_dag(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        dag1 = Dag(nodes)

        dag1.add_directed_edge(nodes[0], nodes[1])
        dag1.add_directed_edge(nodes[0], nodes[2])
        dag1.add_directed_edge(nodes[1], nodes[2])

        pgv_g = GraphUtils.to_pgv(dag1)
        pgv_g.draw('dag.png', prog='dot', format='png')

    def test_draw_pag(self):
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

        pgv_g = GraphUtils.to_pgv(pag)
        pgv_g.draw('pag.png', prog='dot', format='png')

