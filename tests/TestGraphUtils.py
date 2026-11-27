import sys

sys.path.append("")
import unittest

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
from causallearn.utils.GraphUtils import GraphUtils


class TestGraphUtils(unittest.TestCase):
    @staticmethod
    def graphs(number_of_nodes=3):
        nodes = [GraphNode(f"X{i}") for i in range(number_of_nodes)]
        return nodes, GeneralGraph(nodes), GeneralGraph(nodes)

    @staticmethod
    def add_edge(graph, node1, node2, endpoint1, endpoint2):
        graph.add_edge(Edge(node1, node2, endpoint1, endpoint2))

    def test_identical_graphs_with_empty_pair(self):
        nodes, graph1, graph2 = self.graphs()
        graph1.add_directed_edge(nodes[0], nodes[1])
        graph2.add_directed_edge(nodes[0], nodes[1])

        self.assertEqual(GraphUtils().sdh(graph1, graph2), 0)
        self.assertEqual(GraphUtils().sdh(graph1, graph2), SHD(graph1, graph2).get_shd())

    def test_edges_involving_first_node(self):
        nodes, empty, directed = self.graphs(2)
        directed.add_directed_edge(nodes[0], nodes[1])
        self.assertEqual(GraphUtils().sdh(empty, directed), 2)

        nodes, empty, undirected = self.graphs(2)
        self.add_edge(undirected, nodes[0], nodes[1], Endpoint.TAIL, Endpoint.TAIL)
        self.assertEqual(GraphUtils().sdh(empty, undirected), 1)

    def test_directed_orientation(self):
        nodes, graph1, graph2 = self.graphs()
        graph1.add_directed_edge(nodes[1], nodes[2])
        graph2.add_directed_edge(nodes[2], nodes[1])
        self.assertEqual(GraphUtils().sdh(graph1, graph2), 1)

        nodes, graph1, graph2 = self.graphs(2)
        graph1.add_directed_edge(nodes[0], nodes[1])
        graph2.add_directed_edge(nodes[1], nodes[0])
        self.assertEqual(GraphUtils().sdh(graph1, graph2), 1)

        nodes, graph1, graph2 = self.graphs(3)
        self.add_edge(graph1, nodes[2], nodes[1], Endpoint.ARROW, Endpoint.TAIL)
        self.add_edge(graph2, nodes[1], nodes[2], Endpoint.TAIL, Endpoint.ARROW)
        self.assertEqual(GraphUtils().sdh(graph1, graph2), 0)

    def test_other_edge_costs(self):
        nodes, undirected, directed = self.graphs(2)
        self.add_edge(undirected, nodes[0], nodes[1], Endpoint.TAIL, Endpoint.TAIL)
        directed.add_directed_edge(nodes[0], nodes[1])
        self.assertEqual(GraphUtils().sdh(undirected, directed), 1)

        nodes, bidirected, missing = self.graphs(3)
        self.add_edge(bidirected, nodes[1], nodes[2], Endpoint.ARROW, Endpoint.ARROW)
        self.assertEqual(GraphUtils().sdh(bidirected, missing), 2)

        nodes, bidirected, directed = self.graphs(2)
        self.add_edge(bidirected, nodes[0], nodes[1], Endpoint.ARROW, Endpoint.ARROW)
        directed.add_directed_edge(nodes[0], nodes[1])
        self.assertEqual(GraphUtils().sdh(bidirected, directed), 2)

    def test_sdh_is_symmetric_for_mixed_graphs(self):
        nodes, graph1, graph2 = self.graphs(4)
        graph1.add_directed_edge(nodes[0], nodes[1])
        self.add_edge(graph1, nodes[1], nodes[2], Endpoint.TAIL, Endpoint.TAIL)
        self.add_edge(graph1, nodes[2], nodes[3], Endpoint.ARROW, Endpoint.ARROW)

        graph2.add_directed_edge(nodes[1], nodes[0])
        graph2.add_directed_edge(nodes[2], nodes[3])
        graph2.add_directed_edge(nodes[0], nodes[3])

        graph_utils = GraphUtils()
        self.assertEqual(graph_utils.sdh(graph1, graph2), 6)
        self.assertEqual(graph_utils.sdh(graph1, graph2), graph_utils.sdh(graph2, graph1))

    def test_sdh_with_fresh_reversed_node_order(self):
        nodes = [GraphNode(f"X{i}") for i in range(4)]
        new_nodes = [GraphNode(f"X{i}") for i in range(4)]
        graph1 = GeneralGraph(nodes)
        graph2 = GeneralGraph(list(reversed(new_nodes)))

        graph1.add_directed_edge(nodes[0], nodes[1])
        graph2.add_directed_edge(new_nodes[1], new_nodes[0])
        graph1.add_directed_edge(nodes[1], nodes[2])
        graph2.add_directed_edge(new_nodes[1], new_nodes[2])
        self.add_edge(graph2, new_nodes[2], new_nodes[3], Endpoint.TAIL, Endpoint.TAIL)

        graph_utils = GraphUtils()
        self.assertEqual(graph_utils.sdh(graph1, graph2), 2)
        self.assertEqual(graph_utils.sdh(graph2, graph1), 2)

    def test_agrees_with_shd_for_reversed_dag_edge(self):
        nodes, graph1, graph2 = self.graphs(2)
        graph1.add_directed_edge(nodes[0], nodes[1])
        graph2.add_directed_edge(nodes[1], nodes[0])

        self.assertEqual(GraphUtils().sdh(graph1, graph2), SHD(graph1, graph2).get_shd())
