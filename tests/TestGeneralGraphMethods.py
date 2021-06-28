#!/usr/bin/env python3
import unittest
from graph.GeneralGraph import GeneralGraph
from graph.GraphNode import GraphNode
from graph.Edge import Edge
from graph.Endpoint import Endpoint
import numpy as np

class TestGeneralGraphMethods(unittest.TestCase):

    def test_set_nodes(self):
        node_names = ["x1", "x2", "x3"]
        nodes = []

        for name in node_names:
            node = GraphNode(name)
            nodes.append(node)

        dag = GeneralGraph(nodes)

        new_nodes = ["x", "y", "z"]

        dag.set_nodes(new_nodes)
        self.assertEqual(dag.get_nodes(), new_nodes)

    # def test_add_directed_edge(self):
    #
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #
    #     dag.add_directed_edge(node1, node2)
    #
    #     true_graph = np.array([(0, -1, 0),(1, 0, 0),(0, 0, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)

    # def test_add_edge(self):
    #
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #
    #     dag.add_edge(edge)
    #
    #     true_graph = np.array([(0, -1, 0), (1, 0, 0), (0, 0, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)
    #
    #     edge = Edge(node1, node2, 6, 6)
    #     dag.add_edge(edge)
    #
    #     true_graph = np.array([(0, 5, 0), (7, 0, 0), (0, 0, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)

    # def test_add_node(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #
    #     dag.add_edge(edge)
    #
    #     node3 = GraphNode("x4")
    #     dag.add_node(node3)
    #
    #     true_graph = np.array([(0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)

    # def test_contains_edge(self):
    #
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node1, node3, -1, 1)
    #
    #     dag.add_edge(edge)
    #
    #     self.assertTrue(dag.contains_edge(edge))
    #     self.assertFalse(dag.contains_edge(edge2))
    #
    #     edge3  = Edge(node1, node2, 6, 6)
    #     dag.add_edge(edge3)
    #
    #     self.assertTrue(dag.contains_edge(edge3))
    #
    #     edge4 = Edge(node1, node3, 1, 2)
    #     edge5 = Edge(node1, node3, 6, 6)
    #
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #
    #     self.assertTrue(dag.contains_edge(edge4))
    #     self.assertFalse(dag.contains_edge(edge5))

    # def test_get_adjacent_nodes(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node1, node3, -1, 2)
    #     edge3 = Edge(node1, node4, 6, 6)
    #     edge4 = Edge(node1, node5, 2, 2)
    #     edge5 = Edge(node1, node6, -1, -1)
    #     edge6 = Edge(node1, node2, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge6)
    #
    #     adj_nodes = dag.get_adjacent_nodes(node1)
    #
    #     self.assertTrue(node2 in adj_nodes)
    #     self.assertTrue(node3 in adj_nodes)
    #     self.assertTrue(node4 in adj_nodes)
    #     self.assertTrue(node5 in adj_nodes)
    #     self.assertTrue(node6 in adj_nodes)
    #     self.assertTrue(len(adj_nodes) == 5)

    # def test_get_parents(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node4, node2, -1, 2)
    #     edge4 = Edge(node5, node2, -1, -1)
    #     edge5 = Edge(node6, node2, 2, 2)
    #     edge6 = Edge(node3, node2, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge6)
    #
    #     adj_nodes = dag.get_parents(node2)
    #
    #     self.assertTrue(node1 in adj_nodes)
    #     self.assertTrue(node3 in adj_nodes)
    #     self.assertTrue(len(adj_nodes) == 2)

    # def test_get_ancestors(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node4, node3, -1, 1)
    #     edge4 = Edge(node5, node2, 2, 1)
    #     edge5 = Edge(node6, node2, -1, -1)
    #
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #
    #     node_list = [node2]
    #
    #     adj_nodes = dag.get_ancestors(node_list)
    #
    #     self.assertTrue(node2 in adj_nodes)
    #     self.assertTrue(node3 in adj_nodes)
    #     self.assertTrue(node4 in adj_nodes)
    #     self.assertTrue(len(adj_nodes) == 3)

    # def test_get_children(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #
    #     edge = Edge(node2, node1, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node2, node4, -1, -1)
    #     edge4 = Edge(node2, node5, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #
    #     adj_nodes = dag.get_children(node2)
    #
    #     self.assertTrue(node1 in adj_nodes)
    #     self.assertTrue(node3 in adj_nodes)
    #     self.assertTrue(len(adj_nodes) == 2)

    # def test_get_indegree(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node4, node2, 2, 1)
    #     edge4 = Edge(node5, node2, 6, 6)
    #     edge5 = Edge(node6, node2, 6, 6)
    #     edge6 = Edge(node6, node2, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #
    #     print(str(dag))
    #
    #     indegree = dag.get_indegree(node2)
    #
    #     self.assertEqual(indegree, 6)

    # def test_get_outdegree(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge = Edge(node2, node1, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node2, node4, -1, -1)
    #     edge4 = Edge(node2, node5, 6, 6)
    #     edge5 = Edge(node2, node6, 6, 6)
    #     edge6 = Edge(node2, node6, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #
    #     print(str(dag.graph))
    #
    #     outdegree = dag.get_outdegree(node2)
    #
    #     self.assertEqual(outdegree, 4)

    # def test_get_degree(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, -1)
    #     edge3 = Edge(node2, node4, 2, 2)
    #     edge4 = Edge(node2, node5, 6, 6)
    #     edge5 = Edge(node2, node6, -1, 1)
    #     edge6 = Edge(node2, node6, 6, 6)
    #     edge7 = Edge(node2, node7, -1, 1)
    #
    #     print(str(dag.add_edge(edge)))
    #     print(str(dag.add_edge(edge2)))
    #     print(str(dag.add_edge(edge3)))
    #     print(str(dag.add_edge(edge4)))
    #     print(str(dag.add_edge(edge6)))
    #     print(str(dag.add_edge(edge5)))
    #     print(str(dag.add_edge(edge7)))
    #
    #     print(str(dag))
    #
    #     degree = dag.get_degree(node2)
    #
    #     self.assertEqual(degree, 7)

    # def test_get_num_edges(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, -1)
    #     edge3 = Edge(node2, node4, 2, 2)
    #     edge4 = Edge(node2, node5, 6, 6)
    #     edge5 = Edge(node2, node6, -1, 1)
    #     edge6 = Edge(node2, node6, 6, 6)
    #     edge7 = Edge(node2, node7, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge7)
    #
    #     num_edges = dag.get_num_edges()
    #
    #     self.assertEqual(num_edges, 7)

    # def test_get_num_connected_edges(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node2, node4, 2, 2)
    #     edge4 = Edge(node2, node5, 6, 6)
    #     edge5 = Edge(node2, node6, -1, 1)
    #     edge6 = Edge(node2, node6, 6, 6)
    #     edge7 = Edge(node2, node7, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge7)
    #
    #     num_edges = dag.get_num_connected_edges(node2)
    #
    #     self.assertEqual(num_edges, 7)

    # def test_is_adjacent_to(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, -1)
    #     edge3 = Edge(node2, node4, 2, 2)
    #     edge4 = Edge(node2, node5, 6, 6)
    #     edge5 = Edge(node2, node6, -1, 1)
    #     edge6 = Edge(node2, node6, 6, 6)
    #     edge7 = Edge(node2, node7, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge7)
    #
    #     self.assertTrue(dag.is_adjacent_to(node1, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node1))
    #
    #     self.assertTrue(dag.is_adjacent_to(node3, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node3))
    #
    #     self.assertTrue(dag.is_adjacent_to(node4, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node4))
    #
    #     self.assertTrue(dag.is_adjacent_to(node5, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node5))
    #
    #     self.assertTrue(dag.is_adjacent_to(node6, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node6))
    #
    #     self.assertTrue(dag.is_adjacent_to(node7, node2))
    #     self.assertTrue(dag.is_adjacent_to(node2, node7))

    # def test_is_ancestor_of(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node4, node1, -1, -1)
    #     edge4 = Edge(node5, node1, 6, 6)
    #     edge5 = Edge(node6, node1, 6, 6)
    #     edge6 = Edge(node6, node1, -1, 1)
    #     edge7 = Edge(node7, node1, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge6)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge7)
    #
    #     self.assertTrue(dag.is_ancestor_of(node1, node3))
    #     self.assertTrue(dag.is_ancestor_of(node2, node3))
    #     self.assertTrue(dag.is_ancestor_of(node3, node3))
    #     self.assertTrue(dag.is_ancestor_of(node6, node2))
    #
    #     self.assertFalse(dag.is_ancestor_of(node3, node1))
    #     self.assertFalse(dag.is_ancestor_of(node4, node1))
    #     self.assertFalse(dag.is_ancestor_of(node5, node1))
    #     self.assertFalse(dag.is_ancestor_of(node7, node1))

    # def test_is_child_of(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node3, node2, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     self.assertTrue(dag.is_child_of(node2, node1))
    #     self.assertTrue(dag.is_child_of(node2, node3))
    #     self.assertFalse(dag.is_child_of(node1, node2))
    #
    # def test_is_parent_of(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node3, node2, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     self.assertTrue(dag.is_parent_of(node1, node2))
    #     self.assertTrue(dag.is_parent_of(node3, node2))
    #     self.assertFalse(dag.is_parent_of(node2, node1))
    #
    # def test_is_descendant_of(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node4, node2, -1, 1)
    #     edge4 = Edge(node4, node2, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge3)
    #
    #     self.assertTrue(dag.is_descendant_of(node3, node1))
    #     self.assertTrue(dag.is_descendant_of(node3, node2))
    #     self.assertTrue(dag.is_descendant_of(node3, node3))
    #     self.assertTrue(dag.is_descendant_of(node2, node4))
    #
    #     self.assertFalse(dag.is_descendant_of(node1, node3))
    #
    # def test_is_proper_descendant_of(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #
    #     self.assertTrue(dag.is_proper_descendant_of(node3, node1))
    #     self.assertTrue(dag.is_proper_descendant_of(node3, node2))
    #
    #     self.assertFalse(dag.is_proper_descendant_of(node3, node3))
    #     self.assertFalse(dag.is_proper_descendant_of(node1, node3))
    #
    # def test_is_proper_ancestor_of(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #
    #     self.assertTrue(dag.is_proper_ancestor_of(node1, node3))
    #     self.assertTrue(dag.is_proper_ancestor_of(node2, node3))
    #
    #     self.assertFalse(dag.is_proper_ancestor_of(node3, node3))
    #     self.assertFalse(dag.is_proper_ancestor_of(node3, node1))

    # def test_get_edge(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node3, node2, -1, 1)
    #     edge3 = Edge(node3, node2, 6, 6)
    #     edge4 = Edge(node4, node2, 2, 2)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #
    #     outedge2 = dag.get_edge(node1, node2)
    #     outedge3 = dag.get_edge(node2, node1)
    #     outedge4 = dag.get_edge(node3, node2)
    #     outedge5 = dag.get_edge(node4, node2)
    #
    #     self.assertEqual(edge, outedge2)
    #     self.assertEqual(edge, outedge3)
    #     self.assertEqual(edge4, outedge5)
    #     self.assertEqual(None, outedge4)

    # def test_get_node_edges(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node1, node3, -1, -1)
    #     edge3 = Edge(node1, node4, -1, 1)
    #     edge4 = Edge(node1, node4, 6, 6)
    #     edge5 = Edge(node1, node5, 2, 2)
    #
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #
    #     edges = dag.get_node_edges(node1)
    #
    #     self.assertTrue(edge in edges)
    #     self.assertTrue(edge2 in edges)
    #     self.assertTrue(edge3 in edges)
    #     self.assertTrue(edge4 in edges)
    #     self.assertTrue(edge5 in edges)
    #
    #     self.assertTrue(len(edges) == 5)

    # def test_is_def_noncollider(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node1, node3, -1, 1)
    #     edge4 = Edge(node1, node4, 2, 1)
    #     edge5 = Edge(node5, node4, 2, 1)
    #     edge6 = Edge(node4, node6, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge6)
    #
    #     self.assertTrue(dag.is_def_noncollider(node1, node2, node3))
    #     self.assertFalse(dag.is_def_noncollider(node1, node3, node2))
    #
    #     self.assertTrue(dag.is_def_noncollider(node5, node4, node6))
    #     self.assertFalse(dag.is_def_noncollider(node1, node4, node5))
    #
    # def test_is_def_collider(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, 2, 1)
    #     edge2 = Edge(node2, node3, 2, 1)
    #     edge3 = Edge(node1, node3, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     self.assertFalse(dag.is_def_collider(node1, node2, node3))
    #     self.assertTrue(dag.is_def_collider(node1, node3, node2))
    #
    # def test_is_dconnected_to(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, 2, 1)
    #     edge2 = Edge(node3, node2, 2, 1)
    #     edge3 = Edge(node2, node4, 2, 1)
    #     edge4 = Edge(node5, node1, 2, 1)
    #     edge5 = Edge(node6, node5, 2, 1)
    #     edge6 = Edge(node2, node7, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge6)
    #
    #     self.assertTrue(dag.is_dconnected_to(node6, node1, []))
    #     self.assertFalse(dag.is_dconnected_to(node6, node3, []))
    #     self.assertTrue(dag.is_dconnected_to(node6, node3, [node2]))
    #     self.assertFalse(dag.is_dconnected_to(node6, node3, [node7]))
    #     self.assertFalse(dag.is_dconnected_to(node6, node3, [node2, node5]))
    #
    # def test_is_dseparated_from(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #     node5 = dag.get_node("x5")
    #     node6 = dag.get_node("x6")
    #     node7 = dag.get_node("x7")
    #
    #     edge = Edge(node1, node2, 2, 1)
    #     edge2 = Edge(node3, node2, 2, 1)
    #     edge3 = Edge(node2, node4, 2, 1)
    #     edge4 = Edge(node5, node1, 2, 1)
    #     edge5 = Edge(node6, node5, 2, 1)
    #     edge6 = Edge(node2, node7, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #     dag.add_edge(edge6)
    #
    #     self.assertFalse(dag.is_dseparated_from(node6, node1, []))
    #     self.assertTrue(dag.is_dseparated_from(node6, node3, []))
    #     self.assertFalse(dag.is_dseparated_from(node6, node3, [node2]))
    #     self.assertTrue(dag.is_dseparated_from(node6, node3, [node7]))
    #     self.assertTrue(dag.is_dseparated_from(node6, node3, [node2, node5]))

    # def test_is_directed_from_to(self):
    #     node_names = ["x1", "x2"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #
    #     dag.add_edge(edge)
    #
    #     self.assertTrue(dag.is_directed_from_to(node1, node2))
    #     self.assertFalse(dag.is_directed_from_to(node2, node1))
    #
    # def test_is_exogenous(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node1, node3, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     self.assertTrue(dag.is_exogenous(node1))
    #     self.assertFalse(dag.is_exogenous(node2))
    #
    # def test_get_nodes_into(self):
    #     node_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node1, node3, -1, 1)
    #     edge4 = Edge(node4, node2, 1, 2)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #
    #     arrow_nodes = dag.get_nodes_into(node2, Endpoint.ARROW)
    #     tail_nodes = dag.get_nodes_into(node2, Endpoint.TAIL)
    #     circle_nodes = dag.get_nodes_into(node2, Endpoint.CIRCLE)
    #
    #     self.assertTrue(node1 in arrow_nodes)
    #     self.assertTrue(node3 in tail_nodes)
    #     self.assertTrue(node4 in circle_nodes)
    #     self.assertEqual(len(arrow_nodes), 1)
    #     self.assertEqual(len(tail_nodes), 1)
    #     self.assertEqual(len(circle_nodes), 1)
    #
    # def test_get_nodes_out_of(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node1, node3, -1, 1)
    #     edge4 = Edge(node2, node4, 1, 2)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #
    #     arrow_nodes = dag.get_nodes_out_of(node2, Endpoint.ARROW)
    #     tail_nodes = dag.get_nodes_out_of(node2, Endpoint.TAIL)
    #     circle_nodes = dag.get_nodes_out_of(node2, Endpoint.CIRCLE)
    #
    #
    #     self.assertTrue(node3 in arrow_nodes)
    #     self.assertTrue(node1 in tail_nodes)
    #     self.assertTrue(node4 in circle_nodes)
    #     self.assertEqual(len(arrow_nodes), 1)
    #     self.assertEqual(len(tail_nodes), 1)
    #     self.assertEqual(len(circle_nodes), 1)

    # def test_remove_edge(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node2, node3, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     dag.remove_edge(edge2)
    #
    #     true_graph = np.array([[ 0, -1,  0], [ 1,  0, 6], [ 0,  6,  0]])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)
    #
    # def test_remove_connecting_edge(self):
    #     node_names = ["x1", "x2", "x3"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node2, node3, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     dag.remove_connecting_edge(node2, node3)
    #
    #     true_graph = np.array([(0, -1, 0), (1, 0, 5), (0, 7, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)

    # def test_remove_node(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node1, node4, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #
    #     dag.remove_node(node4)
    #
    #     true_graph = np.array([(0, -1, 0), (1, 0, 0), (0, 0, 0)])
    #
    #     np.testing.assert_array_equal(dag.graph, true_graph)

    # def test_subgraph(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node1, node4, -1, 1)
    #     edge3 = Edge(node2, node4, 2, 2)
    #     edge4 = Edge(node3, node4, -1, 1)
    #     edge5 = Edge(node3, node4, 6, 6)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #     dag.add_edge(edge4)
    #     dag.add_edge(edge5)
    #
    #     sub_nodes = [node1, node2, node3]
    #
    #     subgraph = dag.subgraph(sub_nodes)
    #
    #     true_graph = np.array([(0, -1, 0), (1, 0, 0), (0, 0, 0)])
    #
    #     np.testing.assert_array_equal(subgraph.graph, true_graph)
    #     self.assertTrue(subgraph.is_ancestor_of(node1, node2))

    # def test_get_causal_ordering(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = Dag(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node1, node2, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #
    #     order = dag.get_causal_ordering()
    #
    #     self.assertEqual(order[0], node1)
    #     self.assertEqual(order[1], node2)
    #     self.assertEqual(order[2], node3)
    #
    #     edge3 = Edge(node1, node4, -1, 1)
    #
    #     dag.add_edge(edge3)
    #
    #     order = dag.get_causal_ordering()
    #
    #     for node in order:
    #         print(str(node) + " ")
    #
    #     self.assertEqual(order[0], node1)
    #     self.assertEqual(order[3], node3)

    # def test_exists_trek(self):
    #     node_names = ["x1", "x2", "x3", "x4"]
    #     nodes = []
    #
    #     for name in node_names:
    #         node = GraphNode(name)
    #         nodes.append(node)
    #
    #     dag = GeneralGraph(nodes)
    #     node1 = dag.get_node("x1")
    #     node2 = dag.get_node("x2")
    #     node3 = dag.get_node("x3")
    #     node4 = dag.get_node("x4")
    #
    #     edge = Edge(node2, node1, -1, 1)
    #     edge2 = Edge(node2, node3, -1, 1)
    #     edge3 = Edge(node3, node4, 2, 1)
    #
    #     dag.add_edge(edge)
    #     dag.add_edge(edge2)
    #     dag.add_edge(edge3)
    #
    #     self.assertFalse(dag.exists_trek(node1, node4))

if __name__ == '__main__':
    unittest.main()