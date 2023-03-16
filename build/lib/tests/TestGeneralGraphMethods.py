#!/usr/bin/env python3
import unittest

import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


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

    def test_is_ancestor_of(self):
        nodes = [GraphNode(str(i)) for i in range(0, 5)]
        graph = GeneralGraph(nodes)
        no_of_var = len(nodes)
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))

        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                edge = graph.get_edge(nodes[i], nodes[j])
                graph.remove_edge(edge)

        graph.add_edge(Edge(nodes[0], nodes[3], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[1], nodes[3], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[3], nodes[2], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[3], nodes[4], Endpoint.TAIL, Endpoint.ARROW))


        assert graph.is_ancestor_of(nodes[3], nodes[2])
        assert not graph.is_ancestor_of(nodes[2], nodes[3])