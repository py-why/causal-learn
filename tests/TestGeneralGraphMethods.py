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
