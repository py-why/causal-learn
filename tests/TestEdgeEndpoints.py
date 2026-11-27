import unittest

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


class TestEdgeEndpoints(unittest.TestCase):
    def setUp(self):
        self.nodes = [GraphNode("A"), GraphNode("B")]

    def test_reapplying_arrow_preserves_other_endpoint(self):
        for endpoint in (Endpoint.TAIL, Endpoint.CIRCLE, Endpoint.ARROW):
            with self.subTest(endpoint=endpoint):
                edge = Edge(*self.nodes, endpoint, Endpoint.ARROW)
                edge.set_endpoint2(Endpoint.ARROW)
                self.assertEqual(edge.get_endpoint1(), endpoint)
                self.assertEqual(edge.get_numerical_endpoint1(), endpoint.value)
                self.assertEqual(edge.get_numerical_endpoint2(), Endpoint.ARROW.value)

    def test_remove_edge_after_reapplying_arrow(self):
        for endpoint in (Endpoint.TAIL, Endpoint.CIRCLE):
            with self.subTest(endpoint=endpoint):
                graph = GeneralGraph(self.nodes)
                edge = Edge(*self.nodes, endpoint, Endpoint.ARROW)
                graph.add_edge(edge)
                edge.set_endpoint2(Endpoint.ARROW)
                graph.remove_edge(edge)
                self.assertEqual(graph.get_num_edges(), 0)

    def test_remove_directed_edge_preserves_bidirected_edge(self):
        graph = GeneralGraph(self.nodes)
        directed = Edge(*self.nodes, Endpoint.TAIL, Endpoint.ARROW)
        bidirected = Edge(*self.nodes, Endpoint.ARROW, Endpoint.ARROW)
        graph.add_edge(directed)
        graph.add_edge(bidirected)
        directed.set_endpoint2(Endpoint.ARROW)
        graph.remove_edge(directed)
        self.assertFalse(graph.contains_edge(directed))
        self.assertTrue(graph.contains_edge(bidirected))
        self.assertEqual(graph.get_num_edges(), 1)

    def test_remove_edge_without_reapplying_arrow(self):
        for endpoint in (Endpoint.TAIL, Endpoint.CIRCLE, Endpoint.ARROW):
            with self.subTest(endpoint=endpoint):
                graph = GeneralGraph(self.nodes)
                edge = Edge(*self.nodes, endpoint, Endpoint.ARROW)
                graph.add_edge(edge)
                graph.remove_edge(edge)
                self.assertEqual(graph.get_num_edges(), 0)

    def test_setting_arrow_on_tail_or_circle_endpoint(self):
        for endpoint in (Endpoint.TAIL, Endpoint.CIRCLE):
            with self.subTest(endpoint=endpoint):
                edge = Edge(*self.nodes, Endpoint.TAIL, endpoint)
                edge.set_endpoint2(Endpoint.ARROW)
                expected = Edge(*self.nodes, Endpoint.TAIL, Endpoint.ARROW)
                self.assertEqual(edge, expected)
                self.assertEqual(edge.get_numerical_endpoint1(), Endpoint.TAIL.value)
                self.assertEqual(edge.get_numerical_endpoint2(), Endpoint.ARROW.value)


if __name__ == "__main__":
    unittest.main()
