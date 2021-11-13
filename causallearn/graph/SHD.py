from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph


class SHD:
    __SHD = 0

    def __init__(self, truth: Graph, est: Graph):
        nodes = truth.get_nodes()
        nodes_name = [node.get_name() for node in nodes]
        self.__SHD = 0

        # Assumes the the list of nodes for the two graphs are the same.
        for i in list(range(0, len(nodes))):
            for j in list(range(i + 1, len(nodes))):
                if truth.get_edge(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])) and (
                        not est.get_edge(est.get_node(nodes_name[i]), est.get_node(nodes_name[j]))):
                    self.__SHD += 1
                if (not truth.get_edge(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j]))) and est.get_edge(
                        est.get_node(nodes_name[i]), est.get_node(nodes_name[j])):
                    self.__SHD += 1

        for i in list(range(0, len(nodes))):
            for j in list(range(0, len(nodes))):
                if not truth.get_edge(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])):
                    continue
                if not est.get_edge(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])):
                    continue
                if truth.get_endpoint(truth.get_node(nodes_name[i]),
                                      truth.get_node(nodes_name[j])) == Endpoint.ARROW and est.get_endpoint(
                    est.get_node(nodes_name[j]), est.get_node(nodes_name[i])) == Endpoint.ARROW:
                    self.__SHD += 1

    def get_shd(self):
        return self.__SHD
