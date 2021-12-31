#!/usr/bin/env python3

from enum import Enum

from causallearn.graph.Endpoint import Endpoint

# Represents an edge node1 *-# node2 where * and # are endpoints of type
# Endpoint--that is, Endpoint.TAIL, Endpoint.ARROW, or Endpoint.CIRCLE.
#
# Note that because speed is of the essence, and Edge cannot be compared to an
# object of any other type; this will throw an exception.


class Edge:
    class Property(Enum):
        dd = 1
        nl = 2
        pd = 3
        pl = 4

    def __init__(self, node1, node2, end1, end2):
        self.properties = []

        if node1 is None or node2 is None:
            raise TypeError('Nodes must not be of NoneType. node1 = ' + str(node1) + ' node2 = ' + str(node2))

        if end1 is None or end2 is None:
            raise TypeError(
                'Endpoints must not be of NoneType. endpoint1 = ' + str(end1) + ' endpoint2 = ' + str(end2))

        # assign nodes and endpoints; if the edge points left, flip it
        if self.pointing_left(end1, end2):
            self.node1 = node2
            self.node2 = node1
            self.endpoint1 = end2
            self.endpoint2 = end1
            self.numerical_endpoint_1 = end2.value
            self.numerical_endpoint_2 = end1.value
        else:
            self.node1 = node1
            self.node2 = node2
            self.endpoint1 = end1
            self.endpoint2 = end2
            self.numerical_endpoint_1 = end1.value
            self.numerical_endpoint_2 = end2.value

    # return the A node
    def get_node1(self):
        return self.node1

    # return the B node
    def get_node2(self):
        return self.node2

    # return the endpoint of the edge at the A node
    def get_endpoint1(self):
        return self.endpoint1

    # return the endpoint of the edge at the B node
    def get_endpoint2(self):
        return self.endpoint2

    # # set the endpoint of the edge at the A node
    # def set_endpoint1(self, endpoint):
    #     self.endpoint1 = endpoint
    #
    # # set the endpoint of the edge at the B node
    # def set_endpoint2(self, endpoint):
    #     self.endpoint2 = endpoint

    def get_numerical_endpoint1(self):
        return self.numerical_endpoint_1

    def get_numerical_endpoint2(self):
        return self.numerical_endpoint_2

    # set the endpoint of the edge at the A node
    def set_endpoint1(self, endpoint):
        self.endpoint1 = endpoint

        if self.numerical_endpoint_1 == 1 and self.numerical_endpoint_2 == 1:
            if endpoint is Endpoint.ARROW:
                pass
            else:
                if endpoint is Endpoint.TAIL:
                    self.numerical_endpoint_1 = -1
                    self.numerical_endpoint_2 = 1
                else:
                    if endpoint is Endpoint.CIRCLE:
                        self.numerical_endpoint_1 = 2
                        self.numerical_endpoint_2 = 1
        else:
            if endpoint is Endpoint.ARROW and self.numerical_endpoint_2 == 1:
                self.numerical_endpoint_1 = 1
                self.numerical_endpoint_2 = 1
            else:
                if endpoint is Endpoint.ARROW:
                    self.numerical_endpoint_1 = 1
                else:
                    if endpoint is Endpoint.TAIL:
                        self.numerical_endpoint_1 = -1
                    else:
                        if endpoint is Endpoint.CIRCLE:
                            self.numerical_endpoint_1 = 2

        if self.pointing_left(self.endpoint1, self.endpoint2):
            tempnode = self.node1
            self.node1 = self.node2
            self.node2 = tempnode

            tempend = self.endpoint1
            self.endpoint1 = self.endpoint2
            self.endpoint2 = tempend

            tempnum = self.numerical_endpoint_1
            self.numerical_endpoint_1 = self.numerical_endpoint_2
            self.numerical_endpoint_2 = tempnum

    def set_endpoint2(self, endpoint):
        self.endpoint2 = endpoint

        if self.numerical_endpoint_1 == 1 and self.numerical_endpoint_2 == 1:
            if endpoint is Endpoint.ARROW:
                pass
            else:
                if endpoint is Endpoint.TAIL:
                    self.numerical_endpoint_1 = 1
                    self.numerical_endpoint_2 = -1
                else:
                    if endpoint is Endpoint.CIRCLE:
                        self.numerical_endpoint_1 = 1
                        self.numerical_endpoint_2 = 2
        else:
            if endpoint is Endpoint.ARROW and self.numerical_endpoint_2 == 1:
                self.numerical_endpoint_1 = 1
                self.numerical_endpoint_2 = 1
            else:
                if endpoint is Endpoint.ARROW:
                    self.numerical_endpoint_2 = 1
                else:
                    if endpoint is Endpoint.TAIL:
                        self.numerical_endpoint_2 = -1
                    else:
                        if endpoint is Endpoint.CIRCLE:
                            self.numerical_endpoint_2 = 2

        if self.pointing_left(self.endpoint1, self.endpoint2):
            tempnode = self.node1
            self.node1 = self.node2
            self.node2 = tempnode

            tempend = self.endpoint1
            self.endpoint1 = self.endpoint2
            self.endpoint2 = tempend

            tempnum = self.numerical_endpoint_1
            self.numerical_endpoint_1 = self.numerical_endpoint_2
            self.numerical_endpoint_2 = tempnum

    # return the endpoint nearest to the given node; returns NoneType if the
    # given node is not along the edge
    def get_proximal_endpoint(self, node):
        if self.node1 is node:
            return self.endpoint1
        else:
            if self.node2 is node:
                return self.endpoint2
            else:
                return None

    # return the endpoint furthest from the given node; returns NoneType if the
    # given node is not along the edge
    def get_distal_endpoint(self, node):
        if self.node1 is node:
            return self.endpoint2
        else:
            if self.node2 is node:
                return self.endpoint1
            else:
                return None

    # traverses the edge in an undirected fashion: given one node along the
    # edge, returns the node at the opposite end of the edge
    def get_distal_node(self, node):
        if self.node1 is node:
            return self.node2
        else:
            if self.node2 is node:
                return self.node1
            else:
                return None

    def points_toward(self, node):

        proximal = self.get_proximal_endpoint(node)
        distal = self.get_distal_endpoint(node)
        return proximal == Endpoint.ARROW and (distal == Endpoint.TAIL or distal == Endpoint.CIRCLE)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            raise TypeError("Not an edge")

        if self.endpoint1 == other.endpoint1 and self.endpoint2 == other.endpoint2 and self.node1 == other.node1 and self.node2 == other.node2:
            return True
        else:
            return False

    def __lt__(self, other):
        return self.node1 < other.node1 or self.node2 < other.node2

    def __str__(self):
        node1 = self.get_node1()
        node2 = self.get_node2()

        endpoint1 = self.get_endpoint1()
        endpoint2 = self.get_endpoint2()

        edge_string = node1.get_name() + " "

        if endpoint1 is Endpoint.TAIL:
            edge_string = edge_string + "-"
        else:
            if endpoint1 is Endpoint.ARROW:
                edge_string = edge_string + "<"
            else:
                edge_string = edge_string + "o"

        edge_string = edge_string + "-"

        if endpoint2 is Endpoint.TAIL:
            edge_string = edge_string + "-"
        else:
            if endpoint2 is Endpoint.ARROW:
                edge_string = edge_string + ">"
            else:
                edge_string = edge_string + "o"

        edge_string = edge_string + " " + node2.get_name()
        return edge_string

    #
    # Helper Methods
    #

    # returns True if the edge is pointing "left"
    def pointing_left(self, endpoint1, endpoint2):
        return endpoint1 == Endpoint.ARROW and (endpoint2 == Endpoint.TAIL or endpoint2 == Endpoint.CIRCLE)
