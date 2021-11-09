#!/usr/bin/env python3

# Implements a basic node in a graph--that is, a node that is not itself a variable.
from causallearn.graph.Node import Node
from causallearn.graph.NodeType import NodeType


class GraphNode(Node):

    def __init__(self, name):
        self.name = name
        self.node_type = NodeType.MEASURED
        self.center_x = -1
        self.center_y = -1
        self.attributes = {}

    #  @return the name of the variable.
    def get_name(self):
        return self.name

    # @return the node type
    def get_node_type(self):
        return self.node_type

    # @return the x coordinate of the center of the node
    def get_center_x(self):
        return self.center_x

    # @return the y coordinate of the center of the node
    def get_center_y(self):
        return self.center_y

    # sets the name of the node
    def set_name(self, name):
        if name == None:
            raise TypeError('Name cannot be of NoneType')
        self.name = name

    # sets the node type
    def set_node_type(self, type):
        if type == None:
            raise TypeError('Node cannot be of NoneType')
        self.node_type = type

    # sets the x coordinate of the center of the node
    def set_center_x(self, center_x):
        self.center_x = center_x

    # sets the y coordinate of the center of the node
    def set_center_y(self, center_y):
        self.center_y = center_y

    # sets the (x, y) coordinates of the center of the node
    def set_center(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y

    # @return the name of the node as its string representation
    def __str__(self):
        return self.name

    # Two continuous variables are equal if they have the same name and the same
    # missing value marker.
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.name == other.get_name()

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def like(self, name):
        node = GraphNode(name)
        node.set_node_type(self.get_node_type())

    def get_all_attributes(self):
        return self.attributes

    def get_attribute(self, key):
        return self.attributes[key]

    def __getitem__(self, key):
        return self.get_attribute(key)

    def remove_attribute(self, key):
        self.attributes.pop(key)

    def __delitem__(self, key):
        self.remove_attribute(key)

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def __setitem__(self, key, value):
        self.add_attribute(key, value)
