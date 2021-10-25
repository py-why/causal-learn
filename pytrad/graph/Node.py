#!/usr/bin/env python3

# Represents an object with a name, node type, and position that can serve as a
# node in a graph.
class Node:

    #  @return the name of the variable.
    def get_name(self):
        pass

    # set the name of the variable
    def set_name(self, name):
        pass

    # @return the node type of the variable
    def get_node_type(self):
        pass

    # set the node type of the variable
    def set_node_type(self, type):
        pass

    # @return the intervention type
    def get_node_variable_type(self):
        pass

    # set the type (domain, interventional status, interventional value) for
    # this node variable
    def set_node_variable_type(self, type):
        pass

    # @return the name of the node as its string representation
    def __str__(self):
        pass

    # @return the x coordinate of the center of the node
    def get_center_x(self):
        pass

    # sets the x coordinate of the center of the node
    def set_center_x(self, center_x):
        pass

    # @return the y coordinate of the center of the node
    def get_center_y(self):
        pass

    # sets the y coordinate of the center of the node
    def set_center_y(self, center_y):
        pass

    # sets the [x, y] coordinates of the center of the node
    def set_center(self, center_x, center_y):
        pass

    # @return a hashcode for this variable
    def __hash__(self):
        pass

    # @return true iff this variable is equal to the given variable
    def __eq__(self, other):
        pass

    # creates a new node of the same type as this one with the given name
    def like(self, name):
        pass

    def get_all_attributes(self):
        pass

    def get_attribute(self, key):
        pass

    def remove_attribute(self, key):
        pass

    def add_attribute(self, key, value):
        pass
