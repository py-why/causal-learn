#!/usr/bin/env python3
from itertools import combinations

import networkx as nx
import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Graph import Graph
from causallearn.utils.GraphUtils import GraphUtils


# Represents a directed acyclic graph--that is, a graph containing only
# directed edges, with no cycles--using a matrix. Variables are permitted to be either measured
# or latent, with at most one edge per node pair, and no edges to self.
class Dag(Graph):

    def __init__(self, nodes):

        # for node in nodes:
        #     if not isinstance(node, type(GraphNode)):
        #         raise TypeError("Graphs must be instantiated with a list of GraphNodes")

        self.nodes = nodes
        self.num_vars = len(nodes)

        node_map = {}

        for i in range(self.num_vars):
            node = nodes[i]
            node_map[node] = i

        self.node_map = node_map

        self.graph = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        self.dpath = np.zeros((self.num_vars, self.num_vars), np.dtype(int))

        self.reconstitute_dpath([])

        self.ambiguous_triples = []
        self.underline_triples = []
        self.dotted_underline_triples = []

        self.attributes = {}

    ### Helper Functions ###

    def adjust_dpath(self, i, j):
        dpath = self.dpath
        dpath[j, i] = 1

        for k in range(self.num_vars):
            if dpath[i, k] == 1:
                dpath[j, k] = 1

            if dpath[k, j] == 1:
                dpath[k, i] = 1

        self.dpath = dpath

    def reconstitute_dpath(self, edges):

        for i in range(self.num_vars):
            self.adjust_dpath(i, i)

        while len(edges) > 0:
            edge = edges.pop()
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            self.adjust_dpath(i, j)

    def collect_ancestors(self, node, ancestors):
        if node in ancestors:
            return

        ancestors.append(node)
        parents = self.get_parents(node)

        if parents:
            for parent in parents:
                self.collect_ancestors(parent, ancestors)

    ### Public Functions ###

    def get_adjacency_matrix(self):
        return self.graph

    def get_node_map(self):
        return self.node_map

    # Adds a directed edge --> to the graph.
    def add_directed_edge(self, node1, node2):
        i = self.node_map[node1]
        j = self.node_map[node2]
        self.graph[j, i] = 1
        self.graph[i, j] = -1

        self.adjust_dpath(i, j)

    # Adds the specified edge to the graph, provided it is not already in the
    # graph.
    def add_edge(self, edge):

        if edge.get_endpoint1().name == "TAIL" and edge.get_endpoint2().name == "ARROW":
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            self.graph[j, i] = 1
            self.graph[i, j] = -1
            self.adjust_dpath(i, j)
            return True
        else:
            return False

    # Adds a node to the graph. Precondition: The proposed name of the node
    # cannot already be used by any other node in the same graph.
    def add_node(self, node):

        # ADD A CHECK FOR WHETHER THE NODE NAME ALREADY EXISTS
        nodes = self.nodes
        nodes.append(node)
        self.nodes = nodes

        self.num_vars = self.num_vars + 1

        self.node_map[node] = self.num_vars - 1

        row = np.zeros(self.num_vars - 1)
        graph = np.vstack((self.graph, row))
        dpath = np.vstack((self.dpath, row))

        col = np.zeros(self.num_vars)
        graph = np.column_stack((graph, col))
        dpath = np.column_stack((dpath, col))

        self.graph = graph
        self.dpath = dpath

    # Removes all nodes (and therefore all edges) from the graph.
    def clear(self):
        raise NotImplementedError

    # Determines whether this graph contains the given edge.
    #
    # Returns true iff the graph contain 'edge'.
    def contains_edge(self, edge):
        if edge.get_endpoint1() != Endpoint.TAIL or edge.get_endpoint2() != Endpoint.ARROW:
            return False
        else:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            if self.graph[j, i] == 1:
                return True
            else:
                return False

    # Determines whether this graph contains the given node.
    #
    # Returns true iff the graph contains 'node'.
    def contains_node(self, node):
        node_list = self.nodes
        return node in node_list

    # Returns true iff there is a directed cycle in the graph.
    def exists_directed_cycle(self):
        return False

        # Returns true iff a trek exists between two nodes in the graph.  A trek
        # exists if there is a directed path between the two nodes or else, for
        # some third node in the graph, there is a path to each of the two nodes in
        # question.

    def exists_trek(self, node1, node2):

        for node in self.nodes:
            if self.is_ancestor_of(node, node1) and self.is_ancestor_of(node, node2):
                return True

        return False

    # Determines whether this graph is equal to some other graph, in the sense
    # that they contain the same nodes and the sets of edges defined over these
    # nodes in the two graphs are isomorphic typewise. That is, if node A and B
    # exist in both graphs, and if there are, e.g., three edges between A and B
    # in the first graph, two of which are directed edges and one of which is
    # an undirected edge, then in the second graph there must also be two
    # directed edges and one undirected edge between nodes A and B.
    def __eq__(self, other):

        if isinstance(other, Dag):
            sorted_list = self.nodes.sort()
            if sorted_list == other.nodes.sort() and np.array_equal(self.graph, other.graph):
                return True
            else:
                return False
        else:
            return False

    # Returns a mutable list of nodes adjacent to the given node.
    def get_adjacent_nodes(self, node):

        j = self.node_map[node]
        adj_list = []

        for i in range(len(self.nodes)):
            if self.graph[j, i] == 1 or self.graph[i, j] == 1:
                node2 = self.nodes[i]
                adj_list.append(node2)

        return adj_list

    # Return the list of parents of a node.
    def get_parents(self, node):

        i = self.node_map[node]
        parents = []

        for j in range(len(self.nodes)):
            if self.graph[i, j] == 1:
                node2 = self.nodes[j]
                parents.append(node2)

        return parents

    # Returns a mutable list of ancestors for the given nodes.
    def get_ancestors(self, nodes):

        if isinstance(nodes, list):
            pass
        else:
            raise TypeError("Must be a list of nodes")

        ancestors = []

        for node in nodes:
            self.collect_ancestors(node, ancestors)

        return ancestors

    # Returns a mutable list of children for a node.
    def get_children(self, node):

        i = self.node_map[node]
        children = []

        print(self.nodes)
        print(self.num_vars)

        for j in range(len(self.nodes)):
            if self.graph[j, i] == 1:
                node2 = self.nodes[j]
                children.append(node2)

        return children

    # Returns the number of arrow endpoints adjacent to the node.
    def get_indegree(self, node):

        i = self.node_map[node]

        indegree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == 1:
                indegree = indegree + 1

        return indegree

    # Returns the number of null endpoints adjacent to the node.
    def get_outdegree(self, node):

        i = self.node_map[node]

        outdegree = 0

        for j in range(self.num_vars):
            if self.graph[j, i] == 1:
                outdegree = outdegree + 1

        return outdegree

    # Returns the total number of edges into and out of the node.
    def get_degree(self, node):
        return self.get_indegree(node) + self.get_outdegree(node)

    # Returns the node with the given string name.  In case of accidental
    # duplicates, the first node encountered with the given name is returned.
    # In case no node exists with the given name, None is returned.
    def get_node(self, name):

        for node in self.nodes:
            if node.get_name() == name:
                return node

        return None

    # Returns the list of nodes for the graph.
    def get_nodes(self):
        return self.nodes

    # Returns the names of the nodes, in the order of get_nodes.
    def get_node_names(self):

        node_names = []

        for node in self.nodes:
            node_names.append(node.get_name)

        return node_names

    # Returns the number of edges in the entire graph.
    def get_num_edges(self):

        edges = 0

        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if self.graph[j, i] != 0:
                    edges = edges + 1

        return edges

    # Returns the number of edges in the graph which are connected to a particular node.
    def get_num_connected_edges(self, node):

        i = self.node_map[node]

        edges = 0

        for j in range(self.num_vars):

            if self.graph[j, i] == 1 or self.graph[i, j] == 1:
                edges = edges + 1

        return edges

    # Return the number of nodes in the graph.
    def get_num_nodes(self):
        return self.num_vars

    # Return true iff node1 is adjacent to node2 in the graph.
    def is_adjacent_to(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[j, i] != 0

    # Return true iff node1 is an ancestor of node2.
    def is_ancestor_of(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.dpath[j, i] == 1

    # Return true iff node1 is a child of node2.
    def is_child_of(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[i, j] == 1

    # Returns true iff node1 is a parent of node2.
    def is_parent_of(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[j, i] == 1

    # Returns true iff node1 is a proper ancestor of node2.
    def is_proper_ancestor_of(self, node1, node2):
        return (self.is_ancestor_of(node1, node2) and not (node1 == node2))

    # Returns true iff node1 is a proper descendant of node2.
    def is_proper_descendant_of(self, node1, node2):
        return (self.is_descendant_of(node1, node2) and not (node1 == node2))

    # Returns true iff node1 is a descendant of node2.
    def is_descendant_of(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.dpath[i, j] == 1

    # Returns the edge connecting node1 and node2, provided a unique such edge exists.
    def get_edge(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 == 0 and end_2 == 0:
            return None

        edge = Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))
        return edge

    # Returns the directed edge from node1 to node2, if there is one.
    def get_directed_edge(self, node1, node2):
        return self.get_edge(node1, node2)

    # Returns the list of edges connected to a particular node. No particular ordering of the edges in the list is guaranteed.
    def get_node_edges(self, node):

        i = self.node_map[node]
        edges = []

        for j in range(self.num_vars):
            if self.graph[j, i] != 0:
                node2 = self.nodes[j]
                edges.append(self.get_edge(node, node2))

        return edges

    def get_graph_edges(self):

        edges = []

        for i in range(self.num_vars):
            node = self.nodes[i]
            for j in range(i + 1, self.num_vars):
                if self.graph[j, i] != 0:
                    node2 = self.nodes[j]
                    edges.append(self.get_edge(node, node2))

        return edges

    # Returns true if node2 is a definite noncollider between node1 and node3.
    def is_def_noncollider(self, node1, node2, node3):

        edges = self.get_node_edges(node2)

        for edge in edges:
            is_node1 = edge.get_distal_node(node2) == node1
            is_node3 = edge.get_distal_node(node2) == node3

            if is_node1 and edge.points_toward(node1):
                return True
            if is_node3 and edge.points_toward(node3):
                return True

        return False

    # Returns true if node2 is a definite collider between node1 and node3.
    def is_def_collider(self, node1, node2, node3):

        edge1 = self.get_edge(node1, node2)
        edge2 = self.get_edge(node2, node3)

        if edge1 is None or edge2 is None:
            return False

        return str(edge1.get_proximal_endpoint(node2)) == "ARROW" and str(edge2.get_proximal_endpoint(node2)) == "ARROW"

    # Returns true if node1 and node2 are d-connected on the set of nodes z.
    def is_dconnected_to(self, node1, node2, z):
        utils = GraphUtils()
        return utils.is_dconnected_to(node1, node2, z, self)

    # Returns true if node1 and node2 are d-separated on the set of nodes z.
    def is_dseparated_from(self, node1, node2, z):
        return not self.is_dconnected_to(node1, node2, z)

    # Returns true if the graph is a pattern.
    def is_pattern(self):
        return False

    # Returns true if the graph is a PAG.
    def is_pag(self):
        return False

    # Returns true iff there is a single directed edge from node1 to node2.
    def is_directed_from_to(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[j, i] == 1

    # REturns true iff there is a single undirected edge between node1 and node2.
    def is_undirected_from_to(self, node1, node2):
        return False

    # Returns true iff the given node is exogenous.
    def is_exogenous(self, node):
        return self.get_indegree(node) == 0

    # Returns the nodes adjacent to the given node with the given proximal endpoint.
    def get_nodes_into(self, node, endpoint):
        if not (str(endpoint) == "ARROW" or str(endpoint) == "TAIL"):
            return []

        i = self.node_map[node]
        nodes = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[i, j] == 1:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            for j in range(self.num_vars):
                if self.graph[j, i] == 1:
                    node2 = self.nodes[j]
                    nodes.append(node2)

        return nodes

    # Returns the nodes adjacent to the given node with the given distal endpoint.
    def get_nodes_out_of(self, node, endpoint):
        if not (str(endpoint) == "ARROW" or str(endpoint) == "TAIL"):
            return []

        i = self.node_map[node]
        nodes = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[j, i] == 1:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            for j in range(self.num_vars):
                if self.graph[i, j] == 1:
                    node2 = self.nodes[j]
                    nodes.append(node2)

        return nodes

    # Removes the given edge from the graph.
    def remove_edge(self, edge):

        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = self.node_map[node1]
        j = self.node_map[node2]

        self.graph[j, i] = 0
        self.graph[i, j] = 0

    # Removes the edge connecting the given two nodes, provided there is exactly one such edge.
    def remove_connecting_edge(self, node1, node2):

        i = self.node_map[node1]
        j = self.node_map[node2]

        self.graph[j, i] = 0
        self.graph[i, j] = 0

    # Removes all edges connecting node A to node B.  In most cases, this will
    # remove at most one edge, but since multiple edges are permitted in some
    # graph implementations, the number will in some cases be greater than
    # one.
    def remove_connecting_edges(self, node1, node2):

        self.remove_connecting_edge(node1, node2)

    # Iterates through the list and removes any permissible edges found.  The
    # order in which edges are removed is the order in which they are presented
    # in the iterator.
    def remove_edges(self, edges):

        for edge in edges:
            self.remove_edge(edge)

    # Removes a node from the graph.
    def remove_node(self, node):

        i = self.node_map[node]

        graph = self.graph

        graph = np.delete(graph, (i), axis=0)
        graph = np.delete(graph, (i), axis=1)

        self.graph = graph

        nodes = self.nodes
        nodes.remove(node)
        self.nodes = nodes

        node_map = self.node_map
        node_map.pop(node)
        self.node_map = node_map

    # Iterates through the list and removes any permissible nodes found.  The
    # order in which nodes are removed is the order in which they are presented
    # in the iterator.
    def remove_nodes(self, nodes):

        for node in nodes:
            self.remove_node(node)

    # Constructs and returns a subgraph consisting of a given subset of the
    # nodes of this graph together with the edges between them.
    def subgraph(self, nodes):

        subgraph = Dag(nodes)

        graph = self.graph

        for i in range(self.num_vars):
            if not (self.nodes[i] in nodes):
                graph = np.delete(graph, (i), axis=0)

        for i in range(self.num_vars):
            if not (self.nodes[i] in nodes):
                graph = np.delete(graph, (i), axis=1)

        subgraph.graph = graph
        subgraph.reconstitute_dpath(subgraph.get_graph_edges())

        return subgraph

    # Returns a string representation of the graph.
    def __str__(self):
        utils = GraphUtils()
        return utils.graph_string(self)

    # Transfers nodes and edges from one graph to another.  One way this is
    # used is to change graph types.  One constructs a new graph based on the
    # old graph, and this method is called to transfer the nodes and edges of
    # the old graph to the new graph.
    def transfer_nodes_and_edges(self, graph):

        for node in graph.nodes:
            self.add_node(node)

        for edge in graph.get_graph_edges():
            self.add_edge(edge)

    def transfer_attributes(self, graph):
        graph.attributes = self.attributes

    # Returns the list of ambiguous triples associated with this graph. Triples <x, y, z> that no longer
    # lie along a path in the getModel graph are removed.
    def get_ambiguous_triples(self):
        return self.ambiguous_triples

    # Returns the set of underlines associated with this graph.
    def get_underlines(self):
        return self.underline_triples

    # Returns the set of dotted underlines associated with this graph.
    def get_dotted_underlines(self):
        return self.dotted_underline_triples

    # Returns true iff the triple <node1, node2, node3> is set as ambiguous.
    def is_ambiguous_triple(self, node1, node2, node3):
        return (node1, node2, node3) in self.ambiguous_triples

    # Returns true iff the triple <node1, node2, node3> is set as underlined.
    def is_underline_triple(self, node1, node2, node3):
        return (node1, node2, node3) in self.underline_triples

    # Returns true iff the triple <node1, node2, node3> is set as dotted underlined.
    def is_dotted_underline_triple(self, node1, node2, node3):
        return (node1, node2, node3) in self.dotted_underline_triples

    # Adds the triple <node1, node2, node3> as an ambiguous triple to the graph.
    def add_ambiguous_triple(self, node1, node2, node3):
        self.ambiguous_triples.append((node1, node2, node3))

    # Adds the triple <node1, node2, node3> as an underlined triple to the graph.
    def add_underline_triple(self, node1, node2, node3):
        self.underline_triples.append((node1, node2, node3))

    # Adds the triple <node1, node2, node3> as a dotted underlined triple to the graph.
    def add_dotted_underline_triple(self, node1, node2, node3):
        self.dotted_underline_triples.append((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of ambiguous triples.
    def remove_ambiguous_triple(self, node1, node2, node3):
        self.ambiguous_triples.remove((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of underlined triples.
    def remove_underline_triple(self, node1, node2, node3):
        self.underline_triples.remove((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of dotted underlined triples.
    def remove_dotted_underline_triple(self, node1, node2, node3):
        self.dotted_underline_triples.remove((node1, node2, node3))

    # Sets the list of ambiguous triples to the triples in the given set.
    def set_ambiguous_triples(self, triples):
        self.ambiguous_triples = triples

    # Sets the list of underlined triples to the triples in the given set.
    def set_underline_triples(self, triples):
        self.underline_triples = triples

    # Sets the list of dotted underlined triples to the triples in the given set.
    def set_dotted_underline_triples(self, triples):
        self.dotted_underline_triples = triples

    # Returns a tier ordering for acyclic graphs.
    def get_causal_ordering(self):
        utils = GraphUtils()
        return utils.get_causal_order(self)

    # Returns true if the given node is parameterizable.
    def is_parameterizable(self, node):
        return True

    # Returns true if this is a time lag model.
    def is_time_lag_model(self):
        return False

    # Returns the nodes in the sepset of node1 and node2.
    def get_sepset(self, node1, node2):
        return GraphUtils.get_sepset(node1, node2, self)

    # Sets the list of nodes for this graph.
    def set_nodes(self, nodes):
        if len(nodes) != self.num_vars:
            raise ValueError("Sorry, there is a mismatch in the number of variables you are trying to set.")

        self.nodes = nodes

    def get_all_attributes(self):
        return self.attributes

    def get_attribute(self, key):
        return self.attributes[key]

    def remove_attribute(self, key):
        self.attributes.pop[key]

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def is_dag(B):
        """Check whether B corresponds to a DAG.

        Args:
            B (numpy.ndarray): [d, d] binary or weighted matrix.
        """
        return nx.is_directed_acyclic_graph(nx.DiGraph(B))

    def dag2pag(self, observable):

        G = GeneralGraph(self.get_nodes())
        for u, v in combinations(self.get_nodes(), 2):
            edge = self.get_edge(u, v)
            if edge:
                G.add_edge(edge)

        for u in self.get_nodes():
            if u in observable:
                continue
            for parent in self.get_parents(u):
                for child in self.get_children(u):
                    edge = Edge(parent, child, Endpoint.TAIL, Endpoint.ARROW)
                    mod_endpoint(edge, parent, Endpoint.TAIL)
                    mod_endpoint(edge, child, Endpoint.ARROW)
                    G.add_edge(edge)
            for x, y in combinations(self.get_children(u), 2):
                edge = Edge(x, y, Endpoint.ARROW, Endpoint.ARROW)
                mod_endpoint(edge, x, Endpoint.ARROW)
                mod_endpoint(edge, y, Endpoint.ARROW)
                G.add_edge(edge)

        pag = GeneralGraph(observable)
        for u, v in combinations(observable, 2):
            edge = G.get_edge(u, v)
            pag.add_edge(edge)

        return pag


def is_fully_directed(edge):
    if edge:
        if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            return True
    return False


def is_endpoint(edge, z, end):
    if edge.get_node1() == z:
        if edge.get_endpoint1() == end:
            return True
        else:
            return False
    elif edge.get_node2() == z:
        if edge.get_endpoint2() == end:
            return True
        else:
            return False
    else:
        raise ValueError("z not in edge")


def mod_endpoint(edge, z, end):
    if edge.get_node1() == z:
        edge.set_endpoint1(end)
    elif edge.get_node2() == z:
        edge.set_endpoint2(end)
    else:
        raise ValueError("z not in edge")
