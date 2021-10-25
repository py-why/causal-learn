#!/usr/bin/env python3

# Implements a graph capable of storing edges of type N1 *-$ N2 where * and
# $ are endpoints of type Endpoint.

class Graph:

    # Adds a bidirected edges <-> to the graph.
    def add_bidirected_edge(self, node1, node2):
        raise NotImplementedError

    # Adds a directed edge --> to the graph.
    def add_directed_edge(self, node1, node2):
        raise NotImplementedError

    # Adds an undirected edge --- to the graph.
    def add_undirected_edge(self, node1, node2):
        raise NotImplementedError

    # Adds an nondirected edges o-o to the graph.
    def add_nondirected_edge(self, node1, node2):
        raise NotImplementedError

    # Adds a partially oriented edge o-> to the graph.
    def add_partially_oriented_edge(self, node1, node2):
        raise NotImplementedError

    # Adds the specified edge to the graph, provided it is not already in the
    # graph.
    def add_edge(self, edge):
        raise NotImplementedError

    # Adds a node to the graph. Precondition: The proposed name of the node
    # cannot already be used by any other node in the same graph.
    def add_node(self, node):
        raise NotImplementedError

    # Removes all nodes (and therefore all edges) from the graph.
    def clear(self):
        raise NotImplementedError

    # Determines whether this graph contains the given edge.
    #
    # Returns true iff the graph contain 'edge'.
    def contains_edge(self, edge):
        raise NotImplementedError

    # Determines whether this graph contains the given node.
    #
    # Returns true iff the graph contains 'node'.
    def contains_node(self, node):
        raise NotImplementedError

    # Returns true iff there is a directed cycle in the graph.
    def exists_directed_cycle(self):
        raise NotImplementedError

    # Returns true iff there is a directed path from node1 to node2 in the graph.
    def exists_directed_path_from_to(self, node1, node2):
        raise NotImplementedError

    # Returns true iff there is a path from node1 to node2 in the graph.
    def exists_undirected_path_from_to(self, node1, node2):
        raise NotImplementedError

    # A semi-directed path from A to B is an undirected path in which no
    # edge has an arrowhead pointing "back" towards A.
    #
    # Return true iff there is a semi-directed path from node1 to a node in nodes in the graph.
    def exists_semidirected_path_from_to(self, node1, nodes):
        raise NotImplementedError

    # Determines whether an inducing path exists between node1 and node2, given
    # a set O of observed nodes and a set sem of conditioned nodes.
    def exists_inducing_path(self, node1, node2):
        raise NotImplementedError

    # Returns true iff a trek exists between two nodes in the graph.  A trek
    # exists if there is a directed path between the two nodes or else, for
    # some third node in the graph, there is a path to each of the two nodes in
    # question.
    def exists_trek(self, node1, node2):
        raise NotImplementedError

    # Determines whether this graph is equal to some other graph, in the sense
    # that they contain the same nodes and the sets of edges defined over these
    # nodes in the two graphs are isomorphic typewise. That is, if node A and B
    # exist in both graphs, and if there are, e.g., three edges between A and B
    # in the first graph, two of which are directed edges and one of which is
    # an undirected edge, then in the second graph there must also be two
    # directed edges and one undirected edge between nodes A and B.
    def __eq__(self, other):
        raise NotImplementedError

    # Removes all edges from the graph and fully connects it using $-$ edges, where $ is the given endpoint.
    def fully_connect(self, endpoint):
        raise NotImplementedError

    # Reorients all edges in the graph with the given endpoint.
    def reorient_all_with(self, endpoint):
        raise NotImplementedError

    # Returns a mutable list of nodes adjacent to the given node.
    def get_adjacent_nodes(self, node):
        raise NotImplementedError

    # Returns a mutable list of ancestors for the given nodes.
    def get_ancestors(self, nodes):
        raise NotImplementedError

    # Returns a mutable list of children for a node.
    def get_children(self, node):
        raise NotImplementedError

    # Returns the connectivity of the graph.
    def get_connectivity(self):
        raise NotImplementedError

    # Returns a mutable list of descendants for the given nodes.
    def get_descendants(self, nodes):
        raise NotImplementedError

    # Returns the edge connecting node1 and node2, provided a unique such edge exists.
    def get_edge(self, node1, node2):
        raise NotImplementedError

    # Returns the directed edge from node1 to node2, if there is one.
    def get_directed_edge(self, node1, node2):
        raise NotImplementedError

    # Returns the list of edges connected to a particular node. No particular ordering of the edges in the list is guaranteed.
    def get_node_edges(self, node):
        raise NotImplementedError

    # Returns the edges connecting node1 and node2.
    def get_connecting_edges(self, node1, node2):
        raise NotImplementedError

    # Returns the list of edges in the graph. No particular ordering is guaranteed.
    def get_graph_edges(self):
        raise NotImplementedError

    # Returns the endpoint along the edge from node1 to node2, at the node2 end.
    def get_endpoint(self, node1, node2):
        raise NotImplementedError

    # Returns the number of arrow endpoints adjacent to the node.
    def get_indegree(self, node):
        raise NotImplementedError

    # Returns the number of null endpoints adjacent to the node.
    def get_outdegree(self, node):
        raise NotImplementedError

    # Returns the total number of edges into and out of the node.
    def get_degree(self, node):
        raise NotImplementedError

    # Returns the node with the given string name.  In case of accidental
    # duplicates, the first node encountered with the given name is returned.
    # In case no node exists with the given name, null is returned.
    def get_node(self, name):
        raise NotImplementedError

    # Returns the list of nodes for the graph.
    def get_nodes(self):
        raise NotImplementedError

    # Returns the names of the nodes, in the order of get_nodes.
    def get_node_names(self):
        raise NotImplementedError

    # Returns the number of edges in the entire graph.
    def get_num_edges(self):
        raise NotImplementedError

    # Returns the number of edges in the graph which are connected to a particular node.
    def get_num_connected_edges(self, node):
        raise NotImplementedError

    # Return the number of nodes in the graph.
    def get_num_nodes(self):
        raise NotImplementedError

    # Return the list of parents of a node.
    def get_parents(self, node):
        raise NotImplementedError

    # Return true iff node1 is adjacent to node2 in the graph.
    def is_adjacent_to(self, node1, node2):
        raise NotImplementedError

    # Return true iff node1 is an ancestor of node2.
    def is_ancestor_of(self, node1, node2):
        raise NotImplementedError

    # Return true iff node1 is a possible ancestor of node2.
    #
    # This is a low priority method and may not be implemented.
    def possible_ancestor(self, node1, node2):
        raise NotImplementedError

    # Return true iff node1 is a child of node2.
    def is_child_of(self, node1, node2):
        raise NotImplementedError

    # Returns true iff node1 is a parent of node2.
    def is_parent_of(self, node1, node2):
        raise NotImplementedError

    # Returns true iff node1 is a proper ancestor of node2.
    def is_proper_ancestor_of(self, node1, node2):
        raise NotImplementedError

    # Returns true iff node1 is a proper descendant of node2.
    def is_proper_descendant_of(self, node1, node2):
        raise NotImplementedError

    # Returns true iff node1 is a descendant of node2.
    def is_descendant_of(self, node1, node2):
        raise NotImplementedError

    # A node Y is a definite nondescendent of a node X just in case there is no
    # semi-directed path from X to Y.
    #
    # Returns true if node 2 is a definite nondecendent of node 1.
    #
    # This is a low priority method and may not be implemented
    def def_non_descendent(self, node1, node2):
        raise NotImplementedError

    # Returns true if node2 is a definite noncollider between node1 and node3.
    def is_def_noncollider(self, node1, node2, node3):
        raise NotImplementedError

    # Returns true if node2 is a definite collider between node1 and node3.
    def is_def_collider(self, node1, node2, node3):
        raise NotImplementedError

    # Returns true if node1 and node2 are d-connected on the set of nodes z.
    def is_dconnected_to(self, node1, node2, z):
        raise NotImplementedError

    # Returns true if node1 and node2 are d-separated on the set of nodes z.
    def is_dseparated_from(self, node1, node2, z):
        raise NotImplementedError

    # A path U is possibly-d-connecting if every definite collider on U
    # is a possible ancestor of a node in z and every definite non-collider is
    # not in z.
    #
    # Returns true iff node1 and node2 are possibly d-connected on z.
    #
    # This is a low priority method and may not be implemented.
    def poss_dconnected_to(self, node1, node2, z):
        raise NotImplementedError

    # Returns true if the graph is a pattern.
    def is_pattern(self):
        raise NotImplementedError

    # Sets whether the graph is a pattern.
    def set_pattern(self, pattern):
        raise NotImplementedError

    # Returns true if the graph is a PAG.
    def is_pag(self):
        raise NotImplementedError

    # Sets whether the graph is a PAG.
    def set_pag(self, pag):
        raise NotImplementedError

    # Returns true iff there is a single directed edge from node1 to node2.
    def is_directed_from_to(self, node1, node2):
        raise NotImplementedError

    # REturns true iff there is a single undirected edge between node1 and node2.
    def is_undirected_from_to(self, node1, node2):
        raise NotImplementedError

    # A directed edge A->B is definitely visible if there is a node C not
    # adjacent to B such that C*->A is in the PAG_of_the_true_DAG.
    #
    # Returns true iff the given edge is definitely visible.
    #
    # This is a low priority method and may not be implemented.
    def def_visible(self, edge):
        raise NotImplementedError

    # Returns true iff the given node is exogenous.
    def is_exogenous(self, node):
        raise NotImplementedError

    # Returns the nodes adjacent to the given node with the given proximal endpoint.
    def get_nodes_into(self, node, endpoint):
        raise NotImplementedError

    # Returns the nodes adjacent to the given node with the given distal endpoint.
    def get_nodes_out_of(self, node, endpoint):
        raise NotImplementedError

    # Removes the given edge from the graph.
    def remove_edge(self, edge):
        raise NotImplementedError

    # Removes the edge connecting the given two nodes, provided there is exactly one such edge.
    def remove_connecting_edge(self, node1, node2):
        raise NotImplementedError

    # Removes all edges connecting node A to node B.  In most cases, this will
    # remove at most one edge, but since multiple edges are permitted in some
    # graph implementations, the number will in some cases be greater than
    # one.
    def remove_connecting_edges(self, node1, node2):
        raise NotImplementedError

    # Iterates through the list and removes any permissible edges found.  The
    # order in which edges are removed is the order in which they are presented
    # in the iterator.
    def remove_edges(self, edges):
        raise NotImplementedError

    # Removes a node from the graph.
    def remove_node(self, node):
        raise NotImplementedError

    # Iterates through the list and removes any permissible nodes found.  The
    # order in which nodes are removed is the order in which they are presented
    # in the iterator.
    def remove_nodes(self, nodes):
        raise NotImplementedError

    # Sets the endpoint type at the 'to' end of the edge from 'from' to 'to' to
    # the given endpoint.  Note: NOT CONSTRAINT SAFE
    def set_endpoint(self, node1, node2, endpoint):
        raise NotImplementedError

    # Constructs and returns a subgraph consisting of a given subset of the
    # nodes of this graph together with the edges between them.
    def subgraph(self, nodes):
        raise NotImplementedError

    # Returns a string representation of the graph.
    def __str__(self):
        raise NotImplementedError

    # Transfers nodes and edges from one graph to another.  One way this is
    # used is to change graph types.  One constructs a new graph based on the
    # old graph, and this method is called to transfer the nodes and edges of
    # the old graph to the new graph.
    def transfer_nodes_and_edges(self, graph):
        raise NotImplementedError

    def transfer_attributes(self, graph):
        raise NotImplementedError

    # Returns the list of ambiguous triples associated with this graph. Triples <x, y, z> that no longer
    # lie along a path in the getModel graph are removed.
    def get_ambiguous_triples(self):
        raise NotImplementedError

    # Returns the set of underlines associated with this graph.
    def get_underlines(self):
        raise NotImplementedError

    # Returns the set of dotted underlines associated with this graph.
    def get_dotted_underlines(self):
        raise NotImplementedError

    # Returns true iff the triple <node1, node2, node3> is set as ambiguous.
    def is_ambiguous_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Returns true iff the triple <node1, node2, node3> is set as underlined.
    def is_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Returns true iff the triple <node1, node2, node3> is set as dotted underlined.
    def is_dotted_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Adds the triple <node1, node2, node3> as an ambiguous triple to the graph.
    def add_ambiguous_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Adds the triple <node1, node2, node3> as an underlined triple to the graph.
    def add_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Adds the triple <node1, node2, node3> as a dotted underlined triple to the graph.
    def add_dotted_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Removes the triple <node1, node2, node3> from the set of ambiguous triples.
    def remove_ambiguous_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Removes the triple <node1, node2, node3> from the set of underlined triples.
    def remove_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Removes the triple <node1, node2, node3> from the set of dotted underlined triples.
    def remove_dotted_underline_triple(self, node1, node2, node3):
        raise NotImplementedError

    # Sets the list of ambiguous triples to the triples in the given set.
    def set_ambiguous_triples(self, triples):
        raise NotImplementedError

    # Sets the list of underlined triples to the triples in the given set.
    def set_underline_triples(self, triples):
        raise NotImplementedError

    # Sets the list of dotted underlined triples to the triples in the given set.
    def set_dotted_underline_triples(self, triples):
        raise NotImplementedError

    # Returns a tier ordering for acyclic graphs.
    def get_causal_ordering(self):
        raise NotImplementedError

    # Returns true if the given node is parameterizable.
    def is_parameterizable(self, node):
        raise NotImplementedError

    # Returns true if this is a time lag model.
    def is_time_lag_model(self):
        raise NotImplementedError

    # Returns the nodes in the sepset of node1 and node2.
    def get_sepset(self, node1, node2):
        raise NotImplementedError

    # Sets the list of nodes for this graph.
    def set_nodes(self, nodes):
        raise NotImplementedError

    def get_all_attributes(self):
        raise NotImplementedError

    def get_attribute(self, key):
        raise NotImplementedError

    def remove_attribute(self, key):
        raise NotImplementedError

    def add_attribute(self, key, value):
        raise NotImplementedError
