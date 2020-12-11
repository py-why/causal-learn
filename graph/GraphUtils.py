#!/usr/bin/env python3
from graph.Graph import Graph
from graph.Endpoint import Endpoint
from graph.Edge import Edge
import numpy as np
from collections import deque
from graph.NodeType import NodeType

class GraphUtils:

    def __init__(self):
        pass

    # Returns true if node1 is d-connected to node2 on the set of nodes z.
    ### CURRENTLY THIS DOES NOT IMPLEMENT UNDERLINE TRIPLE EXCEPTIONS ###
    def is_dconnected_to(self, node1, node2, z, graph):
        if node1 == node2:
            return True

        edgenode_deque = deque([])

        for edge in graph.get_node_edges(node1):

            if edge.get_distal_node(node1) == (node2):
                return True

            edgenode_deque.append((edge, node1))


        while len(edgenode_deque) > 0:
            edge, node_a = edgenode_deque.pop()

            node_b = edge.get_distal_node(node_a)

            for edge2 in graph.get_node_edges(node_b):
                node_c = edge2.get_distal_node(node_b)

                if node_c == node_a:
                    continue

                if self.reachable(edge, edge2, node_a, z, graph):
                    if node_c == node2:
                        return True
                    else:
                        edgenode_deque.append((edge2, node_b))

        return False

    def edge_string(self, edge):

        node1 = edge.get_node1()
        node2 = edge.get_node2()

        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

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

        edge_string = edge_string + " " + node2
        return edge_string

    def graph_string(self, graph):

        nodes = graph.get_nodes()
        edges = graph.get_graph_edges()

        graph_string = "Graph Nodes:\n"

        for i in range(len(nodes)-1):
            node = nodes[i]
            graph_string = graph_string + node.get_name() + ";"

        if len(nodes) > 0:
            graph_string = graph_string + nodes[-1].get_name()

        graph_string = graph_string + "\n\nGraph Edges:\n"

        count = 0
        for edge in edges:
            count = count + 1
            graph_string = graph_string + str(count) + ". " + str(edge) + "\n"

        return graph_string

    # Helper method. Determines if two edges do or do not form a block for d-separation, conditional on a set of nodes z
    # starting from a node a
    def reachable(self, edge1, edge2, node_a, z, graph):

        node_b = edge1.get_distal_node(node_a)

        collider = str(edge1.get_proximal_endpoint(node_b)) == "ARROW" and str(edge2.get_proximal_endpoint(node_b)) == "ARROW"

        if  (not collider) and not (node_b in z) :
            return True

        ancestor = self.is_ancestor(node_b, z, graph)

        return collider and ancestor

    # Helper method. Determines if a given node is an ancestor of any node in a set of nodes z.
    def is_ancestor(self, node, z, graph):
        if node in z:
            return True

        nodedeque = deque([])

        for node_z in z:
            nodedeque.append(node_z)

        while len(nodedeque) > 0:
            node_t = nodedeque.pop()
            if node_t == node:
                return True

            for node_c in graph.get_parents(node_t):
                if not (node_c in nodedeque):
                    nodedeque.append(node_c)

    def get_sepset(self, x, y, graph):

        sepset = self.get_sepset_visit(x, y, graph)
        if sepset is None:
            sepset = self.get_sepset_visit(y, x, graph)

        return sepset

    def get_sepset_visit(self, x, y, graph):

        if x == y:
            return None

        z = []

        while True:
            _z = z.copy()
            path = [x]
            colliders = []

            for b in x.get_adjacent_nodes():
                if self.sepset_path_found(x, b, y, path, z, graph, colliders):
                    return None

            z.sort()
            _z.sort()
            if z != _z:
                break

        return z

    def sepset_path_found(self, a, b, y, path, z, graph, colliders):

        if b == y:
            return True

        if b in path:
            return False

        path.append(b)

        if b.get_node_type == NodeType.LATENT or b in z:
            pass_nodes = self.get_pass_nodes(a, b, z, graph, None)

            for c in pass_nodes:
                if self.sepset_path_found(b, c, y, path, z, graph, colliders):
                    path.remove(b)
                    return True

            path.remove(b)
            return False
        else:
            found1 = False
            colliders1 = []
            pass_nodes1 = self.get_pass_nodes(a, b, z, graph, colliders1)

            for c in pass_nodes1:
                if self.sepset_path_found(b, c, y, path, z, graph, colliders1):
                    found1 = True
                    break

            if not found1:
                path.remove(b)
                colliders.extend(colliders1)
                return False

            z.append(b)
            found2 = False
            colliders2 = []
            pass_nodes2 = self.get_pass_nodes(a, b, z, graph, None)

            for c in pass_nodes2:
                if self.sepset_path_found(b, c, y, z, graph, colliders2):
                    found2 = True
                    break

            if not found2:
                path.remove(b)
                colliders.extend(colliders2)
                return False

            z.remove(b)
            path.remove(b)
            return True

    def get_pass_nodes(self, a, b, z, graph, colliders):

        pass_nodes = []

        for c in graph.get_adjacent_nodes(b):
            if c == a:
                continue

            if self.node_reachable(a, b, c, z, graph, colliders):
                pass_nodes.append(c)

        return pass_nodes

    def node_reachable(self, a, b, c, z, graph, colliders):
        collider = graph.is_def_collider(a, b, c)

        if not collider and not (b in z):
            return True

        ancestor = self.is_ancestor(b, z, graph)

        collider_reachable = collider and ancestor

        if colliders is None and collider and not ancestor:
            colliders.append((a, b, c))

        return collider_reachable

    def is_ancestor(self, b, z, graph):

        if b in z:
            return True

        Q = deque([])
        V = []

        for node in z:
            Q.append(node)
            V.append(node)

        while(len(Q) > 0):
            t = Q.pop()

            if t == b:
                return True

            for c in graph.get_parents(t):
                if not (c in V):
                    Q.append(c)
                    V.append(c)

        return False

    # Returns a tiered ordering of variables in an acyclic graph. THIS ALGORITHM IS NOT ALWAYS CORRECT.
    def get_causal_order(self, graph):

        if graph.exists_directed_cycle():
            raise ValueError("Graph must be acyclic.")

        found = []
        not_found = graph.get_nodes()

        sub_not_found = []

        for node in not_found:
            if node.get_node_type() == NodeType.ERROR:
                sub_not_found.append(node)

        not_found = [e for e in not_found if e not in sub_not_found]

        all_nodes = not_found.copy()

        while(len(not_found) > 0):
            sub_not_found = []
            for node in not_found:
                print(node)
                parents = graph.get_parents(node)
                sub_parents = []
                for node1 in parents:
                    if not (node1 in all_nodes):
                        sub_parents.append(node1)

                parents = [e for e in parents if e not in sub_parents]

                if(all(node1 in found for node1 in parents)):
                    found.append(node)
                    sub_not_found.append(node)

            not_found = [e for e in not_found if e not in sub_not_found]

        return found
