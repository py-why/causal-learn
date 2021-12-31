#!/usr/bin/env python3

from collections import deque
from itertools import permutations

import pydot

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.Edge import Edge
from causallearn.graph.Edges import Edges
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.NodeType import NodeType


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

        # nodes.sort()
        # edges.sort()

        graph_string = "Graph Nodes:\n"

        for i in range(len(nodes) - 1):
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

        collider = str(edge1.get_proximal_endpoint(node_b)) == "ARROW" and str(
            edge2.get_proximal_endpoint(node_b)) == "ARROW"

        if (not collider) and not (node_b in z):
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

        while (len(Q) > 0):
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

        while (len(not_found) > 0):
            sub_not_found = []
            for node in not_found:
                print(node)
                parents = graph.get_parents(node)
                sub_parents = []
                for node1 in parents:
                    if not (node1 in all_nodes):
                        sub_parents.append(node1)

                parents = [e for e in parents if e not in sub_parents]

                if (all(node1 in found for node1 in parents)):
                    found.append(node)
                    sub_not_found.append(node)

            not_found = [e for e in not_found if e not in sub_not_found]

        return found

    def find_unshielded_triples(self, graph):
        """Return the list of unshielded triples i o-o j o-o k in adjmat as (i, j, k)"""

        triples = []

        for pair in permutations(graph.get_graph_edges(), 2):
            node1 = pair[0].get_node1()
            node2 = pair[0].get_node2()
            node3 = pair[1].get_node1()
            node4 = pair[1].get_node1()

            node_map = graph.get_node_map()

            if node1 == node3:
                if node2 != node4 and graph.get_adjacency_matrix()[node_map[node2], node_map[node4]] == 0:
                    triples.append((node2, node1, node4))
                    continue
            if node1 == node4:
                if node2 != node3 and graph.get_adjacency_matrix()[node_map[node2], node_map[node3]] == 0:
                    triples.append((node2, node1, node3))
                    continue
            if node2 == node3:
                if node1 != node4 and graph.get_adjacency_matrix()[node_map[node1], node_map[node4]] == 0:
                    triples.append((node1, node2, node4))
                    continue
            if node2 == node4:
                if node2 != node3 and graph.get_adjacency_matrix()[node_map[node2], node_map[node3]] == 0:
                    triples.append((node1, node2, node3))

        return triples

    #    return [(pair[0].get_node1(), pair[0].get_node2(), pair[1].get_node2) for pair in permutations(graph.get_graph_edges(), 2)
    #            if pair[0].get_node2() == pair[1].get_node1() and pair[0].get_node1() != pair[1].get_node2() and graph.get_adjacency_matrix()[graph.get_node_map()[pair[0].get_node1()], graph.get_node_map()[pair[1].get_node2()]] == -1]

    def find_triangles(self, graph):
        """Return the list of triangles i o-o j o-o k o-o i in adjmat as (i, j, k) [with symmetry]"""
        Adj = graph.get_graph_edges()
        triangles = []

        for pair in permutations(Adj, 2):
            node1 = pair[0].get_node1()
            node2 = pair[0].get_node2()
            node3 = pair[1].get_node1()
            node4 = pair[1].get_node2()

            if node1 == node3:
                if graph.is_adjacent_to(node2, node4):
                    triangles.append((node2, node1, node4))
                    continue
            if node1 == node4:
                if graph.is_adjacent_to(node2, node3):
                    triangles.append((node2, node1, node3))
                    continue
            if node2 == node3:
                if graph.is_adjacent_to(node1, node4):
                    triangles.append((node1, node2, node4))
                    continue
            if node2 == node4:
                if graph.is_adjacent_to(node1, node3):
                    triangles.append((node1, node2, node3))

        return triangles

    #    return [(pair[0].get_node1(), pair[0].get_node2(), pair[1].get_node2) for pair in permutations(Adj, 3)
    #            if pair[0].get_node2 == pair[1].get_node1() and pair[0].get_node1() != pair[1].get_node2() and (pair[0][0], pair[1][1]) in Adj]

    def find_kites(self, graph):

        kites = []

        for pair in permutations(self.find_triangles(graph), 2):
            if (pair[0][0] == pair[1][0]) and (pair[0][2] == pair[1][2]) and (
                    graph.node_map[pair[0][1]] < graph.node_map[pair[1][1]]) and (
                    graph.graph[graph.node_map[pair[0][1]], graph.node_map[pair[1][1]]] == 0):
                kites.append((pair[0][0], pair[0][1], pair[1][1], pair[0][2]))

        return kites

        # return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.findTriangles(), 2)
        #        if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
        #        and pair[0][1] < pair[1][1] and self.adjmat[pair[0][1], pair[1][1]] == -1]

    def sdh(self, graph1: Graph, graph2: Graph):
        nodes = graph1.get_nodes()
        error = 0

        for i1 in list(range(1, graph1.get_num_nodes())):
            for i2 in list(range(i1 + 1, graph1.get_num_nodes())):
                e1 = graph1.get_edge(nodes[i1], nodes[i2])
                e2 = graph2.get_edge(nodes[i1], nodes[i2])
                error = error + self.shd_one_edge(e1, e2)

        return error

    def shd_one_edge(self, e1: Edge, e2: Edge):
        if self.no_edge(e1) and self.undirected(e2):
            return 1
        elif self.no_edge(e2) and self.undirected(e1):
            return 1
        elif self.no_edge(e1) and self.directed(e2):
            return 2
        elif self.no_edge(e2) and self.directed(e1):
            return 2
        elif self.undirected(e1) and self.directed(e2):
            return 1
        elif self.undirected(e2) and self.directed(e1):
            return 1
        elif self.directed(e1) and self.directed(e2):
            if e1.get_endpoint1() == e2.get_endpoint2():
                return 1
        elif self.bi_directed(e1) or self.bi_directed(e2):
            return 2

        return 0

    def no_edge(self, e: Edge):
        return e == None

    def undirected(self, e: Edge):
        return e.get_endpoint1() == Endpoint.TAIL and e.get_endpoint2() == Endpoint.TAIL

    def directed(self, e: Edge):
        return (e.get_endpoint1() == Endpoint.TAIL and e.get_endpoint2() == Endpoint.ARROW) \
               or (e.get_endpoint1() == Endpoint.ARROW and e.get_endpoint2() == Endpoint.TAIL)

    def bi_directed(self, e: Edge):
        return e.get_endpoint1() == Endpoint.ARROW and e.get_endpoint2() == Endpoint.ARROW

    def adj_precision(self, truth: Graph, est: Graph):
        confusion = AdjacencyConfusion(truth, est)
        return confusion.get_adj_tp() / (confusion.get_adj_tp() + confusion.get_adj_fp())

    def adj_recall(self, truth: Graph, est: Graph):
        confusion = AdjacencyConfusion(truth, est)
        return confusion.get_adj_tp() / (confusion.get_adj_tp() + confusion.get_adj_fn())

    def arrow_precision(self, truth: Graph, est: Graph):
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fp())

    def arrow_recall(self, truth: Graph, est: Graph):
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fn())

    def arrow_precision_common_edges(self, truth: Graph, est: Graph):
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fp_ce())

    def arrow_recall_common_edges(self, truth: Graph, est: Graph):
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fn_ce())

    def exists_directed_path_from_to_breadth_first(self, node_from, node_to, G):

        Q = deque()
        V = [node_from]
        Q.append(node_from)

        while len(Q) > 0:
            t = Q.pop()

            for u in G.get_adjacent_nodes(t):
                if G.is_parent_of(t, u) and G.is_parent_of(u, t):
                    return True

                edge = G.get_edge(t, u)
                edges = Edges()
                c = edges.traverse_directed(t, edge)

                if c == None:
                    continue
                if c in V:
                    continue
                if c == node_to:
                    return True

                V.append(c)
                Q.append(c)

    @staticmethod
    def to_pgv(G, title=""):
        # warnings.warn("GraphUtils.to_pgv() is deprecated", DeprecationWarning)
        import pygraphviz as pgv
        graphviz_g = pgv.AGraph(directed=True)
        graphviz_g.graph_attr['label'] = title
        graphviz_g.graph_attr['labelfontsize'] = 18
        nodes = G.get_nodes()
        for i, node in enumerate(nodes):
            graphviz_g.add_node(i)
            graphviz_g.get_node(i).attr['label'] = node.get_name()
            if node.get_node_type() == NodeType.LATENT:
                graphviz_g.get_node(i).attr['shape'] = 'square'

        def get_g_arrow_type(endpoint):
            if endpoint == Endpoint.TAIL:
                return 'none'
            elif endpoint == Endpoint.ARROW:
                return 'normal'
            elif endpoint == Endpoint.CIRCLE:
                return 'odot'
            else:
                raise NotImplementedError()

        for edge in G.get_graph_edges():
            if not edge:
                continue
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            node1_id = nodes.index(node1)
            node2_id = nodes.index(node2)
            graphviz_g.add_edge(node1_id, node2_id)
            g_edge = graphviz_g.get_edge(node1_id, node2_id)
            g_edge.attr['dir'] = 'both'

            g_edge.attr['arrowtail'] = get_g_arrow_type(edge.get_endpoint1())
            g_edge.attr['arrowhead'] = get_g_arrow_type(edge.get_endpoint2())

        return graphviz_g

    @staticmethod
    def to_pydot(G, edges=None, title="", dpi=200):
        pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
        pydot_g.obj_dict["attributes"]["dpi"] = dpi
        nodes = G.get_nodes()
        for i, node in enumerate(nodes):
            pydot_g.add_node(pydot.Node(i, label=node.get_name()))
            if node.get_node_type() == NodeType.LATENT:
                pydot_g.add_node(pydot.Node(i, label=node.get_name(), shape='square'))
            else:
                pydot_g.add_node(pydot.Node(i, label=node.get_name()))

        def get_g_arrow_type(endpoint):
            if endpoint == Endpoint.TAIL:
                return 'none'
            elif endpoint == Endpoint.ARROW:
                return 'normal'
            elif endpoint == Endpoint.CIRCLE:
                return 'odot'
            else:
                raise NotImplementedError()

        if edges is None:
            edges = G.get_graph_edges()

        for edge in edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            node1_id = nodes.index(node1)
            node2_id = nodes.index(node2)
            dot_edge = pydot.Edge(node1_id, node2_id, dir='both', arrowtail=get_g_arrow_type(edge.get_endpoint1()),
                                  arrowhead=get_g_arrow_type(edge.get_endpoint2()))

            if Edge.Property.dd in edge.properties:
                dot_edge.obj_dict["attributes"]["color"] = "green3"

            if Edge.Property.nl in edge.properties:
                dot_edge.obj_dict["attributes"]["penwidth"] = 2.0

            pydot_g.add_edge(dot_edge)

        return pydot_g
