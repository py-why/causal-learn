#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import List, Tuple, Deque

import pydot

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.Edge import Edge
from causallearn.graph.Edges import Edges
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.Node import Node
from causallearn.graph.NodeType import NodeType


class GraphUtils:

    def __init__(self):
        pass

    # Returns true if node1 is d-connected to node2 on the set of nodes z.
    ### CURRENTLY THIS DOES NOT IMPLEMENT UNDERLINE TRIPLE EXCEPTIONS ###
    def is_dconnected_to(self, node1: Node, node2: Node, z: List[Node], graph: Graph):
        if node1 == node2:
            return True

        edgenode_deque = deque([])

        for edge in graph.get_node_edges(node1):
            if edge.get_distal_node(node1) == node2:
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

    def edge_string(self, edge: Edge) -> str:
        node1 = edge.get_node1()
        node2 = edge.get_node2()

        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        edge_string = node1.get_name() + " "

        if endpoint1 == Endpoint.TAIL:
            edge_string = edge_string + "-"
        else:
            if endpoint1 == Endpoint.ARROW:
                edge_string = edge_string + "<"
            else:
                edge_string = edge_string + "o"

        edge_string = edge_string + "-"

        if endpoint2 == Endpoint.TAIL:
            edge_string = edge_string + "-"
        else:
            if endpoint2 == Endpoint.ARROW:
                edge_string = edge_string + ">"
            else:
                edge_string = edge_string + "o"

        edge_string = edge_string + " " + node2.get_name()
        return edge_string

    def graph_string(self, graph: Graph) -> str:
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
    def reachable(self, edge1: Edge, edge2: Edge, node_a: Node, z: List[Node], graph: Graph) -> bool:
        node_b = edge1.get_distal_node(node_a)

        collider = str(edge1.get_proximal_endpoint(node_b)) == "ARROW" and str(
            edge2.get_proximal_endpoint(node_b)) == "ARROW"

        if (not collider) and not (node_b in z):
            return True

        ancestor = self.is_ancestor(node_b, z, graph)

        return collider and ancestor

    # Helper method. Determines if a given node is an ancestor of any node in a set of nodes z.
    def is_ancestor(self, node: Node, z: List[Node], graph: Graph) -> bool:
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
                if node_c not in nodedeque:
                    nodedeque.append(node_c)
        return False

    def get_sepset(self, x: Node, y: Node, graph: Graph) -> List[Node] | None:
        sepset = self.get_sepset_visit(x, y, graph)
        if sepset is None:
            sepset = self.get_sepset_visit(y, x, graph)

        return sepset

    def get_sepset_visit(self, x: Node, y: Node, graph: Graph) -> List[Node] | None:
        if x == y:
            return None

        z: List[Node] = []

        while True:
            _z = z.copy()
            path: List[Node] = [x]
            colliders = []

            for b in graph.get_adjacent_nodes(x):
                if self.sepset_path_found(x, b, y, path, z, graph, colliders):
                    return None

            z.sort()
            _z.sort()
            if z == _z:
                break

        return z

    def sepset_path_found(self, a: Node, b: Node, y: Node, path: List[Node], z: List[Node], graph: Graph,
                          colliders: List[Tuple[Node, Node, Node]]) -> bool:
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
            colliders2: List[Tuple[Node, Node, Node]] = []
            pass_nodes2 = self.get_pass_nodes(a, b, z, graph, None)

            for c in pass_nodes2:
                if self.sepset_path_found(b, c, y, path, z, graph, colliders2):
                    found2 = True
                    break

            if not found2:
                path.remove(b)
                colliders.extend(colliders2)
                return False

            z.remove(b)
            path.remove(b)
            return True

    def get_pass_nodes(self, a: Node, b: Node, z: List[Node], graph: Graph,
                       colliders: List[Tuple[Node, Node, Node]] | None) -> List[Node]:
        pass_nodes: List[Node] = []

        for c in graph.get_adjacent_nodes(b):
            if c == a:
                continue

            if self.node_reachable(a, b, c, z, graph, colliders):
                pass_nodes.append(c)

        return pass_nodes

    def node_reachable(self, a: Node, b: Node, c: Node, z: List[Node], graph: Graph,
                       colliders: List[Tuple[Node, Node, Node]] | None) -> bool:
        collider = graph.is_def_collider(a, b, c)

        if not collider and not (b in z):
            return True

        ancestor = self.is_ancestor(b, z, graph)

        collider_reachable = collider and ancestor

        if colliders is not None and collider and not ancestor:
            colliders.append((a, b, c))

        return collider_reachable

    # Returns a tiered ordering of variables in an acyclic graph. THIS ALGORITHM IS NOT ALWAYS CORRECT.
    def get_causal_order(self, graph: Graph) -> List[Node]:
        if graph.exists_directed_cycle():
            raise ValueError("Graph must be acyclic.")

        found: List[Node] = []
        not_found: List[Node] = graph.get_nodes()
        sub_not_found: List[Node] = []

        for node in not_found:
            if node.get_node_type() == NodeType.ERROR:
                sub_not_found.append(node)

        not_found = [e for e in not_found if e not in sub_not_found]

        all_nodes = not_found.copy()

        while len(not_found) > 0:
            sub_not_found: List[Node] = []
            for node in not_found:
                # print(node)
                parents = graph.get_parents(node)
                sub_parents: List[Node] = []
                for node1 in parents:
                    if not (node1 in all_nodes):
                        sub_parents.append(node1)

                parents = [e for e in parents if e not in sub_parents]

                if all(node1 in found for node1 in parents):
                    found.append(node)
                    sub_not_found.append(node)

            not_found = [e for e in not_found if e not in sub_not_found]

        return found

    def find_unshielded_triples(self, graph: Graph):
        """Return the list of unshielded triples i o-o j o-o k in adjmat as (i, j, k)"""
        from causallearn.graph.Dag import Dag
        if not isinstance(graph, Dag):
            raise ValueError("graph must be a DAG")
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

    def find_triangles(self, graph: Graph) -> List[Tuple[Node, Node, Node]]:
        """Return the list of triangles i o-o j o-o k o-o i in adjmat as (i, j, k) [with symmetry]"""
        Adj = graph.get_graph_edges()
        triangles: List[Tuple[Node, Node, Node]] = []

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

    def find_kites(self, graph) -> List[Tuple[Node, Node, Node, Node]]:
        kites: List[Tuple[Node, Node, Node, Node]] = []
        for pair in permutations(self.find_triangles(graph), 2):
            if (pair[0][0] == pair[1][0]) and (pair[0][2] == pair[1][2]) and (
                    graph.node_map[pair[0][1]] < graph.node_map[pair[1][1]]) and (
                    graph.graph[graph.node_map[pair[0][1]], graph.node_map[pair[1][1]]] == 0):
                kites.append((pair[0][0], pair[0][1], pair[1][1], pair[0][2]))

        return kites

        # return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.findTriangles(), 2)
        #        if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
        #        and pair[0][1] < pair[1][1] and self.adjmat[pair[0][1], pair[1][1]] == -1]

    def sdh(self, graph1: Graph, graph2: Graph) -> int:
        nodes = graph1.get_nodes()
        error = 0

        for i1 in list(range(1, graph1.get_num_nodes())):
            for i2 in list(range(i1 + 1, graph1.get_num_nodes())):
                e1 = graph1.get_edge(nodes[i1], nodes[i2])
                e2 = graph2.get_edge(nodes[i1], nodes[i2])
                error = error + self.shd_one_edge(e1, e2)

        return error

    def shd_one_edge(self, e1: Edge, e2: Edge) -> int:
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

    def no_edge(self, e: Edge | None) -> bool:
        return e is None

    def undirected(self, e: Edge) -> bool:
        return e.get_endpoint1() == Endpoint.TAIL and e.get_endpoint2() == Endpoint.TAIL

    def directed(self, e: Edge) -> bool:
        return (e.get_endpoint1() == Endpoint.TAIL and e.get_endpoint2() == Endpoint.ARROW) \
               or (e.get_endpoint1() == Endpoint.ARROW and e.get_endpoint2() == Endpoint.TAIL)

    def bi_directed(self, e: Edge) -> bool:
        return e.get_endpoint1() == Endpoint.ARROW and e.get_endpoint2() == Endpoint.ARROW

    def adj_precision(self, truth: Graph, est: Graph) -> float:
        confusion = AdjacencyConfusion(truth, est)
        return confusion.get_adj_tp() / (confusion.get_adj_tp() + confusion.get_adj_fp())

    def adj_recall(self, truth: Graph, est: Graph) -> float:
        confusion = AdjacencyConfusion(truth, est)
        return confusion.get_adj_tp() / (confusion.get_adj_tp() + confusion.get_adj_fn())

    def arrow_precision(self, truth: Graph, est: Graph) -> float:
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fp())

    def arrow_recall(self, truth: Graph, est: Graph) -> float:
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fn())

    def arrow_precision_common_edges(self, truth: Graph, est: Graph) -> float:
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fp_ce())

    def arrow_recall_common_edges(self, truth: Graph, est: Graph) -> float:
        confusion = ArrowConfusion(truth, est)
        return confusion.get_arrows_tp() / (confusion.get_arrows_tp() + confusion.get_arrows_fn_ce())

    def exists_directed_path_from_to_breadth_first(self, node_from: Node, node_to: Node, G: Graph) -> bool:
        Q: Deque[Node] = deque()
        V: List[Node] = [node_from]
        Q.append(node_from)

        while len(Q) > 0:
            t = Q.pop()
            for u in G.get_adjacent_nodes(t):
                if G.is_parent_of(t, u) and G.is_parent_of(u, t):
                    return True

                edge = G.get_edge(t, u)
                edges = Edges()
                c = edges.traverse_directed(t, edge)

                if c is None:
                    continue
                if c == node_to:
                    return True
                if c in V:
                    continue

                V.append(c)
                Q.append(c)

    @staticmethod
    def to_pgv(G: Graph, title: str = ""):
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
    def to_pydot(G: Graph, edges: List[Edge] | None = None, labels: List[str] | None = None,
                 colors: dict | None = None, title: str = "", dpi: float = 200):
        '''
        Convert a graph object to a DOT object, optionally with colored nodes.

        Parameters
        ----------
        G : Graph
            A graph object of causal-learn.
        edges : list, optional (default=None)
            Edges list of graph G. If None, uses G.get_graph_edges().
        labels : list of str, optional (default=None)
            Node labels. Must have the same length as G.get_nodes().
            If None, uses the default node names from G.
        colors : dict, optional (default=None)
            Mapping from node label (str) to fill color (str).
            Colors can be any CSS/X11 color name (e.g. 'lightblue', 'lightcoral')
            or hex code (e.g. '#AADDFF'). Only nodes whose labels appear
            in this dict will be colored; others remain uncolored.
        title : str, optional (default="")
            The name of graph G.
        dpi : float, optional (default=200)
            The dots per inch of dot object.

        Returns
        -------
        pydot_g : pydot.Dot
            A DOT object ready for rendering.

        Examples
        --------
        >>> # Basic usage without colors
        >>> pyd = GraphUtils.to_pydot(G, labels=['X1', 'X2', 'X3'])
        >>>
        >>> # With colored nodes by category
        >>> colors = {'X1': 'lightblue', 'X2': 'lightblue', 'X3': 'lightcoral'}
        >>> pyd = GraphUtils.to_pydot(G, labels=['X1', 'X2', 'X3'], colors=colors)
        '''

        nodes = G.get_nodes()
        if labels is not None:
            assert len(labels) == len(nodes)

        pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
        pydot_g.obj_dict["attributes"]["dpi"] = dpi

        for i, node in enumerate(nodes):
            node_name = labels[i] if labels is not None else node.get_name()
            node_attrs = {"label": node_name}
            if node.get_node_type() == NodeType.LATENT:
                node_attrs["shape"] = "square"
            if colors is not None and node_name in colors:
                node_attrs["style"] = "filled"
                node_attrs["fillcolor"] = colors[node_name]
            pydot_g.add_node(pydot.Node(i, **node_attrs))

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

    # ----------------------------------------------------------------
    # Color graph visualization utilities
    #
    # Quick-start example:
    #
    #   from causallearn.search.ScoreBased.GES import ges
    #   from causallearn.utils.GraphUtils import GraphUtils
    #
    #   result = ges(data, score_func='local_score_BIC')
    #   G = result['G']
    #   labels = ['Drug_A', 'Drug_B', 'Biomarker', 'Recovery', 'Survival']
    #
    #   # Option 1: color nodes by category (labels auto-derived from categories)
    #   categories = {
    #       'treatment': ['Drug_A', 'Drug_B'],
    #       'biomarker': ['Biomarker'],
    #       'outcome':   ['Recovery', 'Survival'],
    #   }
    #   GraphUtils.plot_graph(G, category_to_features=categories,
    #                         save_path='colored_graph.png')
    #
    #   # Option 2: specify labels and colors manually
    #   colors = {'Drug_A': 'lightblue', 'Drug_B': 'lightblue',
    #             'Biomarker': 'lightgreen',
    #             'Recovery': 'lightcoral', 'Survival': 'lightcoral'}
    #   GraphUtils.plot_graph(G, labels=labels, colors=colors,
    #                         save_path='colored_graph.png')
    # ----------------------------------------------------------------

    @staticmethod
    def get_category_colors(category_to_features: dict) -> dict:
        '''
        Assign a color to each feature based on its category.

        Parameters
        ----------
        category_to_features : dict
            Mapping from category name to list of feature names.
            Example: {'treatment': ['Drug_A', 'Drug_B'], 'outcome': ['Recovery', 'Side_Effect']}

        Returns
        -------
        feature_to_color : dict
            Mapping from feature name to CSS color string.

        Examples
        --------
        >>> categories = {
        ...     'treatment': ['Drug_A', 'Drug_B'],
        ...     'outcome': ['Recovery', 'Side_Effect'],
        ... }
        >>> colors = GraphUtils.get_category_colors(categories)
        >>> # colors = {'Drug_A': 'lightblue', 'Drug_B': 'lightblue',
        >>> #           'Recovery': 'lightcoral', 'Side_Effect': 'lightcoral'}
        '''
        palette = [
            'lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightpink',
            'lightyellow', 'lavender', 'thistle', 'honeydew', 'mintcream',
            'azure', 'aliceblue', 'beige', 'peachpuff', 'moccasin',
            'palegoldenrod', 'powderblue', 'khaki', 'wheat', 'blanchedalmond',
            'papayawhip', 'mistyrose', 'lemonchiffon', 'seashell', 'cornsilk',
            'aquamarine', 'lightcyan', 'lightskyblue', 'lightsteelblue',
            'paleturquoise', 'palegreen', 'pink', 'plum', 'skyblue',
        ]
        category_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(category_to_features)}
        return {
            feat: category_to_color[cat]
            for cat, feats in category_to_features.items()
            for feat in feats
        }

    @staticmethod
    def plot_graph(G: Graph, labels: List[str] | None = None,
                   colors: dict | None = None, category_to_features: dict | None = None,
                   save_path: str | None = None, title: str = "",
                   dpi: float = 500, figsize: tuple = (20, 12)):
        '''
        Render and display a causal graph with optional colored nodes.

        This is a convenience function that combines to_pydot() rendering
        with matplotlib display and optional file saving.

        Parameters
        ----------
        G : Graph
            A graph object of causal-learn.
        labels : list of str, optional (default=None)
            Node labels. Must have the same length as G.get_nodes().
        colors : dict, optional (default=None)
            Mapping from node label (str) to fill color (str).
            If both colors and category_to_features are provided, colors takes precedence.
        category_to_features : dict, optional (default=None)
            Mapping from category name to list of feature names.
            Used to auto-generate colors via get_category_colors().
            Ignored if colors is already provided.
        save_path : str, optional (default=None)
            File path to save the rendered image (e.g. 'output/graph.png').
            If None, the image is only displayed, not saved.
        title : str, optional (default="")
            Title for the graph.
        dpi : float, optional (default=500)
            Resolution in dots per inch.
        figsize : tuple, optional (default=(20, 12))
            Figure size in inches (width, height).

        Examples
        --------
        >>> # Simple plot
        >>> GraphUtils.plot_graph(G, labels=['X1', 'X2', 'X3'])
        >>>
        >>> # Plot with manual colors
        >>> colors = {'X1': 'lightblue', 'X2': 'lightcoral', 'X3': 'lightgreen'}
        >>> GraphUtils.plot_graph(G, labels=['X1', 'X2', 'X3'], colors=colors,
        ...                      save_path='my_graph.png')
        >>>
        >>> # Plot with category-based colors (auto-assigned)
        >>> # labels are auto-derived from category_to_features (flattened in order)
        >>> categories = {
        ...     'treatment': ['X1', 'X2'],
        ...     'outcome': ['X3'],
        ... }
        >>> GraphUtils.plot_graph(G, category_to_features=categories,
        ...                      save_path='my_graph.png')
        '''
        import io
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if colors is None and category_to_features is not None:
            colors = GraphUtils.get_category_colors(category_to_features)
            if labels is None:
                cat_features = [feat for feats in category_to_features.values() for feat in feats]
                n_nodes = len(G.get_nodes())
                if len(cat_features) == n_nodes:
                    labels = cat_features
                else:
                    # category_to_features doesn't cover all nodes;
                    # use default node names, colors will still apply to matching labels
                    labels = [node.get_name() for node in G.get_nodes()]

        pyd = GraphUtils.to_pydot(G, labels=labels, colors=colors, title=title, dpi=dpi)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(img)
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
