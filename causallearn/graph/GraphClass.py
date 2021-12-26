import io
import warnings
from itertools import permutations

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.Helper import list_union, powerset


class CausalGraph:
    def __init__(self, no_of_var):
        node_names = [("X%d" % (i + 1)) for i in range(no_of_var)]
        nodes = []
        for name in node_names:
            node = GraphNode(name)
            nodes.append(node)
        self.G = GeneralGraph(nodes)
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                self.G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))

        self.data = None  # store the data
        self.test = None  # store the name of the conditional independence test
        self.corr_mat = None  # store the correlation matrix of the data
        self.sepset = np.empty((no_of_var, no_of_var), object)  # store the collection of sepsets
        self.definite_UC = []  # store the list of definite unshielded colliders
        self.definite_non_UC = []  # store the list of definite unshielded non-colliders
        self.PC_elapsed = -1  # store the elapsed time of running PC
        self.redundant_nodes = []  # store the list of redundant nodes (for subgraphs)
        self.nx_graph = nx.DiGraph()  # store the directed graph
        self.nx_skel = nx.Graph()  # store the undirected graph
        self.labels = {}
        self.prt_m = {}  # store the parents of missingness indicators
        self.mvpc = None
        self.cardinalities = None  # only works when self.data is discrete, i.e. self.test is chisq or gsq
        self.is_discrete = False
        self.citest_cache = dict()
        self.data_hash_key = None
        self.ci_test_hash_key = None

    def set_ind_test(self, indep_test, mvpc=False):
        """Set the conditional independence test that will be used"""
        # assert name_of_test in ["Fisher_Z", "Chi_sq", "G_sq"]
        if mvpc:
            self.mvpc = True
        self.test = indep_test
        self.ci_test_hash_key = hash(indep_test)

    def ci_test(self, i, j, S):
        """Define the conditional independence test"""
        # assert i != j and not i in S and not j in S
        if self.mvpc:
            return self.test(self.data, self.nx_skel, self.prt_m, i, j, S, self.data.shape[0])

        i, j = (i, j) if (i < j) else (j, i)
        ijS_key = (i, j, frozenset(S), self.data_hash_key, self.ci_test_hash_key)
        if ijS_key in self.citest_cache:
            return self.citest_cache[ijS_key]
        # if discrete, assert self.test is chisq or gsq, pass into cardinalities
        pValue = self.test(self.data, i, j, S, self.cardinalities) if self.is_discrete \
            else self.test(self.data, i, j, S)
        self.citest_cache[ijS_key] = pValue
        return pValue

    def neighbors(self, i):
        """Find the neighbors of node i in adjmat"""
        return np.where(self.G.graph[i, :] != 0)[0]

    def max_degree(self):
        """Return the maximum number of edges connected to a node in adjmat"""
        return max(np.sum(self.G.graph != 0, axis=1))

    def find_arrow_heads(self):
        """Return the list of i o-> j in adjmat as (i, j)"""
        L = np.where(self.G.graph == 1)
        return list(zip(L[1], L[0]))

    def find_tails(self):
        """Return the list of i --o j in adjmat as (j, i)"""
        L = np.where(self.G.graph == -1)
        return list(zip(L[1], L[0]))

    def find_undirected(self):
        """Return the list of undirected edge i --- j in adjmat as (i, j) [with symmetry]"""
        return [(edge[0], edge[1]) for edge in self.find_tails() if self.G.graph[edge[0], edge[1]] == -1]

    def find_fully_directed(self):
        """Return the list of directed edges i --> j in adjmat as (i, j)"""
        return [(edge[0], edge[1]) for edge in self.find_arrow_heads() if self.G.graph[edge[0], edge[1]] == -1]

    def find_bi_directed(self):
        """Return the list of bidirected edges i <-> j in adjmat as (i, j) [with symmetry]"""
        return [(edge[1], edge[0]) for edge in self.find_arrow_heads() if (
                self.G.graph[edge[1], edge[0]] == Endpoint.ARROW.value and self.G.graph[
            edge[0], edge[1]] == Endpoint.ARROW.value)]

    def find_adj(self):
        """Return the list of adjacencies i --- j in adjmat as (i, j) [with symmetry]"""
        return list(self.find_tails() + self.find_arrow_heads())

    def is_undirected(self, i, j):
        """Return True if i --- j holds in adjmat and False otherwise"""
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == -1

    def is_fully_directed(self, i, j):
        """Return True if i --> j holds in adjmat and False otherwise"""
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == 1

    def find_unshielded_triples(self):
        """Return the list of unshielded triples i o-o j o-o k in adjmat as (i, j, k)"""
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(self.find_adj(), 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and self.G.graph[pair[0][0], pair[1][1]] == 0]

    def find_triangles(self):
        """Return the list of triangles i o-o j o-o k o-o i in adjmat as (i, j, k) [with symmetry]"""
        Adj = self.find_adj()
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(Adj, 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and (pair[0][0], pair[1][1]) in Adj]

    def find_kites(self):
        """Return the list of non-ambiguous kites i o-o j o-o l o-o k o-o i o-o l in adjmat \
        (where j and k are non-adjacent) as (i, j, k, l) [with asymmetry j < k]"""
        return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.find_triangles(), 2)
                if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
                and pair[0][1] < pair[1][1] and self.G.graph[pair[0][1], pair[1][1]] == 0]

    def find_cond_sets(self, i, j):
        """return the list of conditioning sets of the neighbors of i or j in adjmat"""
        neigh_x = self.neighbors(i)
        neigh_y = self.neighbors(j)
        pow_neigh_x = powerset(neigh_x)
        pow_neigh_y = powerset(neigh_y)
        return list_union(pow_neigh_x, pow_neigh_y)

    def find_cond_sets_with_mid(self, i, j, k):
        """return the list of conditioning sets of the neighbors of i or j in adjmat which contains k"""
        return [S for S in self.find_cond_sets(i, j) if k in S]

    def find_cond_sets_without_mid(self, i, j, k):
        """return the list of conditioning sets of the neighbors of i or j which in adjmat does not contain k"""
        return [S for S in self.find_cond_sets(i, j) if k not in S]

    def rearrange(self, PATH):
        """Rearrange adjmat according to the data imported at PATH"""
        raw_col_names = list(pd.read_csv(PATH, sep='\t').columns)
        var_indices = []
        for name in raw_col_names:
            var_indices.append(int(name.split('X')[1]) - 1)
        new_indices = np.zeros_like(var_indices)
        for i in range(1, len(new_indices)):
            new_indices[var_indices[i]] = range(len(new_indices))[i]
        output = self.adjmat[:, new_indices]
        output = output[new_indices, :]
        self.adjmat = output

    def to_nx_graph(self):
        """Convert adjmat into a networkx.Digraph object named nx_graph"""
        nodes = range(len(self.G.graph))
        self.labels = {i: self.G.nodes[i].get_name() for i in nodes}
        self.nx_graph.add_nodes_from(nodes)
        undirected = self.find_undirected()
        directed = self.find_fully_directed()
        bidirected = self.find_bi_directed()
        for (i, j) in undirected:
            self.nx_graph.add_edge(i, j, color='g')  # Green edge: undirected edge
        for (i, j) in directed:
            self.nx_graph.add_edge(i, j, color='b')  # Blue edge: directed edge
        for (i, j) in bidirected:
            self.nx_graph.add_edge(i, j, color='r')  # Red edge: bidirected edge

    def to_nx_skeleton(self):
        """Convert adjmat into its skeleton (a networkx.Graph object) named nx_skel"""
        nodes = range(len(self.G.graph))
        self.nx_skel.add_nodes_from(nodes)
        adj = [(i, j) for (i, j) in self.find_adj() if i < j]
        for (i, j) in adj:
            self.nx_skel.add_edge(i, j, color='g')  # Green edge: undirected edge

    def draw_nx_graph(self, skel=False):
        """Draw nx_graph if skel = False and draw nx_skel otherwise"""
        if not skel:
            print("Green: undirected; Blue: directed; Red: bi-directed\n")
        warnings.filterwarnings("ignore", category=UserWarning)
        g_to_be_drawn = self.nx_skel if skel else self.nx_graph
        edges = g_to_be_drawn.edges()
        colors = [g_to_be_drawn[u][v]['color'] for u, v in edges]
        pos = nx.circular_layout(g_to_be_drawn)
        nx.draw(g_to_be_drawn, pos=pos, with_labels=True, labels=self.labels, edge_color=colors)
        plt.draw()
        plt.show()

    def draw_pydot_graph(self):
        """Draw nx_graph if skel = False and draw nx_skel otherwise"""
        warnings.filterwarnings("ignore", category=UserWarning)
        pyd = GraphUtils.to_pydot(self.G)
        tmp_png = pyd.create_png(f="png")
        pyd.write_png("result.png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()
