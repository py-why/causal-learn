#######################################################################################################################
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
from itertools import permutations
from Helper import listIntersection, listMinus, listUnion, powerset, fisherZ, chisq
#######################################################################################################################


class CausalGraph:
    def __init__(self, no_of_var):
        self.adjmat = np.zeros((no_of_var, no_of_var))  # store the adjacency matrix of the estimated graph
        np.fill_diagonal(self.adjmat, None)
        self.data = None  # store the data
        self.test = str()  # store the name of the conditional independence test
        self.corr_mat = None  # store the correlation matrix of the data
        self.nx_graph = nx.DiGraph()  # store the directed graph
        self.nx_skel = nx.Graph()  # store the undirected graph
        self.sepset = np.empty((no_of_var, no_of_var), object)  # store the collection of sepsets
        self.definite_UC = []  # store the list of definite unshielded colliders
        self.definite_non_UC = []  # store the list of definite unshielded non-colliders
        self.PC_elapsed = -1  # store the elapsed time of running PC
        self.redundant_nodes = []  # store the list of redundant nodes (for subgraphs)

    ####################################################################################################################

    def setTestName(self, name_of_test):
        """Set the conditional independence test that will be used"""
        assert name_of_test in ["Fisher_Z", "Chi_sq", "G_sq"]
        self.test = name_of_test

    ####################################################################################################################

    def ci_test(self, i, j, S):
        """Define the conditional independence test"""
        if self.test == "Fisher_Z":
            return fisherZ(self.corr_mat, i, j, S, self.data.shape[0])
        elif self.test == "Chi_sq":
            return chisq(self.data, i, j, S, G_sq=False)
        elif self.test == "G_sq":
            return chisq(self.data, i, j, S, G_sq=True)

    ####################################################################################################################

    def neighbors(self, i):
        """Find the neighbors of node i in adjmat"""
        l0 = np.where(self.adjmat[i, :] == 0)[0]
        l1 = np.where(self.adjmat[i, :] == 1)[0]
        return np.concatenate((l0, l1))

    ####################################################################################################################

    def maxDegree(self):
        """Return the maximum number of edges connected to a node in adjmat"""
        nodes = range(len(self.adjmat))
        max_degree = 0
        for i in nodes:
            len_neigh_i = len(self.neighbors(i))
            if len_neigh_i > max_degree:
                max_degree = len_neigh_i
        return max_degree

    ####################################################################################################################

    def minDegree(self):
        """Return the minimum number of edges connected to a node in adjmat"""
        nodes = range(len(self.adjmat))
        non_redundant_nodes = [node for node in nodes if not all(np.isnan(self.adjmat[node, :]))]
        min_degree = len(non_redundant_nodes)
        for i in non_redundant_nodes:
            len_neigh_i = len(self.neighbors(i))
            if len_neigh_i < min_degree:
                min_degree = len_neigh_i
        return min_degree

    ####################################################################################################################

    def avgDegree(self):
        """Return the average number of edges connected to a node in adjmat"""
        nodes = range(len(self.adjmat))
        non_redundant_nodes = [node for node in nodes if not all(np.isnan(self.adjmat[node, :]))]
        num_of_edges = 0
        for i in non_redundant_nodes:
            num_of_edges += len(self.neighbors(i))
        num_of_nodes = len(non_redundant_nodes)
        return (0.5 * num_of_edges) / num_of_nodes

    ####################################################################################################################

    def density(self):
        """Return the density of adjmat"""
        num_of_nodes = len([node for node in range(len(self.adjmat)) if not all(np.isnan(self.adjmat[node, :]))])
        return self.avgDegree() / (num_of_nodes - 1)

    ####################################################################################################################

    def findArrowheads(self):
        """Return the list of i o-> j in adjmat as (i, j)"""
        L = np.where(self.adjmat == 1)
        return list(zip(L[0], L[1]))

    ####################################################################################################################

    def findTails(self):
        """Return the list of i --o j in adjmat as (j, i)"""
        L = np.where(self.adjmat == 0)
        return list(zip(L[0], L[1]))

    ####################################################################################################################

    def findUndirected(self):
        """Return the list of undirected edge i --- j in adjmat as (i, j) [with symmetry]"""
        return [(edge[0], edge[1]) for edge in self.findTails() if self.adjmat[edge[1], edge[0]] == 0]

    ####################################################################################################################

    def findFullyDirected(self):
        """Return the list of directed edges i --> j in adjmat as (i, j)"""
        return [(edge[0], edge[1]) for edge in self.findArrowheads() if self.adjmat[edge[1], edge[0]] == 0]

    ####################################################################################################################

    def findBiDirected(self):
        """Return the list of bidirected edges i <-> j in adjmat as (i, j) [with symmetry]"""
        return [(edge[0], edge[1]) for edge in self.findArrowheads() if self.adjmat[edge[1], edge[0]] == 1]

    ####################################################################################################################

    def findAdj(self):
        """Return the list of adjacencies i --- j in adjmat as (i, j) [with symmetry]"""
        return list(self.findTails() + self.findArrowheads())

    ####################################################################################################################

    def isUndirected(self, i, j):
        """Return True if i --- j holds in adjmat and False otherwise"""
        return self.adjmat[i, j] == 0 and self.adjmat[j, i] == 0

    ####################################################################################################################

    def isFullyDirected(self, i, j):
        """Return True if i --> j holds in adjmat and False otherwise"""
        return self.adjmat[i, j] == 1 and self.adjmat[j, i] == 0

    ####################################################################################################################

    def isBiDirected(self, i, j):
        """Return True if i <-> j holds in adjmat and False otherwise"""
        return self.adjmat[i, j] == 1 and self.adjmat[j, i] == 1

    ####################################################################################################################

    def isAdj(self, i, j):
        """Return True if i o-o j holds in adjmat and False otherwise"""
        return self.adjmat[i, j] != -1 and self.adjmat[j, i] != -1

    ####################################################################################################################

    def findUnshieldedTriples(self):
        """Return the list of unshielded triples i o-o j o-o k in adjmat as (i, j, k)"""
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(self.findAdj(), 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and self.adjmat[pair[0][0], pair[1][1]] == -1]

    ####################################################################################################################

    def findTriangles(self):
        """Return the list of triangles i o-o j o-o k o-o i in adjmat as (i, j, k) [with symmetry]"""
        Adj = self.findAdj()
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(Adj, 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and (pair[0][0], pair[1][1]) in Adj]

    ####################################################################################################################

    def findKites(self):
        """Return the list of non-ambiguous kites i o-o j o-o l o-o k o-o i o-o l in adjmat \
        (where j and k are non-adjacent) as (i, j, k, l) [with asymmetry j < k]"""
        return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.findTriangles(), 2)
                if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
                and pair[0][1] < pair[1][1] and self.adjmat[pair[0][1], pair[1][1]] == -1]

    ####################################################################################################################

    def findUC(self):
        """Return the list of unshielded colliders i --> j <-- k in adjmat as (i, j, k) [with asymmetry i < k]"""
        directed = self.findFullyDirected()
        return [(pair[0][0], pair[0][1], pair[1][0]) for pair in permutations(directed, 2)
                if pair[0][1] == pair[1][1] and pair[0][0] < pair[1][0] and self.adjmat[pair[0][0], pair[1][0]] == -1]

    ####################################################################################################################

    def isUnshieldedTriple(self, i, j, k):
        """Return True if (i, j, k) is an unshielded triple in adjmat and False otherwise"""
        return self.isAdj(i, j) and self.isAdj(j, k) and not self.isAdj(i, k)

    ####################################################################################################################

    def isCollider(self, i, j, k):
        """Return True if i o-> j <-o k in adjmat and False otherwise"""
        return self.adjmat[i, j] == 1 and self.adjmat[k, j] == 1

    ####################################################################################################################

    def isNonCollider(self, i, j, k):
        """Return True if i o-o j o-o k in adjmat is a non-collider and False otherwise"""
        return not self.isCollider(i, j, k)

    ####################################################################################################################

    def isUC(self, i, j, k):
        """Return True if the triple (i, j, k) is an unshielded collider in adjmat and False otherwise"""
        return self.isCollider(i, j, k) and self.isUnshieldedTriple(i, j, k)

    ####################################################################################################################

    def isTriangle(self, i, j, k):
        """Return True if the triple (i, j, k) is a triangle in adjmat and False otherwise"""
        return self.isAdj(i, j) and self.isAdj(j, k) and self.isAdj(i, k)

    ####################################################################################################################

    def isParent(self, i, j):
        """Return True if i --> j in adjmat, else False"""
        return self.isFullyDirected(i, j)

    ####################################################################################################################
    def isChild(self, i, j):
        """Return True if isParent(j, i) is True, else False"""
        return self.isParent(j, i)

    ####################################################################################################################

    def findParents(self, i):
        """Return the parents of i in adjmat"""
        return [node for node in self.neighbors(i) if self.isParent(node, i)]

    ####################################################################################################################

    def findChildren(self, i):
        """Return the children of i in adjmat"""
        return [node for node in self.neighbors(i) if self.isChild(node, i)]

    ####################################################################################################################

    def findAncestors(self, i):
        """Return the ancestors of i in adjmat"""
        parents = self.findParents(i)
        ancestors = parents
        for parent in parents:
            k = self.findAncestors(parent)
            if len(k) > 0:
                for node in k:
                    if node not in ancestors:
                        ancestors.append(node)
            else:
                continue
        return ancestors

    ####################################################################################################################

    def findDescendants(self, i):
        """Return the descendants of i in adjmat"""
        children = self.findChildren(i)
        descendants = children
        for child in children:
            k = self.findDescendants(child)
            if len(k) > 0:
                for node in k:
                    if node not in descendants:
                        descendants.append(node)
            else:
                continue
        return descendants

    ####################################################################################################################

    def isAncestor(self, i, j):
        """Return True if i is an ancestor of j in adjmat and False otherwise"""
        return j in self.findAncestors(i)

    ####################################################################################################################

    def isDescendant(self, i, j):
        """Return True if isAncestor(j, i) is True, else False"""
        return self.isAncestor(j, i)

    ####################################################################################################################

    def findCondSets(self, i, j):
        """return the list of conditioning sets of the neighbors of i or j in adjmat"""
        neigh_x = self.neighbors(i)
        neigh_y = self.neighbors(j)
        pow_neigh_x = powerset(neigh_x)
        pow_neigh_y = powerset(neigh_y)
        return listUnion(pow_neigh_x, pow_neigh_y)

    ####################################################################################################################

    def findCondSetsWithMid(self, i, j, k):
        """return the list of conditioning sets of the neighbors of i or j in adjmat which contains k"""
        return [S for S in self.findCondSets(i, j) if k in S]

    ####################################################################################################################

    def findCondSetsWithoutMid(self, i, j, k):
        """return the list of conditioning sets of the neighbors of i or j which in adjmat does not contain k"""
        return [S for S in self.findCondSets(i, j) if k not in S]

    ####################################################################################################################

    def isDag(self):
        """Return True if adjmat represents a DAG and False otherwise"""
        assert len(self.nx_graph.nodes) > 0
        return nx.is_directed_acyclic_graph(self.nx_graph)

    ####################################################################################################################

    def isDSep(self, i, j, S):
        """Return True if i and j are d-separated by the set S in nx_graph (networkx.Digraph object)
        and False otherwise. [Throw an error if nx_graph is not a DAG]"""
        assert self.isDag()
        return nx.d_separated(self.nx_graph, {i}, {j}, S)

    ####################################################################################################################

    def hasPath(self, i, j):
        """Return True if there exists a path between i and j, and False otherwise"""
        assert len(self.nx_skel.nodes) > 0
        return nx.has_path(self.nx_skel, i, j)

    ####################################################################################################################

    def findAllPaths(self, i, j):
        """Return the list of paths from i to j in adjmat"""
        assert len(self.nx_skel.nodes) > 0
        return list(nx.all_simple_paths(self.nx_skel, i, j))

    ####################################################################################################################

    def findAllFullyDirectedPaths(self, i, j):
        """Return the list of all fully directed paths from i to j in adjmat"""
        all_paths = self.findAllPaths(i, j)
        all_directed_paths = []
        for path in all_paths:
            directed = True
            for x in range(len(path) - 1):
                if self.isParent(path[x], path[x + 1]):
                    continue
                else:
                    directed = False
                    break
            if directed:
                all_directed_paths.append(path)
        return all_directed_paths

    ####################################################################################################################

    def hasDirectedCycle(self):
        """Return true if adjmat has a directed cycle"""
        if len(self.findBiDirected()) > 0:
            return True
        else:
            nodes = range(len(self.adjmat))
            G = nx.DiGraph()
            self.nx_graph.add_nodes_from(nodes)
            directed = self.findFullyDirected()
            for (i, j) in directed:
                G.add_edge(i, j)
            return nx.is_directed_acyclic_graph(G)

    ####################################################################################################################

    def addDirectedEdge(self, i, j):
        """Add i --> j to adjmat"""
        self.adjmat[i, j] = 1
        self.adjmat[j, i] = 0

    ####################################################################################################################

    def addBiDirectedEdge(self, i, j):
        """Add i <-> j to adjmat"""
        self.adjmat[i, j] = 1
        self.adjmat[j, i] = 1

    ####################################################################################################################

    def addUndirectedEdge(self, i, j):
        """Add i --- j to adjmat"""
        self.adjmat[i, j] = 0
        self.adjmat[j, i] = 0

    ####################################################################################################################

    def removeAdj(self, i, j):
        """Remove the adjacency of i o-o j from adjmat. [Throw an error if i and j are non-adjacent in adjmat]"""
        assert (self.adjmat[i, j] != -1 and self.adjmat[j, i] != -1)
        self.adjmat[i, j] = -1
        self.adjmat[j, i] = -1

    ####################################################################################################################

    def removeOrientation(self, i, j):
        """Remove the orientation of i o-o j from adjmat. [Throw an error if i and j are non-adjacent in adjmat]"""
        assert (self.adjmat[i, j] != -1 and self.adjmat[j, i] != -1)
        self.adjmat[i, j] = 0
        self.adjmat[j, i] = 0

    ####################################################################################################################

    def toNxGraph(self):
        """Convert adjmat into a networkx.Digraph object named nx_graph"""
        nodes = range(len(self.adjmat))
        self.nx_graph.add_nodes_from(nodes)
        undirected = self.findUndirected()
        directed = self.findFullyDirected()
        bidirected = self.findBiDirected()
        for (i, j) in undirected:
            self.nx_graph.add_edge(i, j, color='g')  # Green edge: undirected edge
        for (i, j) in directed:
            self.nx_graph.add_edge(i, j, color='b')  # Blue edge: directed edge
        for (i, j) in bidirected:
            self.nx_graph.add_edge(i, j, color='r')  # Red edge: bidirected edge

    ####################################################################################################################

    def toNxSkeleton(self):
        """Convert adjmat into its skeleton (a networkx.Graph object) named nx_skel"""
        nodes = range(len(self.adjmat))
        self.nx_skel.add_nodes_from(nodes)
        adj = [(i, j) for (i, j) in self.findAdj() if i < j]
        for (i, j) in adj:
            self.nx_skel.add_edge(i, j, color='g')  # Green edge: undirected edge

    ####################################################################################################################

    def drawNxGraph(self, skel=False):
        """Draw nx_graph if skel = False and draw nx_skel otherwise"""
        if not skel:
            print("Green: undirected; Blue: directed; Red: bi-directed\n")
        warnings.filterwarnings("ignore", category=UserWarning)
        g_to_be_drawn = self.nx_skel if skel else self.nx_graph
        edges = g_to_be_drawn.edges()
        colors = [g_to_be_drawn[u][v]['color'] for u, v in edges]
        pos = nx.circular_layout(g_to_be_drawn)
        nx.draw(g_to_be_drawn, pos=pos, with_labels=True, edge_color=colors)
        plt.draw()
        plt.show()

    ####################################################################################################################

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

    ####################################################################################################################

    def nxGraphToAdjmat(self):
        """Convert nx_graph to adjmat"""
        no_of_var = len(self.nx_graph.nodes)
        assert no_of_var > 0
        self.adjmat = np.zeros((no_of_var, no_of_var))
        np.fill_diagonal(self.adjmat, None)
        self.adjmat[self.adjmat == 0] = -1

        for (i, j) in self.nx_graph.edges:
            self.addDirectedEdge(i, j)

    ####################################################################################################################

    def toTetradTxt(self, PATH):
        """Convert adjmat into a text file (readable by TETRAD) output at PATH"""
        directed = self.findFullyDirected()
        undirected = [(i, j) for (i, j) in self.findUndirected() if i < j]
        bidirected = [(i, j) for (i, j) in self.findBiDirected() if i < j]
        file = open(str(PATH), 'w')

        file.write('Graph Nodes: \n')
        node_size = self.adjmat.shape[0]
        for node in range(node_size - 1):
            file.write('X' + str(node + 1) + ';')
        file.write('X' + str(node_size) + '\n')
        file.write('\n')

        file.write('Graph Edges: \n')

        a = iter(range(1, len(directed) + len(undirected) + len(bidirected) + 1))
        for (i, j) in directed:
            file.write(str(next(a)) + '. ' + 'X' + str(i + 1) + ' --> X' + str(j + 1) + '\n')
        for (i, j) in undirected:
            file.write(str(next(a)) + '. ' + 'X' + str(i + 1) + ' --- X' + str(j + 1) + '\n')
        for (i, j) in bidirected:
            file.write(str(next(a)) + '. ' + 'X' + str(i + 1) + ' <-> X' + str(j + 1) + '\n')

        file.close()

    ####################################################################################################################

    def printSummary(self):
        """Print summary of adjmat to the console"""
        print("DAG:", self.isDag())
        print("Number of variables:",
              len([node for node in range(len(self.adjmat)) if not all(np.isnan(self.adjmat[node, :]))]))
        print("Max degree:", self.maxDegree())
        print("Min degree:", self.minDegree())
        print("Average degree:", round(self.avgDegree(), 3))
        print("Density:", round(self.density(), 3))
        print("Number of edges:", int(len(self.findAdj()) / 2))
        print("Number of directed edges:", len(self.findFullyDirected()))
        print("Number of undirected edges:", int(len(self.findUndirected()) / 2))
        print("Number of bi-directed edges:", int(len(self.findBiDirected()) / 2))
        print("PC elapsed time (in seconds):", round(self.PC_elapsed, 3), "\n")

    ####################################################################################################################

    def comparison(self, truth, compare_pattern=True, adj_only=False, uc_also=True, print_to_console=True):
        """Return the performance statistics by comparing adjmat with the true CausalGraph named truth
        :param truth: the true DAG (a CausalGraph object)
        :param compare_pattern: compare the CausalGraph with the true pattern if True \
        and compare with the true DAG otherwise (default = True)
        :param adj_only: return only adjacency-related performance statistics if True (default = False)
        :param uc_also: return unshielded colliders-related performance statistics if True (default = True)
        :param print_to_console: print performance statistics to console if True (default = True)
        :return: a list of performance statistics including:
        0: Adjacency precision
        1: Adjacency recall
        2: Adjacency F1-score
        3: Arrowhead precision
        4: Arrowhead recall
        5: Arrowhead F1-score
        6: Arrowhead precision (per common edges)
        7: Arrowhead recall (per common edges)
        8: Arrowhead F1-score (per common edges)
        9: Unshielded collider precision
        10: Unshielded collider recall
        11: Unshielded collider F1-score
        12: Unshielded collider precision (per common edges)
        13: Unshielded collider recall (per common edges)
        14: Unshielded collider F1-score (per common edges)
        """
        str_list = ["Adjacency precision:", "Adjacency recall:", "Adjacency F1-score:",
                    "Arrowhead precision:", "Arrowhead recall:", "Arrowhead F1-score:",
                    "Arrowhead precision (per common edges):", "Arrowhead recall (per common edges):",
                    "Arrowhead F1-score (per common edges):",
                    "Unshielded colliders precision:", "Unshielded colliders recall:", "Unshielded colliders F1-score:",
                    "Unshielded colliders precision (per common edges):",
                    "Unshielded colliders recall (per common edges):",
                    "Unshielded colliders F1-score (per common edges):"]

        if compare_pattern:
            truth = toPattern(truth)

        # Adjacency-related performance statistics
        truth_adj = [(i, j) for (i, j) in truth.findAdj() if i < j]
        est_adj = [(i, j) for (i, j) in self.findAdj() if i < j]

        adj_positive = len(est_adj)
        adj_true_positive = len(listIntersection(truth_adj, est_adj))
        adj_false_negative = len(listMinus(truth_adj, est_adj))

        AP = adj_true_positive / adj_positive if adj_positive != 0 else np.nan
        AR = adj_true_positive / (adj_true_positive + adj_false_negative) if not (
                adj_true_positive == 0 and adj_false_negative == 0) else np.nan
        AF1 = np.nan
        if (not np.isnan(AP)) and (not np.isnan(AR)):
            if AP + AR != 0:
                AF1 = (2 * AP * AR) / (AP + AR)

        stat_list = [AP, AR, AF1]

        if not adj_only:
            # Arrowhead-related performance statistics

            truth_ah = truth.findArrowheads()
            est_ah = self.findArrowheads()

            ah_positive_n = len(est_ah)
            ah_true_positive = listIntersection(truth_ah, est_ah)
            ah_false_negative = listMinus(truth_ah, est_ah)
            ah_true_positive_n = len(ah_true_positive)
            ah_false_negative_n = len(ah_false_negative)

            AHP = ah_true_positive_n / ah_positive_n if ah_positive_n != 0 else np.nan
            AHR = ah_true_positive_n / (ah_true_positive_n + ah_false_negative_n) if not (
                    ah_true_positive_n == 0 and ah_false_negative_n == 0) else np.nan
            AHF1 = np.nan
            if (not np.isnan(AHP)) and (not np.isnan(AHR)):
                if AHP + AHR != 0:
                    AHF1 = (2 * AHP * AHR) / (AHP + AHR)

            ah_false_positive = listMinus(est_ah, truth_ah)
            ah_false_positive_common_n = len(listIntersection(ah_false_positive, truth.findAdj()))
            ah_false_negative_common_n = len(listIntersection(ah_false_negative, self.findAdj()))

            AHPC = ah_true_positive_n / (ah_true_positive_n + ah_false_positive_common_n) if not (
                    ah_true_positive_n == 0 and ah_false_positive_common_n == 0) else np.nan
            AHRC = ah_true_positive_n / (ah_true_positive_n + ah_false_negative_common_n) if not (
                    ah_true_positive_n == 0 and ah_false_negative_common_n == 0) else np.nan
            AHF1C = np.nan
            if (not np.isnan(AHPC)) and (not np.isnan(AHRC)):
                if AHPC + AHRC != 0:
                    AHF1C = (2 * AHPC * AHRC) / (AHPC + AHRC)

            for AH_stat in [AHP, AHR, AHF1, AHPC, AHRC, AHF1C]:
                stat_list.append(AH_stat)

            if uc_also:
                # Unshielded colliders-related performance statistics

                truth_uc = truth.findUC()
                est_uc = self.findUC()

                uc_positive_n = len(est_uc)
                uc_true_positive = listIntersection(truth_uc, est_uc)
                uc_false_negative = listMinus(truth_uc, est_uc)
                uc_true_positive_n = len(uc_true_positive)
                uc_false_negative_n = len(uc_false_negative)

                UCP = uc_true_positive_n / uc_positive_n if uc_positive_n != 0 else np.nan
                UCR = uc_true_positive_n / (uc_true_positive_n + uc_false_negative_n) if not (
                        uc_true_positive_n == 0 and uc_false_negative_n == 0) else np.nan
                UCF1 = np.nan
                if (not np.isnan(UCP)) and (not np.isnan(UCR)):
                    if UCP + UCR != 0:
                        UCF1 = (2 * UCP * UCR) / (UCP + UCR)

                uc_false_positive = listMinus(est_uc, truth_uc)

                uc_false_positive_common_n = len(listIntersection(uc_false_positive, truth.findUnshieldedTriples()))
                uc_false_negative_common_n = len(listIntersection(uc_false_negative, self.findUnshieldedTriples()))

                UCPC = uc_true_positive_n / (uc_true_positive_n + uc_false_positive_common_n) if not (
                        uc_true_positive_n == 0 and uc_false_positive_common_n == 0) else np.nan
                UCRC = uc_true_positive_n / (uc_true_positive_n + uc_false_negative_common_n) if not (
                        uc_true_positive_n == 0 and uc_false_negative_common_n == 0) else np.nan
                UCF1C = np.nan
                if (not np.isnan(UCPC)) and (not np.isnan(UCRC)):
                    if UCPC + UCRC != 0:
                        UCF1C = (2 * UCPC * UCRC) / (UCPC + UCRC)

                for UC_stat in [UCP, UCR, UCF1, UCPC, UCRC, UCF1C]:
                    stat_list.append(UC_stat)

        if print_to_console:
            for stat_index in range(len(stat_list)):
                print(str_list[stat_index],
                      round(stat_list[stat_index], 3) if not np.isnan(stat_list[stat_index]) else None)
            print("\n")

        return stat_list


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def tetradToCausalGraph(path):
    """Convert the graph (.txt output by TETRAD) at path into a CausalGraph object"""
    tetrad_file = pd.read_csv(path, sep='\t')
    var_names = str()
    if ',' in str(tetrad_file.loc[0][0]):
        var_names = str(tetrad_file.loc[0][0]).split(',')
    elif ';' in str(tetrad_file.loc[0][0]):
        var_names = str(tetrad_file.loc[0][0]).split(';')

    cg = CausalGraph(len(var_names))
    cg.adjmat[cg.adjmat == 0] = -1

    bidirected = 0

    for i in range(2, tetrad_file.shape[0]):
        STR = str(tetrad_file.loc[i][0])
        if '-->' in STR:
            STR_truncated = STR.split('. ')[1].split(' --> ')
            LEFT = int(STR_truncated[0].split('X')[1]) - 1
            RIGHT = int(STR_truncated[1].split('X')[1]) - 1
            if cg.adjmat[LEFT, RIGHT] != -1 and cg.adjmat[RIGHT, LEFT] != -1:
                if cg.adjmat[LEFT, RIGHT] != 1 or cg.adjmat[RIGHT, LEFT] != 0:
                    raise ValueError("Inconsistency detected. Check the source file on", STR_truncated[0], "and",
                                     STR_truncated[1], ".")
            else:
                cg.adjmat[LEFT, RIGHT] = 1
                cg.adjmat[RIGHT, LEFT] = 0

        elif '---' in STR:
            STR_truncated = STR.split('. ')[1].split(' --- ')
            LEFT = int(STR_truncated[0].split('X')[1]) - 1
            RIGHT = int(STR_truncated[1].split('X')[1]) - 1
            if cg.adjmat[LEFT, RIGHT] != -1 and cg.adjmat[RIGHT, LEFT] != -1:
                if cg.adjmat[LEFT, RIGHT] != 0 or cg.adjmat[RIGHT, LEFT] != 0:
                    raise ValueError("Inconsistency detected. Check the source file on", STR_truncated[0], "and",
                                     STR_truncated[1], ".")
            else:
                cg.adjmat[LEFT, RIGHT] = 0
                cg.adjmat[RIGHT, LEFT] = 0

        elif '<->' in STR:
            bidirected += 1
            STR_truncated = STR.split('. ')[1].split(' <-> ')
            LEFT = int(STR_truncated[0].split('X')[1]) - 1
            RIGHT = int(STR_truncated[1].split('X')[1]) - 1
            if cg.adjmat[LEFT, RIGHT] != -1 and cg.adjmat[RIGHT, LEFT] != -1:
                if cg.adjmat[LEFT, RIGHT] != 1 or cg.adjmat[RIGHT, LEFT] != 1:
                    raise ValueError("Inconsistency detected. Check the source file on", STR_truncated[0], "and",
                                     STR_truncated[1], ".")
            else:
                cg.adjmat[(LEFT, RIGHT)] = 1
                cg.adjmat[(RIGHT, LEFT)] = 1

    if bidirected > 0:
        print("The source file contains", bidirected, "bi-directed edges.")

    cg.toNxGraph()
    cg.toNxSkeleton()

    return cg


#######################################################################################################################

def toPattern(cg):
    """Convert cg (a Causal Graph object) to its pattern [Throw an error if cg.nx_graph is not a DAG]"""
    assert cg.isDag()
    cg_pattern = deepcopy(cg)
    cg_pattern.adjmat[cg_pattern.adjmat == 1] = 0  # remove all arrowheads to obtain the skeleton

    UC = cg.findUC()
    for (i, j, k) in UC:
        cg_pattern.adjmat[i, j] = 1
        cg_pattern.adjmat[k, j] = 1

    UT = cg.findUnshieldedTriples()
    Tri = cg.findTriangles()
    Kites = cg.findKites()

    Loop = True
    while Loop:
        Loop = False
        for (i, j, k) in UT:
            if cg.isFullyDirected(i, j) and cg.isUndirected(j, k):
                cg_pattern.adjmat[j, k] = 1
                Loop = True

        for (i, j, k) in Tri:
            if cg.isFullyDirected(i, j) and cg.isFullyDirected(j, k) and cg.isUndirected(i, k):
                cg_pattern.adjmat[i, k] = 1
                Loop = True

        for (i, j, k, l) in Kites:
            if cg.isUndirected(i, j) and cg.isUndirected(i, k) and cg.isFullyDirected(j, l) \
                    and cg.isFullyDirected(k, l) and cg.isUndirected(i, l):
                cg_pattern.adjmat[i, l] = 1
                Loop = True

    cg_pattern.toNxGraph()
    cg_pattern.toNxSkeleton()

    return cg_pattern


#######################################################################################################################

def subgraph(cg, list_of_nodes):
    """Create a CausalGraph object from cg by only including nodes in list_of_nodes (list)"""
    sub_cg = deepcopy(cg)
    redundant_nodes = listMinus(range(len(cg.adjmat)), list_of_nodes)
    for i in redundant_nodes:
        sub_cg.adjmat[i, :] = None
        sub_cg.adjmat[:, i] = None
    sub_cg.nx_graph.remove_nodes_from(redundant_nodes)
    sub_cg.nx_skel.remove_nodes_from(redundant_nodes)
    sub_cg.redundant_nodes = redundant_nodes
    sub_cg.PC_elapsed = -1
    return sub_cg

#######################################################################################################################
