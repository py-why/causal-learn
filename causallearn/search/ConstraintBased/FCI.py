from copy import deepcopy
from itertools import combinations, permutations

import networkx as nx
import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import fisherz


def trans_nodeset2str(Z):
    return tuple(str(z) for z in Z)


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


# Zhang, Jiji. "A characterization of markov equivalence classes for directed acyclic graphs with latent variables."
def rule1(G):
    for x, y, z in permutations(G.get_nodes(), 3):
        if G.get_edge(x, z):
            continue
        edgexy = G.get_edge(x, y)
        edgeyz = G.get_edge(y, z)
        if not edgexy or not edgeyz:
            continue

        if is_endpoint(edgexy, y, Endpoint.ARROW) and is_endpoint(edgeyz, y, Endpoint.CIRCLE):
            G.remove_edge(edgeyz)
            mod_endpoint(edgeyz, y, Endpoint.TAIL)
            mod_endpoint(edgeyz, z, Endpoint.ARROW)
            print(f"Orient {y} --> {z}")
            G.add_edge(edgeyz)


def rule2(G):
    for x, y, z in permutations(G.get_nodes(), 3):
        if not G.get_edge(x, y) or not G.get_edge(x, z) or not G.get_edge(y, z):
            continue
        edgexy = G.get_edge(x, y)
        edgexz = G.get_edge(x, z)
        edgeyz = G.get_edge(y, z)

        if (is_fully_directed(edgexy) and is_endpoint(edgeyz, z, Endpoint.ARROW)) or \
                (is_fully_directed(edgeyz) and is_endpoint(edgexy, y, Endpoint.ARROW)):
            if is_endpoint(edgexz, z, Endpoint.CIRCLE):
                G.remove_edge(edgexz)
                mod_endpoint(edgexz, z, Endpoint.ARROW)
                print(f"Orient {x} *-> {z}")
                G.add_edge(edgexz)


def rule3(G):
    for x, y, z, w in permutations(G.get_nodes(), 4):
        edgexy = G.get_edge(x, y)
        edgexz = G.get_edge(x, z)
        edgexw = G.get_edge(x, w)
        edgeyz = G.get_edge(y, z)
        edgeyw = G.get_edge(y, w)
        edgezw = G.get_edge(z, w)

        if edgexz:
            continue
        cont = False

        for edge in [edgexy, edgexw, edgeyz, edgeyw, edgezw]:
            if not edge:
                cont = True
                break

        if cont:
            continue

        chain1 = is_endpoint(edgexy, y, Endpoint.ARROW) and is_endpoint(edgeyz, y, Endpoint.ARROW)
        chain2 = is_endpoint(edgexw, w, Endpoint.CIRCLE) and is_endpoint(edgezw, w, Endpoint.CIRCLE)
        if chain1 and chain2 and is_endpoint(edgeyw, y, Endpoint.CIRCLE):
            G.remove_edge(edgeyw)
            mod_endpoint(edgeyw, y, Endpoint.ARROW)
            print(f"Orient {w} *-> {y}")
            G.add_edge(edgeyw)


def rule4(G, sepset):
    nxG = nx.Graph()
    nodes = G.get_nodes()
    nodes_ids = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    for i in range(n):
        nxG.add_node(i)
    for edge in G.get_graph_edges():
        nxG.add_edge(nodes_ids[edge.get_node1()], nodes_ids[edge.get_node2()])

    nodes = G.get_nodes()

    def has_directed_edge(prev, next):
        edge = G.get_edge(nodes[prev], nodes[next])
        if edge:
            if is_endpoint(edge, nodes[next], Endpoint.ARROW):
                return True
            else:
                return False
        else:
            return False

    def get_discriminating_paths(u, v, b):
        all_paths = nx.all_simple_paths(nxG, u, v)
        discpaths = []
        for path in all_paths:
            if b in path:
                b_pred = (path.index(v) - path.index(b)) == 1 and nxG.has_edge(b, v)
                all_colliders = True
                for node in path[1:-1]:
                    prev = path[path.index(node) - 1]
                    suc = path[path.index(node) + 1]
                    if (node != b) and not ((has_directed_edge(prev, node)) and (has_directed_edge(suc, node))):
                        all_colliders = False
                all_pred = True
                for node in path[1:-2]:
                    if not (has_directed_edge(node, v)):
                        all_pred = False
                nonadj = not nxG.has_edge(u, v)
                if (b_pred and all_colliders and all_pred and nonadj):
                    discpaths.append(path)
        return discpaths

    for x, y, z, w in permutations(nodes, 4):
        paths = get_discriminating_paths(nodes_ids[z], nodes_ids[w], nodes_ids[y])
        for path in paths:
            if nodes_ids[x] in path:
                edge_yz = G.get_edge(y, z)
                if not edge_yz:
                    continue
                if path.index(nodes_ids[x]) == len(path) - 3 and is_endpoint(edge_yz, y, Endpoint.CIRCLE):
                    if y in sepset[(z, w)]:
                        G.remove_edge(edge_yz)
                        mod_endpoint(edge_yz, y, Endpoint.TAIL)
                        mod_endpoint(edge_yz, z, Endpoint.ARROW)
                        G.add_edge(edge_yz)
                        print(f"Orient {y} --> {z}")
                    else:
                        edge_xy = G.get_edge(x, y)
                        if edge_xy:
                            G.remove_edge(edge_xy)
                            mod_endpoint(edge_xy, x, Endpoint.ARROW)
                            mod_endpoint(edge_xy, y, Endpoint.ARROW)
                            G.add_edge(edge_xy)
                            print(f"Orient {x} <-> {y}")
                        G.remove_edge(edge_yz)
                        mod_endpoint(edge_yz, y, Endpoint.ARROW)
                        mod_endpoint(edge_yz, z, Endpoint.ARROW)
                        G.add_edge(edge_yz)
                        print(f"Orient {y} <-> {z}")


def fci(data, indep_test=fisherz, alpha=0.05, verbose=False):
    '''
    Causal Discovery with Fast Causal Inference

    Parameters
    ----------
    data : Input data matrix
    indep_test : Independence test method function
    alpha : Significance level of individual partial correlation tests
    verbose: 0 - no output, 1 - detailed output

    Returns
    -------
    G : Causal graph
    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    # assert test_name in ["Fisher_Z", "Chi_sq", "G_sq", "KCI"]

    no_of_var = data.shape[1]

    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # corr_mat = None
    # if test_name == "Fisher_Z":
    #     corr_mat = np.corrcoef(data.T)
    #
    # kci_uind = None
    # kci_cind = None
    # if test_name == "KCI":
    #     kci_uind = KCI_UInd(sample_size=data.shape[0], kernelX=GaussianKernel(), kernelY=GaussianKernel(), approx=False)
    #     kci_cind = KCI_CInd(sample_size=data.shape[0], kernelX=GaussianKernel(), kernelY=GaussianKernel(),
    #                         kernelZ=GaussianKernel(), approx=False)

    nodes = []
    for i in range(no_of_var):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
    G = GeneralGraph(nodes)

    sepset = {(nodex, nodey): set() for nodex, nodey in permutations(nodes, 2)}

    for x, y in combinations(nodes, 2):
        edge = Edge(x, y, Endpoint.CIRCLE, Endpoint.CIRCLE)
        edge.set_endpoint1(Endpoint.CIRCLE)
        edge.set_endpoint2(Endpoint.CIRCLE)
        G.add_edge(edge)

    depth = 0
    while depth <= no_of_var - 2:
        for x, y in permutations(nodes, 2):
            adj_x = G.get_adjacent_nodes(x)
            if y in adj_x:
                adj_x.remove(y)
            else:
                continue

            if len(adj_x) < depth:
                continue
            for Z in combinations(adj_x, depth):
                S = tuple([i["id"] for i in Z])
                pval = indep_test(data, x["id"], y["id"], S)
                # if test_name == "Fisher_Z":
                #     pval = fisherZ(corr_mat, x["id"], y["id"], S, data.shape[0])
                # elif test_name == "Chi_sq":
                #     pval = chisq(data, x["id"], y["id"], S, G_sq=False)
                # elif test_name == "G_sq":
                #     pval = chisq(data, x["id"], y["id"], S, G_sq=True)
                # elif test_name == "KCI":
                #     if len(S) == 0:
                #         pval = kci_uind.compute_pvalue(data[:, x["id"]], data[:, y["id"]])
                #     else:
                #         pval = kci_cind.compute_pvalue(data[:, x["id"]], data[:, y["id"]], data[:, S])
                # else:
                #     raise NotImplementedError()
                if pval > alpha:
                    if verbose:
                        print(f"phase1 remove {x} --- {y} by sepset {trans_nodeset2str(Z)} pval:{pval}")
                    edge = G.get_edge(x, y)
                    if edge is not None:
                        G.remove_edge(edge)
                    sepset[(x, y)] |= set(Z)
                    sepset[(y, x)] |= set(Z)
                    break

        depth += 1

    for x, y in combinations(nodes, 2):
        for z in nodes:
            if x == z or y == z:
                continue
            if (not G.get_edge(x, y)) and G.get_edge(x, z) and G.get_edge(y, z) and z not in sepset[(x, y)]:
                edge_xz = G.get_edge(x, z)
                G.remove_edge(edge_xz)
                mod_endpoint(edge_xz, z, Endpoint.ARROW)
                G.add_edge(edge_xz)

                edge_yz = G.get_edge(y, z)
                G.remove_edge(edge_yz)
                mod_endpoint(edge_yz, z, Endpoint.ARROW)
                G.add_edge(edge_yz)

    nxG = nx.Graph()

    for node in G.get_nodes():
        nxG.add_node(node["id"])

    for edge in G.get_graph_edges():
        nxG.add_edge(edge.get_node1()["id"], edge.get_node2()["id"])

    def is_possible_d_sep(nxG, x, y):
        all_paths = nx.all_simple_paths(nxG, x, y)
        for path in all_paths:
            path_sep = True
            for i in range(1, len(path) - 1):
                is_collider = is_endpoint(G.get_edge(nodes[path[i - 1]], nodes[path[i]]), nodes[path[i]],
                                          Endpoint.ARROW) and \
                              is_endpoint(G.get_edge(nodes[path[i + 1]], nodes[path[i]]), nodes[path[i]],
                                          Endpoint.ARROW)
                is_triangle = G.get_edge(nodes[path[i - 1]], nodes[path[i + 1]]) is not None
                if not (is_collider or is_triangle):
                    path_sep = False
            if path_sep:
                return True
        return False

    pdsepset = {node: set() for node in nodes}

    for nodex, nodey in combinations(nodes, 2):
        if is_possible_d_sep(nxG, nodex["id"], nodey["id"]):
            pdsepset[nodex] |= {nodey}
            pdsepset[nodey] |= {nodex}

    # Phase 2
    # G = GeneralGraph(nodes)
    #
    # for x, y in combinations(nodes, 2):
    #     edge = Edge(x, y, Endpoint.CIRCLE, Endpoint.CIRCLE)
    #     edge.set_endpoint1(Endpoint.CIRCLE)
    #     edge.set_endpoint2(Endpoint.CIRCLE)
    #     G.add_edge(edge)

    depth = 0
    while depth <= no_of_var - 2:
        for x, y in permutations(nodes, 2):
            if not G.get_edge(x, y):
                continue

            pdseps = deepcopy(pdsepset[x])
            if y in pdseps:
                pdseps.remove(y)
            pdseps |= sepset[(x, y)]
            if len(pdseps) < depth:
                continue
            for Z in combinations(pdseps, depth):
                S = tuple([i["id"] for i in Z])
                # pval = 0.0
                pval = indep_test(data, x["id"], y["id"], S)
                # if test_name == "Fisher_Z":
                #     pval = fisherZ(corr_mat, x["id"], y["id"], S, data.shape[0])
                # elif test_name == "Chi_sq":
                #     pval = chisq(data, x["id"], y["id"], S, G_sq=False)
                # elif test_name == "G_sq":
                #     pval = chisq(data, x["id"], y["id"], S, G_sq=True)
                # elif test_name == "KCI":
                #     if len(S) == 0:
                #         pval = kci_uind.compute_pvalue(data[:, x["id"]], data[:, y["id"]])
                #     else:
                #         pval = kci_cind.compute_pvalue(data[:, x["id"]], data[:, y["id"]], data[:, S])
                # else:
                #     raise NotImplementedError()
                if pval >= alpha:
                    if verbose:
                        print(f"phase2 remove {x} --- {y} by sepset {trans_nodeset2str(Z)} pval:{pval}")
                    edge = G.get_edge(x, y)
                    if edge is not None:
                        G.remove_edge(edge)
                    sepset[(x, y)] |= set(Z)
                    sepset[(y, x)] |= set(Z)
                    break

        depth += 1

    for x, y in combinations(nodes, 2):
        for z in nodes:
            if x == z or y == z:
                continue
            if (not G.get_edge(x, y)) and G.get_edge(x, z) and G.get_edge(y, z) and (z not in sepset[(x, y)]):
                print(f"orient_V: {x},{z},{y}")
                edge_xz = G.get_edge(x, z)
                G.remove_edge(edge_xz)
                mod_endpoint(edge_xz, z, Endpoint.ARROW)
                G.add_edge(edge_xz)

                edge_yz = G.get_edge(y, z)
                G.remove_edge(edge_yz)
                mod_endpoint(edge_yz, z, Endpoint.ARROW)
                G.add_edge(edge_yz)

    rule1(G)
    rule2(G)
    rule3(G)
    rule4(G, sepset)
    return G
