'''
    File name: GIN.py
    Discription: Learning Hidden Causal Representation with GIN condition
    Author: ZhiyiHuang@DMIRLab, RuichuCai@DMIRLab
    Form DMIRLab: https://dmir.gdut.edu.cn/
'''

import numpy as np
from itertools import combinations

from graph.NodeType import NodeType
from graph.GraphNode import GraphNode
from graph.GeneralGraph import GeneralGraph
from ..independence import hsic_test_gamma, mutual_information
from ..utils import merge_overlaping_cluster


def cal_dep_for_gin(data, cov, X, Z, indep_func):
    cov_m = cov[np.ix_(Z, X)]
    _, _, v = np.linalg.svd(cov_m)
    omega = v.T[:, -1]
    e_xz = np.dot(omega, data[:, X].T)

    sta = 0
    for i in Z:
        if indep_func == "MI":
            sta += mutual_information(e_xz, data[:, i])
        else:
            sta += hsic_test_gamma(e_xz, data[:, i])[0]
    sta /= len(Z)
    return sta


def find_root(data, cov, clusters, K, indep_func):
    if len(clusters) == 1:
        return clusters[0]
    root = clusters[0]
    dep_statistic_score = 1e30
    for i in clusters:
        for j in clusters:
            if i == j:
                continue
            X = [i[0], j[0]]
            Z = []
            for k in range(1, len(i)):
                Z.append(i[k])

            if K:
                for k in K:
                    X.append(k[0])
                    Z.append(k[1])

            dep_statistic = cal_dep_for_gin(data, cov, X, Z, indep_func=indep_func)
            if dep_statistic < dep_statistic_score:
                dep_statistic_score = dep_statistic
                root = i

    return root


def GIN(data, indep_func='HSIC'):
    v_labels = list(range(data.shape[1]))
    v_set = set(v_labels)
    cov = np.cov(data.T)

    # Step 1: Finding Causal Clusters
    cluster_list = []
    min_cluster = {i: set() for i in v_set}
    min_dep_score = {i: 1e9 for i in v_set}
    for (x1, x2) in combinations(v_set, 2):
        x_set = {x1, x2}
        z_set = v_set - x_set
        dep_statistic = cal_dep_for_gin(data, cov, list(x_set), list(z_set), indep_func=indep_func)
        for i in x_set:
            if min_dep_score[i] > dep_statistic:
                min_dep_score[i] = dep_statistic
                min_cluster[i] = x_set
    for i in v_labels:
        cluster_list.append(list(min_cluster[i]))

    cluster_list = merge_overlaping_cluster(cluster_list)

    # Step 2: Learning the Causal Order of Latent Variables
    K = []
    while (len(cluster_list) != 0):
        root = find_root(data, cov, cluster_list, K, indep_func=indep_func)
        K.append(root)
        cluster_list.remove(root)

    latent_id = 1
    l_nodes = []
    G = GeneralGraph([])
    for cluster in K:
        l_node = GraphNode(f"L{latent_id}")
        l_node.set_node_type(NodeType.LATENT)
        l_nodes.append(l_node)
        G.add_node(l_node)
        for l in l_nodes:
            if l != l_node:
                G.add_directed_edge(l, l_node)
        for o in cluster:
            o_node = GraphNode(f"X{o+1}")
            G.add_node(o_node)
            G.add_directed_edge(l_node, o_node)
        latent_id += 1

    return G, K
