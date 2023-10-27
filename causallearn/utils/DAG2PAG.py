from itertools import combinations, permutations
from typing import List

import numpy as np
import networkx as nx
from networkx.algorithms import d_separated

from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Node import Node
from causallearn.search.ConstraintBased.FCI import rule0, rulesR1R2cycle, ruleR3, ruleR4B
from causallearn.utils.cit import CIT, d_separation

def dag2pag(dag: Dag, islatent: List[Node]) -> GeneralGraph:
    """
    Convert a DAG to its corresponding PAG
    Parameters
    ----------
    dag : Direct Acyclic Graph
    islatent: the indexes of latent variables. [] means there is no latent variable
    Returns
    -------
    PAG : Partial Ancestral Graph
    """
    dg = nx.DiGraph()
    true_dag = nx.DiGraph()
    nodes = dag.get_nodes()
    observed_nodes = list(set(nodes) - set(islatent))
    mod_nodes = observed_nodes + islatent
    nodes = dag.get_nodes()
    nodes_ids = {node: i for i, node in enumerate(nodes)}
    mod_nodeids = {node: i for i, node in enumerate(mod_nodes)}

    n = len(nodes)
    dg.add_nodes_from(range(n))
    true_dag.add_nodes_from(range(n))

    for x, y in combinations(range(n), 2):
        edge = dag.get_edge(nodes[x], nodes[y])
        if edge:
            if edge.get_endpoint2() == Endpoint.ARROW:
                dg.add_edge(nodes_ids[edge.get_node1()], nodes_ids[edge.get_node2()])
                true_dag.add_edge(mod_nodeids[edge.get_node1()], mod_nodeids[edge.get_node2()])
            else:
                dg.add_edge(nodes_ids[edge.get_node2()], nodes_ids[edge.get_node1()])
                true_dag.add_edge(mod_nodeids[edge.get_node1()], mod_nodeids[edge.get_node2()])


    PAG = GeneralGraph(observed_nodes)
    for nodex, nodey in combinations(observed_nodes, 2):
        edge = Edge(nodex, nodey, Endpoint.CIRCLE, Endpoint.CIRCLE)
        edge.set_endpoint1(Endpoint.CIRCLE)
        edge.set_endpoint2(Endpoint.CIRCLE)
        PAG.add_edge(edge)

    sepset = {(nodes_ids[nodex], nodes_ids[nodey]): set() for nodex, nodey in permutations(observed_nodes, 2)}

    for l in range(0, len(observed_nodes) - 1):
        for nodex, nodey in combinations(observed_nodes, 2):
            edge = PAG.get_edge(nodex, nodey)
            if not edge:
                continue
            for Z in combinations(observed_nodes, l):
                if nodex in Z or nodey in Z:
                    continue
                if d_separated(dg, {nodes_ids[nodex]}, {nodes_ids[nodey]}, set(nodes_ids[z] for z in Z)):
                    if edge:
                        PAG.remove_edge(edge)
                    sepset[(nodes_ids[nodex], nodes_ids[nodey])] |= set(Z)
                    sepset[(nodes_ids[nodey], nodes_ids[nodex])] |= set(Z)

    for nodex, nodey in combinations(observed_nodes, 2):
        if PAG.get_edge(nodex, nodey):
            continue
        for nodez in observed_nodes:
            if nodez == nodex:
                continue
            if nodez == nodey:
                continue
            if nodez not in sepset[(nodes_ids[nodex], nodes_ids[nodey])]:
                edge_xz = PAG.get_edge(nodex, nodez)
                edge_yz = PAG.get_edge(nodey, nodez)
                if edge_xz and edge_yz:
                    PAG.remove_edge(edge_xz)
                    mod_endpoint(edge_xz, nodez, Endpoint.ARROW)
                    PAG.add_edge(edge_xz)

                    PAG.remove_edge(edge_yz)
                    mod_endpoint(edge_yz, nodez, Endpoint.ARROW)
                    PAG.add_edge(edge_yz)

    print()
    change_flag = True

    data = np.empty(shape=(0, len(observed_nodes)))
    independence_test_method = CIT(data, method=d_separation, true_dag=true_dag)
    node_map = PAG.get_node_map()
    sepset_reindexed = {(node_map[nodes[i]], node_map[nodes[j]]): sepset[(i, j)] for (i, j) in sepset}
    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(PAG, None, change_flag, False)
        change_flag = ruleR3(PAG, sepset_reindexed, None, change_flag, False)
        change_flag = ruleR4B(PAG, -1, data, independence_test_method, 0.05, sep_sets=sepset_reindexed,
                          change_flag=change_flag,
                          bk=None, verbose=False)
    return PAG


def is_fully_directed(edge: Edge) -> bool:
    if edge:
        if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            return True
    return False


def is_endpoint(edge: Edge, z: Node, end: Endpoint) -> bool:
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


def mod_endpoint(edge: Edge, z: Node, end: Endpoint):
    if edge.get_node1() == z:
        edge.set_endpoint1(end)
    elif edge.get_node2() == z:
        edge.set_endpoint2(end)
    else:
        raise ValueError("z not in edge")
