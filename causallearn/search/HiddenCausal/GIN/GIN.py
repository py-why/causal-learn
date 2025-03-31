from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.NodeType import NodeType
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma
from causallearn.utils.KCI.KCI import KCI_UInd

import causallearn.search.HiddenCausal.GIN.Infer_Causal_Order as ICO
import causallearn.search.HiddenCausal.GIN.Identifying_Causal_Clusters as ICC



def GIN(data, indep_test_method='kci', alpha=0.05, MAX_Factor = 2):
    '''
    Learning causal structure of Latent Variables for Linear Non-Gaussian Latent Variable Model
    with Generalized Independent Noise Condition
    Parameters
    ----------
    data : numpy ndarray
           data set
    indep_test_method : str, default='kci'
        the name of the independence test being used
    alpha : float, default=0.05
        desired significance level of independence tests (p_value) in (0,1)
    MAX_Factor : int
        Prior maximum factor number
    Returns
    -------
    G : general graph
        causal graph
    causal_order : list
        causal order
    '''
    if indep_test_method not in ['kci', 'hsic']:
        raise NotImplementedError((f"Independent test method {indep_test_method} is not implemented."))

    if indep_test_method == 'kci':
        kci = KCI_UInd()

    if indep_test_method == 'kci':
        def indep_test(x, y):
            return kci.compute_pvalue(x, y)[0]
    elif indep_test_method == 'hsic':
        def indep_test(x, y):
            return hsic_test_gamma(x, y)[1]
    else:
        raise NotImplementedError((f"Independent test method {indep_test_method} is not implemented."))


    '''identifying causal cluster by using HSIC (KCI)'''
    Cluster=ICC.FindCluser(data, indep_test, alpha, MAX_Factor)
    '''identifying causal order by using mutual information, (k nearest neighbors )'''
    CausalOrder=ICO.LearnCausalOrder(Cluster, data)


    latent_id = 1
    l_nodes = []
    G = GeneralGraph([])
    for cluster in CausalOrder:
        l_node = GraphNode(f"L{latent_id}")
        l_node.set_node_type(NodeType.LATENT)
        l_nodes.append(l_node)
        G.add_node(l_node)
        for l in l_nodes:
            if l != l_node:
                G.add_directed_edge(l, l_node)
        for o in cluster:
            o_node = GraphNode(f"X{o + 1}")
            G.add_node(o_node)
            G.add_directed_edge(l_node, o_node)
        latent_id += 1

    return G, CausalOrder



