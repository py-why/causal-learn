from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.NodeType import NodeType
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma
from causallearn.utils.KCI.KCI import KCI_UInd

import causallearn.search.HiddenCausal.GIN.Infer_Causal_Order as ICO
import causallearn.search.HiddenCausal.GIN.Identifying_Causal_Clusters as ICC



def GIN(data, indep_test_method='kerpy', alpha=0.05):

    if indep_test_method not in ['kci', 'hsic', 'kerpy']:
        raise NotImplementedError((f"Independent test method {indep_test_method} is not implemented."))

    if indep_test_method == 'kci':
        kci = KCI_UInd()

    if indep_test_method == 'kerpy':
        from kerpy.GaussianKernel import GaussianKernel
        from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
        kernelX = GaussianKernel(float(0.1))
        kernelY = GaussianKernel(float(0.1))

        num_samples = data.shape[0]

        myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=20, num_rfy=20, num_nullsims=500)


    if indep_test_method == 'kci':
        def indep_test(x, y):
            return kci.compute_pvalue(x, y)[0]
    elif indep_test_method == 'hsic':
        def indep_test(x, y):
            return hsic_test_gamma(x, y)[1]
    elif indep_test_method == 'kerpy':
        def indep_test(x, y):
            return myspectralobject.compute_pvalue(x, y)
    else:
        raise NotImplementedError((f"Independent test method {indep_test_method} is not implemented."))


    #identifying causal cluster by using fast HSIC, set Signification Level(alhpa) =0.05
    Cluster=ICC.FindCluser(data,indep_test,alpha)
    #identifying causal order by using mutual information, (k nearest neighbors (method='1') or sklearn package (method='2')
    CausalOrder=ICO.LearnCausalOrder(Cluster,data,method='1')
    

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
