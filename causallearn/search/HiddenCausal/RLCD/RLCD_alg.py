import pandas as pd
import importlib
from .logger import LOGGER
from .DSU import DSU
from communities.algorithms import bron_kerbosch
import numpy as np
import copy
import os
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.NodeType import NodeType
from .LatentGroups import LatentGroups, getLfromLatentGroups
from .Chi2RankTest import Chi2RankTest
from . import misc as M
from .misc import Independences, Edges, powerset
from .Cover import getOrderedVarsString, setLength, setDifference, Cover, pairwiseOverlap, getVars
from itertools import combinations
from .GraphDrawer import DotGraph
from joblib import delayed, Parallel


def _adjacency_to_causal_graph(adjacency, var_names):
    """Convert RLCD's adjacency matrix into causal-learn's CausalGraph wrapper."""
    nodes = []
    for name in var_names:
        node = GraphNode(name)
        if name.startswith("L"):
            node.set_node_type(NodeType.LATENT)
        nodes.append(node)

    graph = GeneralGraph(nodes)
    for i in range(len(adjacency)):
        for j in range(i + 1, len(adjacency)):
            if adjacency[i, j] == 0 and adjacency[j, i] == 0:
                continue
            if adjacency[i, j] == -1 and adjacency[j, i] == 1:
                graph.add_directed_edge(nodes[i], nodes[j])
            elif adjacency[i, j] == 1 and adjacency[j, i] == -1:
                graph.add_directed_edge(nodes[j], nodes[i])
            else:
                graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))

    cg = CausalGraph(len(var_names), var_names)
    cg.G = graph
    return cg


def _rlcd_impl(
    sample,
    xvars: list = None,
    df: pd.DataFrame = None,
    input_parameters: dict = None,
):

    parameters = {
        "xvars": xvars,
        "alpha_dict": {0: 0.01, 1: 0.01, 2: 0.01, 3:0.01},
        "maxk": 3,
        "allow_nonleafx": True,
        "unfold_covers": True,
        "check_v": True,        
        "stages": 2,
        "stage1_method": "fges",
        "stage1_ges_sparsity": 2,
        "stage1_CI_alpha": 0.01,
        "stage1_partition_thres": 3,
        "ranktest_method": None,
        "citest_method": None,
    }

    if input_parameters is not None:
        parameters.update(input_parameters)
    parameters['sample'] = sample
    xvars = parameters["xvars"]
    if xvars is None:
        if df is not None:
            xvars = list(df.columns)
        elif parameters["ranktest_method"] is not None and hasattr(parameters["ranktest_method"], "data"):
            xvars = [f"X{i + 1}" for i in range(parameters["ranktest_method"].data.shape[1])]
        else:
            raise ValueError("xvars must be provided when neither df nor ranktest_method.data is available.")
        parameters["xvars"] = xvars

    if parameters['stages']>=1:
        if not parameters['sample']:
            if parameters['stage1_method']=='all':
                Adj_stage1 = np.ones((len(xvars),len(xvars)))
                partition = [xvars]
            elif parameters['stage1_method']=='fci':
                from .FCI_CovRank import fci_true_cov_rank
                G, edges = fci_true_cov_rank(np.zeros((1, len(xvars))), parameters['citest_method'])
                Adj_stage1 = process_fci_result(G.graph)
                partition = getPartition(xvars, abs(Adj_stage1), parameters['stage1_partition_thres'])
        else:
            if parameters['stage1_method']=='all':
                Adj_stage1 = np.ones((len(xvars),len(xvars)))
                partition = [xvars]
            elif parameters['stage1_method']=='fges':
                jpype = importlib.import_module("jpype")
                importlib.import_module("jpype.imports")
                try:
                    jpype.startJVM(classpath=[f"./pytetrad/tetrad-current_old.jar"])
                    #current_dirname = os.path.dirname(__file__)
                    #jpype.startJVM(classpath=[f"{current_dirname}/../../utils/pytetrad/tetrad-current.jar"])
                    LOGGER.info("JVM started")
                except OSError:
                    LOGGER.info("JVM already started")
                LOGGER.info('running fges')
                TetradSearch = importlib.import_module("pytetrad.TetradSearch_old").TetradSearch
                pytetrad_search = TetradSearch(df)
                pytetrad_search.set_verbose(False)
                pytetrad_search.use_sem_bic(penalty_discount=parameters['stage1_ges_sparsity'])
                pytetrad_search.run_fges()
                cg = pytetrad_search.get_causal_learn()
                Adj_stage1 = cg.graph
                #Adj_stage1_dag = pdag2dag(cg).graph

                partition = getPartition(xvars, abs(Adj_stage1), parameters['stage1_partition_thres'])

            elif parameters['stage1_method']=='ges':
                LOGGER.info('running ges')
                from causallearn.search.ScoreBased.GES import ges
                Record = ges(df.to_numpy(), parameters={'lambda':parameters['stage1_ges_sparsity']})
                Adj_stage1 = Record['G'].graph 
                #Adj_dag = pdag2dag(Record['G']).graph

                partition = getPartition(xvars, abs(Adj_stage1), parameters['stage1_partition_thres'])
            else:
                raise NotImplementedError

        LOGGER.info("Partition of Cliques")
        for group in partition:
            LOGGER.info(group)

    Adj = Adj_stage1

    if parameters['stages']>=2:

        for current_xvars in partition:

            current_xvars_idx = [xvars.index(x) for x in current_xvars]

            def get_neighbour_set(all_xvars, current_xvars, Adj):
                nb_set = set()

                for xvar1 in current_xvars:
                    xvar1_idx = all_xvars.index(xvar1)

                    for xvar2 in all_xvars:
                        if xvar2 not in current_xvars:
                            xvar2_idx = all_xvars.index(xvar2)

                            if Adj[xvar1_idx, xvar2_idx]!=0: #adjacent
                                nb_set.add(xvar2)

                return nb_set
            
            neighbour_set = get_neighbour_set(xvars, current_xvars, Adj)

            #local_Adj = Adj[current_xvars_idx].T[current_xvars_idx].T
            local_Adj = Adj[np.ix_(current_xvars_idx, current_xvars_idx)]

            current_G = LatentGroups(X=current_xvars, Xns=current_xvars, all_nb_set=neighbour_set, nb_set_dict={x_var:set() for x_var in current_xvars}, \
                                     local_Adj=local_Adj)
            
            current_G = rlcd_find_latent(current_G, parameters)
            current_output_Adj = getLfromLatentGroups(current_G, current_xvars)
            current_output_Adj = getReducedAdj(current_output_Adj, [i for i in range(len(current_xvars))])

            num_new_latent = current_output_Adj.shape[0] - len(current_xvars)

            if num_new_latent>0:

                # pad Adj by num_new_latent
                temp = np.zeros((Adj.shape[0]+num_new_latent, Adj.shape[0]+num_new_latent))
                temp[:Adj.shape[0],:Adj.shape[0]] = Adj
                Adj = temp

                def copy_by_idx(A_, B, indexes1_in_A, indexes2_in_A):
                    A=A_.copy()
                    assert(len(indexes1_in_A)==B.shape[0])
                    assert(len(indexes2_in_A)==B.shape[1])
                    for idx1_B, idx1_A in enumerate(indexes1_in_A):
                        for idx2_B, idx2_A in enumerate(indexes2_in_A):
                            A[idx1_A,idx2_A] = B[idx1_B,idx2_B]
                    return A

                # update x*x
                Adj = copy_by_idx(Adj, current_output_Adj[:len(current_xvars), :len(current_xvars)], current_xvars_idx, current_xvars_idx)
                # update x*l, l*x, and l*l
                current_lvars_idx = [x for x in range(Adj.shape[0]-num_new_latent, Adj.shape[0])]
                Adj = copy_by_idx(Adj, current_output_Adj[:len(current_xvars), len(current_xvars):], current_xvars_idx, current_lvars_idx)
                Adj = copy_by_idx(Adj, current_output_Adj[len(current_xvars):, :len(current_xvars)], current_lvars_idx, current_xvars_idx)
                Adj = copy_by_idx(Adj, current_output_Adj[len(current_xvars):, len(current_xvars):], current_lvars_idx, current_lvars_idx)

    # ending
    all_vars = [x for x in xvars]
    for i in range(Adj.shape[0]-len(xvars)):
        all_vars.append(f"L{i+1}")

    atomic_mask = np.zeros_like(Adj)
    for i in range(len(Adj)):
        for j in range(len(Adj)):
            if Adj[i,j]==-2:
                Adj[i,j]=1
                atomic_mask[i,j]=1

    Adj_combined = Adj.copy()
    for i in range(len(Adj_combined)):
        for j in range(len(Adj_combined)):
            if atomic_mask[i,j]==1:
                Adj_combined[i,j]=0

    cg = _adjacency_to_causal_graph(Adj_combined, all_vars)
    stage1_cg = _adjacency_to_causal_graph(Adj_stage1, xvars)
    
    return cg, stage1_cg, Adj_combined, all_vars



def RLCD(
    data,
    ranktest_method=None,
    stage1_method="ges",
    alpha_dict=None,
    maxk=3,
    node_names=None,
    stage1_ges_sparsity=2,
    stage1_partition_thres=3,
    allow_nonleafx=True,
    **kwargs,
):
    """Run Rank-based Latent Causal Discovery.

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix with shape (n_samples, n_features).
    ranktest_method : object, optional
        Rank test object with a ``test(pcols, qcols, r, alpha)`` method. If
        omitted, ``Chi2RankTest(data)`` is used.
    stage1_method : str, default="ges"
        Stage-1 method used to partition observed variables. Supported values
        are inherited from RLCD's structure-learning implementation.
    alpha_dict : dict, optional
        Significance levels for rank tests by rank.
    maxk : int, default=3
        Maximum rank-search cardinality.
    node_names : list, optional
        Names for observed variables in the returned graph. If omitted,
        variables are named X1, X2, ...

    Returns
    -------
    cg : CausalGraph
        Learned graph over observed and latent variables, where
        cg.G.graph[j, i] = 1 and cg.G.graph[i, j] = -1 indicate i --> j.
        Additional RLCD outputs are attached as ``stage1_cg``, ``adjacency``,
        and ``all_vars``.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-dimensional array.")

    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    if len(node_names) != data.shape[1]:
        raise ValueError("node_names must have the same length as the number of columns in data.")

    if alpha_dict is None:
        alpha_dict = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01}
    if ranktest_method is None:
        ranktest_method = Chi2RankTest(data)

    input_parameters = {
        "ranktest_method": ranktest_method,
        "stage1_method": stage1_method,
        "alpha_dict": alpha_dict,
        "maxk": maxk,
        "allow_nonleafx": allow_nonleafx,
        "stage1_ges_sparsity": stage1_ges_sparsity,
        "stage1_partition_thres": stage1_partition_thres,
    }
    input_parameters.update(kwargs)

    df = pd.DataFrame(data, columns=node_names)
    cg, stage1_cg, adjacency, all_vars = _rlcd_impl(
        sample=True,
        xvars=node_names,
        df=df,
        input_parameters=input_parameters,
    )
    cg.stage1_cg = stage1_cg
    cg.adjacency = adjacency
    cg.all_vars = all_vars
    return cg

def getReducedAdj(Adj, cid_ls):

    num_vars = len(Adj)

    def dfs(start_id, current_id, travel_record):

        result_id_set = set()

        if current_id in cid_ls and current_id!=start_id:
            result_id_set |= set([i for i, x in enumerate(travel_record) if x==1])
            return result_id_set

        for j in range(num_vars):
            if Adj[current_id, j]!=0 and travel_record[j]==0:
                travel_record_new = travel_record.copy()
                travel_record_new[j]=1

                result_id_set = result_id_set | dfs(start_id, j, travel_record_new)

        return result_id_set
    

    result_id_set = set(cid_ls)

    for start_id in cid_ls:
        travel_record = [0 for i in range(num_vars)]
        travel_record[start_id] = 1
        result_id_set |= dfs(start_id, start_id, travel_record)

    result_id_list = list(result_id_set)
    result_id_list.sort()

    return Adj[result_id_list,:][:,result_id_list]



def getPartition(xvars, Adj, clique_size_thres, direct_mode=False):

    def checkRelationBetweenCliques(clique1, clique2):
        
        common = set.intersection(clique1, clique2)
        if len(common)<2:
            return False
        else:
            return True

    # Adj is pc's result
    partition = []
    communities = bron_kerbosch(abs(Adj), pivot=True) #if c1 subset c2 then it does not output c1
    communities = [x for x in communities if len(x)>=clique_size_thres]

    if direct_mode:
        for clique in communities:
            if len(clique)>=clique_size_thres+1:
                temp = {xvars[i] for i in clique}
                partition.append(temp)
                LOGGER.info(f"Put {temp} with length {len(temp)} into queue")
        return partition
    
    else:
        dsu = DSU(len(communities))
        for i, clique1 in enumerate(communities):
            for j, clique2 in enumerate(communities):
                if i!=j and checkRelationBetweenCliques(clique1, clique2):
                    dsu.union(i, j)

        fa_set = set()
        for i in range(len(communities)):
            fa_set.add(dsu.find(i))
            
        partition = []
        for fa in fa_set:
            cliques = []
            for i in range(len(communities)):
                if dsu.find(i)==fa:
                    cliques.append(communities[i])
            temp = {xvars[i] for i in set.union(*cliques)}
            if len(temp)>=4:
                partition.append(list(temp))
                LOGGER.info(f"Put {temp} with length {len(temp)} into queue")
        
    return partition

def process_fci_result(adj_L):
    result_L = np.zeros_like(adj_L)
    for i in range(adj_L.shape[0]):
        for j in range(adj_L.shape[1]):
            if j<i:
                if adj_L[i,j]!=0 and adj_L[j,i]!=0:
                    if (adj_L[i,j]==-1 and adj_L[j,i]==1):
                        result_L[i,j]=-1
                        result_L[j,i]=1
                    elif (adj_L[i,j]==1 and adj_L[j,i]==-1):
                        result_L[i,j]=1
                        result_L[j,i]=-1
                    elif (adj_L[i,j]==2 and adj_L[j,i]==1):
                        result_L[i,j]=-1
                        result_L[j,i]=1
                    elif (adj_L[i,j]==1 and adj_L[j,i]==2):
                        result_L[i,j]=1
                        result_L[j,i]=-1
                    else:
                        result_L[i,j]=-1
                        result_L[j,i]=-1
    return result_L


def AdjToGraph(Adj, varnames_for_Adj):

    dotGraph = DotGraph()
    for x in varnames_for_Adj:
        if x.startswith("L"):
            dotGraph.addNodeByColor(x, 'red')
        else:
            dotGraph.addNodeByColor(x, 'blue')

    for i in range(len(Adj)):
        for j in range(i+1, len(Adj)):
            
            if (Adj[i,j]==1 and Adj[j,i]==1) or Adj[i,j]==-1 and Adj[j,i]==-1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j])
            elif Adj[i,j]==-1 and Adj[j,i]==1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j], type=1)
            elif Adj[j,i]==-1 and Adj[i,j]==1:
                dotGraph.addEdge(varnames_for_Adj[j], varnames_for_Adj[i], type=1)

    return dotGraph


def rlcd_find_latent(G:LatentGroups, parameters):
    if parameters['stages'] >= 2:
        G, _ = findClusters(G, parameters)

    #if parameters['stages'] >= 3:
    #    G = refineClusters(G)

    return G


def generateLatentPowersetFromActiveSet(G):
    """
    Generate an iterator over powerset of active latents,
    Including the combination where all latents are included.
    """

    Ls = set([V for V in G.activeSet if V.is_leaf==False])
    Ls_ordered = list(Ls)
    Ls_ordered.sort(key=lambda x: getOrderedVarsString(x))
    Lsubsets = reversed(list(M.powerset(Ls_ordered)))

    return [x for x in Lsubsets]

def getVarNames(As):
    measuredVars = []
    for A in As:
        if not A.is_observed:
            assert False, "A is not a measured var set"
        for temp in A.vars:
            measuredVars.append(temp)
    return measuredVars
    
def structuralRankTest(xvars, ranktest_method, alpha_dict, G: LatentGroups, As, Bs, k, nonLeafs):
    """
    Test if As forms a cluster by seeing if rank(subcov[A,B]) <= k.

    Returns tuple of whether rank is deficient and lowest rank tested.
    """

    Ameasures = G.pickAllMeasures(As)
    Ameasures = getVarNames(Ameasures)
    Bmeasures = G.pickAllMeasures(Bs)
    Bmeasures = getVarNames(Bmeasures)

    Ameasures += nonLeafs
    Bmeasures += nonLeafs

    Ameasures = list(set(Ameasures))
    Bmeasures = list(set(Bmeasures))

    pcols = [xvars.index(a) for a in Ameasures]
    qcols = [xvars.index(b) for b in Bmeasures]

    fail_to_reject = ranktest_method.test(pcols, qcols, k, alpha_dict[k])

    if fail_to_reject==False:
        return (fail_to_reject, None)
    else:
        min_rank = k
        for h0_k in range(k-1, -1, -1):
            fail_to_reject_h0_k = ranktest_method.test(pcols, qcols, h0_k, alpha_dict[h0_k])
            if fail_to_reject_h0_k:
                min_rank = h0_k
            else:
                break
        return fail_to_reject, min_rank

def findClusters_at_k_by_nonsinks(G: LatentGroups, k, nonsinks, parameters):
    """
    Internal method for searchClusters.
    """
    terminate = False  # Whether we ran out of variables to test
    found = False  # Whether we found any clusters
    res_for_add = []

    num_nonsinks = len(nonsinks)

    current_activeSet = G.activeSet.copy()
    current_ChildrenOfNonAtomicsSet = G.ChildrenOfNonAtomicsSet.copy()

    for temp in nonsinks:
        current_activeSet.discard(G.X_dict[temp])
        current_ChildrenOfNonAtomicsSet.discard(G.X_dict[temp])

    # Terminate if not enough active variables
    # To test, we need n >= 2k+2
    # So terminate if  n < 2k+2
    # i.e.             k > n/2 - 1
    if k-num_nonsinks > setLength(current_activeSet) / 2 - 1:
        terminate = True
        return (found, terminate, res_for_add)
    
    if k!=len(nonsinks): # could induce latent then do not consider those neighbours in active set
        for temp in G.all_nb_set:
            if temp in G.X_dict and G.X_dict[temp] in current_activeSet:
                current_activeSet.discard(G.X_dict[temp])

    allSubsets = [x for x in M.generateSubsetMinimal(current_activeSet, k-num_nonsinks)]

    
    for v in current_activeSet:
        if len(v)>=k-num_nonsinks+1 and k-num_nonsinks!=0: # more than or eq 
            tempset = current_activeSet.copy()
            tempset.remove(v)
            additionalls = [x for x in M.generateSubsetMinimal(tempset, 0)]
            for x in additionalls:
                x.add(v)
            allSubsets=allSubsets+additionalls


    # If no subsets can be generated, terminate
    if allSubsets == [set()]:
        terminate = True
        return (found, terminate, res_for_add)

    #for As in allSubsets:
    for As in reversed(allSubsets):
        #As = set(As)  # test set

        effective_ChildrenOfNonAtomicsSet = current_ChildrenOfNonAtomicsSet.copy()
        temp_set = As | {G.X_dict[t] for t in nonsinks}

        for cover in temp_set:
            #effective_ChildrenOfNonAtomicsSet = effective_ChildrenOfNonAtomicsSet - G.findDescendants(cover, rigorous=False)
            effective_ChildrenOfNonAtomicsSet = effective_ChildrenOfNonAtomicsSet - G.findDescendants(cover, rigorous=False)

        Bs = setDifference(current_activeSet | effective_ChildrenOfNonAtomicsSet, As)  # control set
        #Bs = setDifference(current_activeSet, As)  # control set
        #BBs = setDifference(current_activeSet, As)  # control set
        observed_vars_in_As = {x.__str__() for x in As if x.is_observed}
        observed_vars_in_As_and_nonsinks = observed_vars_in_As.union(set(nonsinks))
        observed_vars_in_As_and_nonsinks = list(observed_vars_in_As_and_nonsinks)
        observed_vars_in_As_and_nonsinks_idx_in_local_adj = [G.x_list_for_local_Adj.index(x) for x in observed_vars_in_As_and_nonsinks]

        temp_local_adj = G.local_Adj[observed_vars_in_As_and_nonsinks_idx_in_local_adj,:][:,observed_vars_in_As_and_nonsinks_idx_in_local_adj]

        def check_dsu(adj): 
            num_var = len(adj)
            dsu = DSU(num_var)
            for i in range(num_var):
                for j in range(num_var):
                    if i!=j and (adj[i,j]!=0 or adj[j,i]!=0):
                        dsu.union(i, j)
            fa_set = set()
            for i in range(num_var):
                fa_set.add(dsu.find(i))

            if len(fa_set)==1:
                return True
            else:
                return False

        if not check_dsu(temp_local_adj):
            continue

        if setLength(Bs)<=k-len(nonsinks):
            continue
        #if len(As) > setLength(Bs):
        #    continue

        # As must not contain more than k elements from
        # any atomic Cover with cardinality <= k-1
        if G.containsCluster(As, nonsinks): # toask
            continue

        #if G.containsonlyaCluster(Bs, nonsinks): # toask
        #    continue

        if G.overlapPaCh(As):
            continue

        if G.MeassuredHasNonSinks(As, nonsinks):
            continue

        if G.checkNonSinksAreAsChildren(As, nonsinks):
            continue

        # Bs parentCardinality cannot be < k+1, since otherwise
        # we get rank <= k regardless of what As is
        if G.parentCardinality(Bs) <= k - num_nonsinks: # dxs seems important to LLHCase2
            continue
            #if len(unfolded)==1 and Bs.issubset(G.findChildren(unfolded[0])):
            #    print("allow parentCardinality(Bs) <= k - num_nonsinks")
            #else:
            #    continue

        fail_to_reject, rk = structuralRankTest(parameters['xvars'], parameters['ranktest_method'], parameters['alpha_dict'], G, As, Bs, k, list(nonsinks))
        

        if fail_to_reject:
            LOGGER.info(f"   {As} is rank deficient! given {nonsinks}, Bs:{Bs}")

            v_structure_found = False

            if parameters['check_v']:
                # check v structure
                for num_colider in range(1,k-num_nonsinks+1): #|As|=k-num_nonsinks+1
                    num_subAs = k-num_nonsinks+1-num_colider
                    for subAs in M.generateSubsetMinimal(As, num_subAs-1):
                        #test_subAs, rk_subAs = self.structuralRankTest(G, subAs, Bs - current_ChildrenOfNonAtomicsSet, num_nonsinks+num_subAs-1, list(nonsinks))
                        test_subAs, rk_subAs = \
                            structuralRankTest(parameters['xvars'], parameters['ranktest_method'], parameters['alpha_dict'], G, subAs, Bs, num_nonsinks+num_subAs-1, list(nonsinks))
                        if test_subAs:
                            LOGGER.info(f"   {As} has v structure! subAs:{subAs} given {nonsinks}, Bs:{Bs}")
                            v_structure_found = True
     
            if v_structure_found == False:
                res_for_add.append((As, rk, nonsinks))
                #G.addRankDefSet(As, rk, used_nonsinks=nonsinks)
                found = True

    return (found, terminate, res_for_add)
    

def findClusters_at_k_mp(G: LatentGroups, k, parameters, n_jobs=-1):
    """
    Run one round of search for clusters of size k.
    """
    LOGGER.info(f"Starting searchClusters k={k}...")
    global_terminate=True
    global_found=False
    found_deficiency = False

    input_list = []

    for num_nonsinks in range(k, -1, -1): # [k,k-1,...,0]
        
        temp_activeNonSink_ls = sorted(list(G.activeNonSinkSet), reverse=True)
        #temp_activeNonSink_ls = list(G.activeNonSinkSet)
        nonsinks_ls = list(combinations(temp_activeNonSink_ls, num_nonsinks))

        for nonsinks in nonsinks_ls:
            input_list.append(list(nonsinks).copy())

    output_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(findClusters_at_k_by_nonsinks)(G, k, nonsinks, parameters)
            for nonsinks in input_list
        )
            
    for output in output_list:
        current_found_deficiency, current_terminate, res_for_add = output
        found_deficiency = found_deficiency or current_found_deficiency
        global_terminate = global_terminate and current_terminate

        for i in range(len(res_for_add)):
            G.addRankDefSet(res_for_add[i][0], res_for_add[i][1], used_nonsinks=res_for_add[i][2])

    if found_deficiency:
        G.determineClusters() # all the input deficient set are based on the same nonLeafs
        found = G.confirmClusters()
        global_found = global_found or found

        if global_found:
            G.updateActiveSet()
            G.updateactiveNonSinkSet()
            M.display(G)
            #printGraph(G)
            return G, (global_found, global_terminate)

    return G, (global_found, global_terminate)

def findClusters_at_k(G: LatentGroups, k, parameters):
    """
    Run one round of search for clusters of size k.
    """
    LOGGER.info(f"Starting searchClusters k={k}...")
    global_terminate=True
    global_found=False
    found_deficiency = False

    for num_nonsinks in range(k, -1, -1): # [k,k-1,...,0]
        
        temp_activeNonSink_ls = sorted(list(G.activeNonSinkSet), reverse=True)
        #temp_activeNonSink_ls = list(G.activeNonSinkSet)
        nonsinks_ls = list(combinations(temp_activeNonSink_ls, num_nonsinks))

        for nonsinks in nonsinks_ls:
            
            current_found_deficiency, current_terminate, res_for_add = findClusters_at_k_by_nonsinks(G, k, list(nonsinks), parameters)
            found_deficiency = found_deficiency or current_found_deficiency
            global_terminate = global_terminate and current_terminate

            for i in range(len(res_for_add)):
                G.addRankDefSet(res_for_add[i][0], res_for_add[i][1], used_nonsinks=res_for_add[i][2])
        

    if found_deficiency:
        G.determineClusters() # all the input deficient set are based on the same nonLeafs
        found = G.confirmClusters()
        global_found = global_found or found

        if global_found:
            G.updateActiveSet()
            G.updateactiveNonSinkSet()
            M.display(G)
            #printGraph(G)
            return G, (global_found, global_terminate)

        
    return G, (global_found, global_terminate)


def findClusters(G: LatentGroups, parameters):

    prevCovers = set(G.latentDict.keys())  # Record current latent Covers

    k = 1
    while True:
        LOGGER.info(f"{'-'*15} Test Cardinality now k={k} {'-'*15}")
        
        if parameters['unfold_covers']:
            LPowerSet = generateLatentPowersetFromActiveSet(G)
        else:
            LPowerSet = [()]
        activeSetCopy = copy.deepcopy(G.activeSet)
        activeNonSinkSetCopy = copy.deepcopy(G.activeNonSinkSet)

        # Select a combination of latents, and replace their place in
        # the activeSet with their children for the search
        for i, Ls in enumerate(LPowerSet):

            if i==0:
                all_unfolded=True
            else:
                all_unfolded=False

            Vprime = copy.deepcopy(activeSetCopy)
            Tprime = copy.deepcopy(activeNonSinkSetCopy)

            for L in Ls:
                # L is a Cover
                #children = G.findChildren(L, rigorous=False) # 
                children = G.findChildren(L, rigorous=True) # 
                meassured_subset_L = G.findMeassuredSubset(L)
                for cover in meassured_subset_L:
                    Tprime |= cover.vars

                # If children of L is just one or zero variable, do not replace
                if len(children) + len(meassured_subset_L) > 1:
                #if len(children)>= 1:
                    Vprime = Vprime - set([L])
                    Vprime |= children
                    Vprime |= meassured_subset_L

            children = G.findChildrenOfAllSubSets(Ls) # 
            Vprime |= children

            for cover in G.activeSet:
                if len(cover.vars)==1 and cover.takeOne() in G.X_dict:
                    if G.X_dict[cover.takeOne()].is_leaf!=True:
                        Tprime |= cover.vars

            
            G.activeSet = Vprime

            if parameters['allow_nonleafx']:
                G.activeNonSinkSet = Tprime
            else:
                G.activeNonSinkSet = set()

            LOGGER.info(f"Unfolding {Ls}")
            LOGGER.info(f"G.activeSet {G.activeSet}")
            LOGGER.info(f"G.activeNonSinkSet {G.activeNonSinkSet}")
            G, (found, terminate) = findClusters_at_k_mp(G, k, parameters, n_jobs=-1)
            #G, (found, terminate) = findClusters_at_k(G, k, parameters)

            if found:
                #G.reconnectNonAtomics()
                G.updateActiveSet() # toask
                G.updateactiveNonSinkSet()

            # CASE 2
            if terminate:
                break

            # CASE 1
            if found:
                k = 1
                break

        # CASE 3
        if not found:
            LOGGER.info("Nothing found!")
            k += 1

        # CASE 2 toask
        #if (k > self.maxk) or terminate:
        if (k > parameters['maxk']):
            LOGGER.info(f"Procedure ending...")
            break

    
    G = findClusters_finish(G, parameters)
    G.updateActiveSet(if_for_finish=True)
    # 1

    newCovers = set(G.latentDict.keys()) - prevCovers
    return G, newCovers


def findClusters_finish(G: LatentGroups, parameters):
    """
    Procedure for completing the graph when no more clusters may be found,
    by introducing a temporary root latent variable.
    """
    LOGGER.info(f"{'-'*15} Check If Introducing temporary root variable ... {'-'*15}")
    G.updateActiveSet(if_for_finish=True)
    if len(G.activeSet) == 1:
        pass
    #elif len(G.activeSet) >2:
    #    G.introduceTempRoot()
    #    LOGGER.info(f"{'-'*15} Introduced temporary root variable ... {'-'*15}")
    else: 

        G.updateActiveSet(if_for_finish=False)
        remain_covers = list(G.activeSet)
        remain_latent_covers = []
        remain_observed_covers = []

        for i in range(len(remain_covers)):
            if not remain_covers[i].is_observed:
                if remain_covers[i].atomic:
                    remain_latent_covers.append(remain_covers[i])
            else:
                remain_observed_covers.append(remain_covers[i])

        if len(remain_observed_covers)!=0:
            for i in range(len(remain_observed_covers)):
                to_add = set()
                for j in range(len(remain_latent_covers)):
                    fail_to_reject, rk = structuralRankTest(parameters['xvars'], parameters['ranktest_method'], parameters['alpha_dict'], G, \
                                                            set([remain_observed_covers[i]]), set([remain_latent_covers[j]]), 0, [])
                    if not fail_to_reject:
                        to_add.add(remain_latent_covers[j])

                if len(to_add)!=0:
                    G.addOrUpdateCover(remain_observed_covers[i], to_add)
                    G.updateActiveSet()
        
        else:
            G.updateActiveSet(if_for_finish=True)
            remain_covers = list(G.activeSet)
            remain_latent_covers = []

            for i in range(len(remain_covers)):
                if not remain_covers[i].is_observed:
                        remain_latent_covers.append(remain_covers[i])

            if len(remain_latent_covers)>=2:
                for i in range(len(remain_latent_covers)-1):
                    G.addOrUpdateCover(remain_latent_covers[i], set([remain_latent_covers[i+1]]))
                    G.updateActiveSet()
        
    M.display(G)
    #printGraph(G)
    #assert len(G.activeSet) == 1, "The graph should have one root variable."
    return G