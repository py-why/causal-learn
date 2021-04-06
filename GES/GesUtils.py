from GES.GesScoreFunction import *
from GES.KCI.gpml import *
import numpy as np
import numpy.matlib
import itertools
from copy import deepcopy
from graph.Edge import Edge
from graph.Endpoint import Endpoint
from graph.GeneralGraph import GeneralGraph

def feval(parameters):
    if parameters[0] == 'local_score_CV_general':
        return local_score_CV_general(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_marginal_general':
        return local_score_marginal_general(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_CV_multi':
        return local_score_CV_multi(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_marginal_multi':
        return local_score_marginal_multi(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covNoise':
        if (len(parameters) == 1):
            return covNoise()
        elif (len(parameters) == 2):
            return covNoise(parameters[1])
        elif (len(parameters) == 3):
            return covNoise(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covNoise(parameters[1], parameters[2], parameters[3])
        else:
            return covNoise(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSEard':
        if (len(parameters) == 1):
            return covSEard()
        elif (len(parameters) == 2):
            return covSEard(parameters[1])
        elif (len(parameters) == 3):
            return covSEard(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSEard(parameters[1], parameters[2], parameters[3])
        else:
            return covSEard(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'gpr_multi_new':
        if (len(parameters) == 1):
            return gpr_multi_new()
        elif (len(parameters) == 2):
            return gpr_multi_new(parameters[1])
        elif (len(parameters) == 3):
            return gpr_multi_new(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
        elif (len(parameters) == 7):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
    elif parameters[0] == 'covSum':
        if (len(parameters) == 1):
            return covSum()
        elif (len(parameters) == 2):
            return covSum(parameters[1])
        elif (len(parameters) == 3):
            return covSum(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSum(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
    else:
        raise Exception('请选择已定义的函数')



def Combinatorial(T0):
    # sub = Combinatorial (T0); % find all the sbusets of T0
    sub = []
    count = 0
    if (len(T0) == 0):
        sub.append(()) # a 1x0 empty matrix
    else:
        if (len(T0) == 1):
            sub.append(())
            sub.append(T0) # when T0 is a scale, it is a special case!!
        else:
            for n in range(len(T0) + 1):
                for S in list(itertools.combinations(T0, n)):
                    sub.append(S)
    return sub

def Score_G(Data, G, score_func, parameters): # calculate the score for the current G
    # here G is a DAG
    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)
        delta_score = feval([score_func, Data, i, PA, parameters])
        score = score + delta_score
    return score

def Insert_validity_test1(G, i,j, T):
    # V=Insert_validity_test1(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[:,i] != 0)[0], np.where(G.graph[i, :] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) # find the neighbours of Xj and are adjacent to Xi
    V = check_clique(G, list(np.union1d(NA, T).astype(int))) # check whether it is a clique
    return V


def check_clique(G, subnode): # check whether node subnode is a clique in G
    # here G is a CPDAG
    # the definition of clique here: a clique is defined in an undirected graph
    # when you ignore the directionality of any directed edges
    Gs = deepcopy(G.graph[np.ix_(subnode, subnode)]) # extract the subgraph
    ns = len(subnode)

    if (ns == 0):
        s = 1
    else:
        row, col = np.where(Gs == 1)
        Gs[row, col] = 6
        Gs[col, row] = 6
        if (np.all(((np.eye(ns) - np.ones((ns, ns))) * -6) == Gs)): # check whether it is a clique
            s = 1
        else:
            s = 0
    return s

def Insert_validity_test2(G, i,j, T):
    # V=Insert_validity_test(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) # find the neighbours of Xj and are adjacent to Xi

    # condition 2: every semi-directed path from Xj to Xi contains a node in union(NA,T)
    # Note: EVERY!!
    s2 = Insert_vC2_new(G, j, i, np.union1d(NA, T))
    if (s2):
        V = 1

    return V

def Insert_vC2_new(G,j,i,NAT): # validity test for condition 2 of Insert operator
    # here G is CPDAG
    # Use Depth-first-Search
    start = j
    target = i
    # stack(1)=start; % initialize the stack
    stack = [{'value':start, 'pa': {}}]
    sign = 1    # If every semi-pathway contains a node in NAT, than sign=1;
    count = 1

    while (len(stack)):
        top = stack[0]
        stack = stack[1:] # pop
        if (top['value'] == target): # if find the target, search that pathway to see whether NAT is in that pathway
            curr = top
            ss = 0
            while True:
                if (len(curr['pa'])):
                    if (curr['pa']['value'] in NAT): # contains a node in NAT
                        ss = 1
                        break
                else:
                    break
                curr = curr['pa']
            if (not ss): # do not include NAT
                sign = 0
                break
        else:
            child = np.concatenate((np.where(G.graph[:, top['value']] == 1)[0], np.where(G.graph[top['value'], :] == 6)[0]))
            sign_child = np.ones(len(child))
            # check each child, whether it has appeared before in the same pathway
            for k in range(len(child)):
                curr = top
                while True:
                    if (len(curr['pa'])):
                        if (curr['pa']['value'] == child[k]):
                            sign_child[k] = 0   # has appeared in that path before
                            break
                    else:
                        break
                    curr = curr['pa']

            for k in range(len(sign_child)):
                if (sign_child[k]):
                    stack.insert(0, {'value': child[k], 'pa': top}) # push
    return sign

def find_subset_include (s0,sub):
    # S = find_subset_include(sub(k),sub); %  find those subsets that include sub(k)
    if (len(s0) == 0 or len(sub) == 0):
        Idx = np.ones(len(sub))
    else:
        Idx = np.zeros(len(sub))
        for i in range(len(sub)):
            tmp = set(s0).intersection(set(sub[i]))
            if (len(tmp)):
                if (tmp == set(s0)):
                    Idx[i] = 1
    return Idx

def Insert_changed_score(Data,G,i,j,T,record_local_score,score_func,parameters):
    # calculate the changed score after the insert operator: i->j
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) #  find the neighbours of Xj and are adjacent to Xi
    Paj = np.where(G.graph[j, :] == 1)[0] # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = np.union1d(NA, T).astype(int)
    tmp2 = np.union1d(tmp1, Paj)
    tmp3 = np.union1d(tmp2, [i]).astype(int)

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if (not np.setxor1d(record_local_score[j][r0][0:-1], tmp3).size):
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if (not np.setxor1d(record_local_score[j][r0][0:-1], tmp2).size): # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            ## 这里用 -1 代替 0，因为 matlab 和 python 的起始坐标不一样，在 matlab 中取不到 0 下标，但是 python 中可以
            if ((not np.setxor1d(record_local_score[j][r0][0:-1], [-1]).size) and (not tmp2.size)):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if (s1 and s2):
            break

    if (not s1):
        score1 = feval([score_func, Data, j, tmp3, parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if(not s2):
        score2 = feval([score_func, Data, j, tmp2, parameters])
        # r = len(record_local_score[j])
        if (len(tmp2) != 0):
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, T]
    return chscore,desc,record_local_score


def Insert(G,i,j,T):
        # Insert operator
        # insert the directed edge Xi->Xj
        nodes = G.get_nodes()
        G.add_edge(Edge(nodes[i], nodes[j], -1, 1))

        for k in range(len(T)): # directing the previous undirected edge between T and Xj as T->Xj
            if G.get_edge(nodes[T[k]], nodes[j]) is not None:
                G.remove_edge(G.get_edge(nodes[T[k]], nodes[j]))
            G.add_edge(Edge(nodes[T[k]], nodes[j], -1, 1))

        return G

def Delete_validity_test(G, i,j, H):
    # V=Delete_validity_test(G, X, Y, H); % do validity test for the operator Delete; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Hj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Hi = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi) # find the neighbours of Xj and are adjacent to Xi
    s1 = check_clique(G, list(set(NA) - set(H))) # check whether it is a clique

    if (s1):
        V = 1

    return V

def Delete_changed_score(Data, G,i,j,H,record_local_score,score_func,parameters):
    # calculate the changed score after the Delete operator
    Hj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Hi = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi) # find the neighbours of Xj and are adjacent to Xi
    Paj = np.union1d(np.where(G.graph[j, :] == 1)[0], [i]) # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = set(NA) - set(H)
    tmp2 = set.union(tmp1, set(Paj))
    tmp3 = tmp2 - {i}

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if (set(record_local_score[j][r0][0:-1]) == tmp3) :
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if (set(record_local_score[j][r0][0:-1]) == tmp2): # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if ((set(record_local_score[j][r0][0:-1]) == {-1}) and len(tmp2) == 0):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if (s1 and s2):
            break

    if (not s1):
        score1 = feval([score_func, Data, j, list(tmp3), parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if (not s2):
        score2 = feval([score_func, Data, j, list(tmp2), parameters])
        r = len(record_local_score[j])
        if (len(tmp2) != 0):
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, H]
    return chscore,desc,record_local_score

def Delete(G,i,j,H):
    # Delete operator
    nodes = G.get_nodes()
    if G.get_edge(nodes[i], nodes[j]) is not None:
        # delete the edge between Xi and Xj
        G.remove_edge(G.get_edge(nodes[i], nodes[j]))
    for k in range(len(H)): # directing the previous undirected edge
        if G.get_edge(nodes[j], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[j], nodes[H[k]]))
        if G.get_edge(nodes[i], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[i], nodes[H[k]]))
        G.add_edge(Edge(nodes[j], nodes[H[k]], -1, 1))
        G.add_edge(Edge(nodes[i], nodes[H[k]], -1, 1))
    return G

def check2(G, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if (len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0):
            s = 0
            break
    return s

# function Gd = PDAG2DAG(G) % transform a PDAG to DAG
def PDAG2DAG(G):
    nodes = G.get_nodes()
    # first create a DAG that contains all the directed edges in PDAG
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()
    for edge in edges:
        if not ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or (edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW)):
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int)) # index whether the ith node has been removed. 1:removed; 0: not
    while 0 in inde:
        for i in range(Gp.num_vars):
            if (inde[i] == 0):
                sign = 0
                if (len(np.intersect1d(np.where(Gp.graph[:,i] == 1)[0], np.where(inde == 0)[0])) == 0): # Xi has no out-going edges
                    sign = sign + 1
                    Nx = np.intersect1d(np.where(Gp.graph[:,i] == 6)[0], np.where(inde == 0)[0]) # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where(Gp.graph[i, :] == 1)[0], np.where(Gp.graph[:,i]==1)[0]), np.where(inde==0)[0]) # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if (len(Nx) > 0):
                        if check2(Gp, Nx, Ax): # according to the original paper
                            sign = sign + 1
                    else:
                        sign = sign + 1
                if (sign == 2):
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.where(Gp.graph[:,i] == 6)[0]:
                        Gd.add_edge(Edge(nodes[index], nodes[i], -1, 1))
                    inde[i] = 1

    return Gd

def DAG2CPDAG(G): # transform a DAG to a CPDAG
    ## -----------------------------------------------------
    # order the edges in G
    nodes_order = list(map(lambda x : G.node_map[x], G.get_causal_ordering())) # Perform a topological sort on the nodes of G
    # nodes_order(1) is the node which has the highest order
    # nodes_order(N) is the node which has the lowest order
    edges_order= np.asmatrix([[],[]], dtype=np.int64).T
    # edges_order(1,:) is the edge which has the highest order
    # edges_order(M,:) is the edge which has the lowest order
    M = G.get_num_edges() # the number of edges in this DAG
    N = G.get_num_nodes() # the number of nodes in this DAG

    while(edges_order.shape[0] < M):
        for ny in range(N-1, -1, -1):
            j = nodes_order[ny]
            inci_all = np.where(G.graph[j, :] == 1)[0]  # all the edges that incident to j
            if (len(inci_all) != 0):
                if (len(edges_order) != 0):
                    inci = edges_order[np.where(edges_order[:, 1] == j)[0], 0] # ordered edge that incident to j
                    if (len(set(inci_all) - set(inci.T.tolist()[0])) != 0):
                        break
                else:
                    break
        for nx in range(N):
            i = nodes_order[nx]
            if(len(edges_order) != 0):
                if(len(np.intersect1d(np.where(edges_order[:,1]==j)[0], np.where(edges_order[:,0]==i)[0])) == 0 and G.graph[j,i]==1):
                    break
            else:
                if (G.graph[j, i] == 1):
                    break
        edges_order = np.r_[edges_order, np.asmatrix([i, j])]

    ## ----------------------------------------------------------------
    sign_edges = np.zeros(M) # 0 means unknown, 1 means compelled, -1 means reversible
    while (len(np.where(sign_edges == 0)[0]) != 0):
        ss = 0
        for m in range(M-1, -1, -1): # let x->y be the lowest ordered edge that is labeled "unknown"
            if sign_edges[m] == 0:
                i = edges_order[m, 0]
                j = edges_order[m, 1]
                break
        idk = np.where(edges_order[:, 1] == i)[0]
        k = edges_order[idk, 0] # w->x
        for m in range(len(k)):
            if (sign_edges[idk[m]] == 1):
                if (G.graph[j, k[m]] != 1): # if w is not a parent of y
                    id = np.where(edges_order[:, 1] == j)[0] # label every edge that incident into y with "complled"
                    sign_edges[id] = 1
                    ss = 1
                    break
                else:
                    id = np.intersect1d(np.where(edges_order[:, 0] == k[m, 0])[0], np.where(edges_order[:, 1] == j)[0]) # label w->y with "complled"
                    sign_edges[id] = 1
        if (ss):
            continue

        z = np.where(G.graph[j, :] == 1)[0]
        if (len(np.intersect1d(np.setdiff1d(z, i), np.union1d(np.union1d(np.where(G.graph[i, :] == 0)[0], np.where(G.graph[i, :] == -1)[0]), np.where(G.graph[i,:]==6)[0]))) != 0):
            id = np.intersect1d(np.where(edges_order[:,0]== i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[id] = 1  # label x->y  with "compelled"
            id1 = np.where(edges_order[:,1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = 1 # label all "unknown" edges incident into y  with "complled"
        else:
            id = np.intersect1d(np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[id] = -1  # label x->y with "reversible"

            id1 = np.where(edges_order[:,1]==j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = -1 # label all "unknown" edges incident into y with "reversible"

    # create CPDAG accoring the labelled edge
    nodes = G.get_nodes()
    Gcp = GeneralGraph(nodes)
    for m in range(M):
        if (sign_edges[m] == 1):
            Gcp.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], -1, 1))
        else:
            Gcp.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], 6, 6))

    return Gcp