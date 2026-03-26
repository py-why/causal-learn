import itertools
from copy import deepcopy
from typing import List

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.score.LocalScoreFunction import *


def feval(parameters: list):
    return parameters[0].score(parameters[2], parameters[3])


def kernel(x, xKern, theta):
    # KERNEL Compute the rbf kernel
    n2 = dist2(x, xKern)
    if theta[0] == 0:
        theta[0] = 2 / np.median(n2[np.where(np.tril(n2) > 0)])
        theta_new = theta[0]
    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]
    return kx, bw_new


def Combinatorial(T0):
    # sub = Combinatorial (T0); % find all the subsets of T0
    sub = []
    count = 0
    if len(T0) == 0:
        sub.append(())  # a 1x0 empty matrix
    else:
        if len(T0) == 1:
            sub.append(())
            sub.append(T0)  # when T0 is a scale, it is a special case!!
        else:
            for n in range(len(T0) + 1):
                for S in list(itertools.combinations(T0, n)):
                    sub.append(S)
    return sub


def score_g(Data, G, score_func, parameters):  # calculate the score for the current G
    # here G is a DAG
    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)
        PAi = list(map(lambda node_PA: G.node_map[node_PA], PA))
        delta_score = feval([score_func, Data, i, PAi, parameters])
        score = score + delta_score
    return score


def insert_validity_test1(G, i, j, T) -> int:
    # V=Insert_validity_test1(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Tj = np.intersect1d(
        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
    )  # neighbors of Xj
    Ti = np.union1d(
        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
    )  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi
    V = check_clique(
        G, list(np.union1d(NA, T).astype(int))
    )  # check whether it is a clique
    return V


def check_clique(G, subnode) -> int:  # check whether node subnode is a clique in G
    # here G is a CPDAG
    # the definition of clique here: a clique is defined in an undirected graph
    # when you ignore the directionality of any directed edges
    Gs = deepcopy(G.graph[np.ix_(subnode, subnode)])  # extract the subgraph
    ns = len(subnode)

    if ns == 0:
        s = 1
    else:
        row, col = np.where(Gs == Endpoint.ARROW.value)
        Gs[row, col] = Endpoint.TAIL.value
        Gs[col, row] = Endpoint.TAIL.value
        if np.all(
            (np.eye(ns) - np.ones((ns, ns))) == Gs
        ):  # check whether it is a clique
            s = 1
        else:
            s = 0
    return s


def insert_validity_test2(G, i, j, T) -> int:
    # V=Insert_validity_test(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0
    Tj = np.intersect1d(
        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
    )  # neighbors of Xj
    Ti = np.union1d(
        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
    )  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi

    # condition 2: every semi-directed path from Xj to Xi contains a node in union(NA,T)
    # Note: EVERY!!
    s2 = insert_vc2_new(G, j, i, np.union1d(NA, T))
    if s2:
        V = 1

    return V


def insert_vc2_new(G, j, i, NAT):  # validity test for condition 2 of Insert operator
    # here G is CPDAG
    # Use Depth-first-Search
    start = j
    target = i
    # stack(1)=start; % initialize the stack
    stack = [{"value": start, "pa": {}}]
    sign = 1  # If every semi-pathway contains a node in NAT, than sign=1;
    count = 1

    while len(stack):
        top = stack[0]
        stack = stack[1:]  # pop
        if (
            top["value"] == target
        ):  # if find the target, search that pathway to see whether NAT is in that pathway
            curr = top
            ss = 0
            while True:
                if len(curr["pa"]):
                    if curr["pa"]["value"] in NAT:  # contains a node in NAT
                        ss = 1
                        break
                else:
                    break
                curr = curr["pa"]
            if not ss:  # do not include NAT
                sign = 0
                break
        else:
            child = np.concatenate(
                (
                    np.where(G.graph[:, top["value"]] == Endpoint.ARROW.value)[0],
                    np.intersect1d(
                        np.where(G.graph[top["value"], :] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[:, top["value"]] == Endpoint.TAIL.value)[0],
                    ),
                )
            )
            sign_child = np.ones(len(child))
            # check each child, whether it has appeared before in the same pathway
            for k in range(len(child)):
                curr = top
                while True:
                    if len(curr["pa"]):
                        if curr["pa"]["value"] == child[k]:
                            sign_child[k] = 0  # has appeared in that path before
                            break
                    else:
                        break
                    curr = curr["pa"]

            for k in range(len(sign_child)):
                if sign_child[k]:
                    stack.insert(0, {"value": child[k], "pa": top})  # push
    return sign


def find_subset_include(s0, sub):
    # S = find_subset_include(sub(k),sub); %  find those subsets that include sub(k)
    if len(s0) == 0 or len(sub) == 0:
        Idx = np.ones(len(sub))
    else:
        Idx = np.zeros(len(sub))
        for i in range(len(sub)):
            tmp = set(s0).intersection(set(sub[i]))
            if len(tmp):
                if tmp == set(s0):
                    Idx[i] = 1
    return Idx


def insert_changed_score(Data, G, i, j, T, record_local_score, score_func, parameters):
    # calculate the changed score after the insert operator: i->j
    Tj = np.intersect1d(
        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
    )  # neighbors of Xj
    Ti = np.union1d(
        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
    )  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = np.union1d(NA, T).astype(int)
    tmp2 = np.union1d(tmp1, Paj)
    tmp3 = np.union1d(tmp2, [i]).astype(int)

    # look up or compute local scores using dict cache: key = (node, tuple(sorted(parents)))
    key1 = (j, tuple(sorted(tmp3)))
    key2 = (j, tuple(sorted(tmp2)))

    if key1 in record_local_score:
        score1 = record_local_score[key1]
    else:
        score1 = feval([score_func, Data, j, tmp3, parameters])
        record_local_score[key1] = score1

    if key2 in record_local_score:
        score2 = record_local_score[key2]
    else:
        score2 = feval([score_func, Data, j, tmp2, parameters])
        record_local_score[key2] = score2

    ch_score = score1 - score2
    desc = [i, j, T]
    return ch_score, desc, record_local_score


def insert(G, i, j, T):
    # Insert operator
    # insert the directed edge Xi->Xj
    nodes = G.get_nodes()
    G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))

    for k in range(
        len(T)
    ):  # directing the previous undirected edge between T and Xj as T->Xj
        if G.get_edge(nodes[T[k]], nodes[j]) is not None:
            G.remove_edge(G.get_edge(nodes[T[k]], nodes[j]))
        G.add_edge(Edge(nodes[T[k]], nodes[j], Endpoint.TAIL, Endpoint.ARROW))

    return G


def delete_validity_test(G, i, j, H):
    # V=Delete_validity_test(G, X, Y, H); % do validity test for the operator Delete; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Hj = np.intersect1d(
        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
    )  # neighbors of Xj
    Hi = np.union1d(
        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
    )  # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi)  # find the neighbours of Xj and are adjacent to Xi
    s1 = check_clique(G, list(set(NA) - set(H)))  # check whether it is a clique

    if s1:
        V = 1

    return V


def delete_changed_score(Data, G, i, j, H, record_local_score, score_func, parameters):
    # calculate the changed score after the Delete operator
    Hj = np.intersect1d(
        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
    )  # neighbors of Xj
    Hi = np.union1d(
        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
    )  # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.union1d(
        np.where(G.graph[j, :] == Endpoint.ARROW.value)[0], [i]
    )  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = set(NA) - set(H)
    tmp2 = set.union(tmp1, set(Paj))
    tmp3 = tmp2 - {i}

    # look up or compute local scores using dict cache: key = (node, tuple(sorted(parents)))
    key1 = (j, tuple(sorted(tmp3)))
    key2 = (j, tuple(sorted(tmp2)))

    if key1 in record_local_score:
        score1 = record_local_score[key1]
    else:
        score1 = feval([score_func, Data, j, list(tmp3), parameters])
        record_local_score[key1] = score1

    if key2 in record_local_score:
        score2 = record_local_score[key2]
    else:
        score2 = feval([score_func, Data, j, list(tmp2), parameters])
        record_local_score[key2] = score2

    ch_score = score1 - score2
    desc = [i, j, H]
    return ch_score, desc, record_local_score


def delete(G, i, j, H):
    # Delete operator
    nodes = G.get_nodes()
    if G.get_edge(nodes[i], nodes[j]) is not None:
        # delete the edge between Xi and Xj
        G.remove_edge(G.get_edge(nodes[i], nodes[j]))
    for k in range(len(H)):  # directing the previous undirected edge
        if G.get_edge(nodes[j], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[j], nodes[H[k]]))
        if G.get_edge(nodes[i], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[i], nodes[H[k]]))
        G.add_edge(Edge(nodes[j], nodes[H[k]], Endpoint.TAIL, Endpoint.ARROW))
        G.add_edge(Edge(nodes[i], nodes[H[k]], Endpoint.TAIL, Endpoint.ARROW))
    return G


def dist2(x, c):
    # DIST2	Calculates squared distance between two sets of points.
    #
    # Description
    # D = DIST2(X, C) takes two matrices of vectors and calculates the
    # squared Euclidean distance between them.  Both matrices must be of
    # the same column dimension.  If X has M rows and N columns, and C has
    # L rows and N columns, then the result has M rows and L columns.  The
    # I, Jth entry is the  squared distance from the Ith row of X to the
    # Jth row of C.
    #
    # See also
    # GMMACTIV, KMEANS, RBFFWD
    #

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise Exception("Data dimension does not match dimension of centres")

    n2 = (
        (np.ones((ncentres, 1)) * np.sum(np.multiply(x, x).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum(np.multiply(c, c).T, axis=0)
        - 2 * (x * c.T)
    )

    # Rounding errors occasionally cause negative entries in n2
    n2[np.where(n2 < 0)] = 0
    return n2


def pdinv(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A).T
        invU = np.eye(numData).dot(np.linalg.inv(U))
        Ainv = invU.dot(invU.T)
    except numpy.linalg.LinAlgError as e:
        warnings.warn("Matrix is not positive definite in pdinv, inverting using svd")
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
    except Exception as e:
        raise e
    return Ainv


# ===========================================================
# Fast versions of hot-path functions (set-based, no deepcopy)
# ===========================================================

def precompute_graph_info(G, N):
    """Precompute neighbors, adjacent, parents, and semi-directed successors for all nodes.

    Returns dicts keyed by node index, with Python set values.
    Should be called once per outer iteration of GES (graph doesn't change within iteration).
    """
    # Cache Endpoint values to avoid repeated enum lookups
    TAIL = Endpoint.TAIL.value   # -1
    ARROW = Endpoint.ARROW.value  # 1

    nbrs = {}   # undirected neighbors
    adj = {}    # all adjacent nodes
    pa = {}     # parents (directed into)
    semi = {}   # semi-directed successors (children + undirected neighbors)

    for node in range(N):
        row = G.graph[node, :]  # G.graph[node, other]
        col = G.graph[:, node]  # G.graph[other, node]
        nbrs[node] = set(np.where((col == TAIL) & (row == TAIL))[0])
        adj[node] = set(np.where((col != 0) | (row != 0))[0])
        pa[node] = set(np.where(row == ARROW)[0])
        children = set(np.where(col == ARROW)[0])
        semi[node] = children | nbrs[node]

    return nbrs, adj, pa, semi


def check_clique_fast(G, nodes):
    """Check if nodes form a clique (ignoring edge direction). No deepcopy."""
    nodes = list(nodes)
    ns = len(nodes)
    if ns <= 1:
        return 1
    for a in range(ns):
        for b in range(a + 1, ns):
            ni, nj = nodes[a], nodes[b]
            if G.graph[ni, nj] == 0 and G.graph[nj, ni] == 0:
                return 0
    return 1


def insert_vc2_fast(j, i, NAT_set, semi_succ):
    """Check if every semi-directed path from j to i contains a node in NAT_set.

    Uses precomputed semi_succ[node] and set-based NAT membership (O(1) lookup).
    """
    start = j
    target = i
    stack = [{"value": start, "pa": {}}]
    sign = 1

    while len(stack):
        top = stack[0]
        stack = stack[1:]
        if top["value"] == target:
            curr = top
            ss = 0
            while True:
                if curr["pa"]:
                    if curr["pa"]["value"] in NAT_set:
                        ss = 1
                        break
                else:
                    break
                curr = curr["pa"]
            if not ss:
                sign = 0
                break
        else:
            children = semi_succ.get(top["value"], set())
            for child in children:
                # Check if child has appeared in path before
                curr = top
                appeared = False
                while True:
                    if curr["pa"]:
                        if curr["pa"]["value"] == child:
                            appeared = True
                            break
                    else:
                        break
                    curr = curr["pa"]
                if not appeared:
                    stack.insert(0, {"value": child, "pa": top})
    return sign


def insert_changed_score_fast(Data, i, j, T, NA_set, Paj_set,
                               record_local_score, score_func, parameters):
    """Compute score change for insert i->j using precomputed NA and Paj sets."""
    tmp2 = NA_set | set(T) | Paj_set
    tmp3 = tmp2 | {i}

    tmp3_key = tuple(sorted(tmp3))
    tmp2_key = tuple(sorted(tmp2))

    key1 = (j, tmp3_key)
    key2 = (j, tmp2_key)

    if key1 in record_local_score:
        score1 = record_local_score[key1]
    else:
        score1 = feval([score_func, Data, j, list(tmp3_key), parameters])
        record_local_score[key1] = score1

    if key2 in record_local_score:
        score2 = record_local_score[key2]
    else:
        score2 = feval([score_func, Data, j, list(tmp2_key), parameters])
        record_local_score[key2] = score2

    ch_score = score1 - score2
    desc = [i, j, T]
    return ch_score, desc, record_local_score


def delete_changed_score_fast(Data, i, j, H, NA_set, Paj_set,
                               record_local_score, score_func, parameters):
    """Compute score change for delete i-j/i->j using precomputed NA and Paj sets."""
    tmp1 = NA_set - set(H)
    tmp2 = tmp1 | Paj_set | {i}
    tmp3 = tmp2 - {i}

    tmp3_key = tuple(sorted(tmp3))
    tmp2_key = tuple(sorted(tmp2))

    key1 = (j, tmp3_key)
    key2 = (j, tmp2_key)

    if key1 in record_local_score:
        score1 = record_local_score[key1]
    else:
        score1 = feval([score_func, Data, j, list(tmp3_key), parameters])
        record_local_score[key1] = score1

    if key2 in record_local_score:
        score2 = record_local_score[key2]
    else:
        score2 = feval([score_func, Data, j, list(tmp2_key), parameters])
        record_local_score[key2] = score2

    ch_score = score1 - score2
    desc = [i, j, list(H) if not isinstance(H, list) else H]
    return ch_score, desc, record_local_score
