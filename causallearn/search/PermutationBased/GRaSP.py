import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.search.PermutationBased.gst import GST
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class Order:
    def __init__(self, p, score):

        self.order = list(range(p))
        self.parents = {}
        self.local_scores = {}
        self.edges = 0

        random.shuffle(self.order)

        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])
            # self.local_scores[y] = -score.score_nocache(y, [])

    def get(self, i):
        return self.order[i]

    def set(self, i, y):
        self.order[i] = y

    def index(self, y):
        return self.order.index(y)

    def insert(self, i, y):
        self.order.insert(i, y)

    def pop(self, i=-1):
        return self.order.pop(i)

    def get_parents(self, y):
        return self.parents[y]

    def set_parents(self, y, y_parents):
        self.parents[y] = y_parents

    def get_local_score(self, y):
        return self.local_scores[y]

    def set_local_score(self, y, local_score):
        self.local_scores[y] = local_score

    def get_edges(self):
        return self.edges

    def set_edges(self, edges):
        self.edges = edges

    def bump_edges(self, bump):
        self.edges += bump

    def len(self):
        return len(self.order)


def grasp(
    X: np.ndarray,
    score_func: str = "local_score_BIC_from_cov",
    depth: Optional[int] = 3,
    parameters: Optional[Dict[str, Any]] = None,
    verbose: Optional[bool] = True,
    node_names: Optional[List[str]] = None,
) -> GeneralGraph:
    """
    Perform a greedy relaxation of the sparsest permutation (GRaSP) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BIC_from_cov', 'local_score_BDeu')).
    depth : allowed maximum depth for DFS
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.
    verbose : whether to print the time cost and verbose output of the algorithm.

    Returns
    -------
    G : learned causal graph, where G.graph[j,i] = 1 and G.graph[i,j] = -1 indicates i --> j, G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j.
    """

    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    if score_func == "local_score_CV_general":
        # k-fold negative cross validated likelihood based on regression in RKHS
        if parameters is None:
            parameters = {
                "kfold": 10,  # 10 fold cross validation
                "lambda": 0.01,
            }  # regularization parameter
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters
        )
    elif score_func == "local_score_marginal_general":
        # negative marginal likelihood based on regression in RKHS
        parameters = {}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters
        )
    elif score_func == "local_score_CV_multi":
        # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {
                "kfold": 10,
                "lambda": 0.01,
                "dlabel": {},
            }  # regularization parameter
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters
        )
    elif score_func == "local_score_marginal_multi":
        # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters
        )
    elif score_func == "local_score_BIC":
        # SEM BIC score
        warnings.warn("Using 'local_score_BIC_from_cov' instead for efficiency")
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BIC_from_cov":
        # SEM BIC score
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BDeu":
        # BDeu score
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )
    else:
        raise Exception("Unknown function!")

    score = localScoreClass
    gsts = [GST(i, score) for i in range(p)]

    node_names = [("X%d" % (i + 1)) for i in range(p)] if node_names is None else node_names
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)

    runtime = time.perf_counter()
    order = Order(p, score)

    for i in range(p):
        y = order.get(i)
        y_parents = order.get_parents(y)

        candidates = [order.get(j) for j in range(0, i)]
        local_score = gsts[y].trace(candidates, y_parents)
        order.set_local_score(y, local_score)
        order.bump_edges(len(y_parents))

    while dfs(depth - 1, set(), [], order, gsts):
        if verbose:
            sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges())
            sys.stdout.flush()

    runtime = time.perf_counter() - runtime

    if verbose:
        sys.stdout.write("\nGRaSP completed in: %.2fs \n" % runtime)
        sys.stdout.flush()

    for y in range(p):
        for x in order.get_parents(y):
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)

    return G


# performs a dfs over covered tucks
def dfs(depth: int, flipped: set, history: List[set], order, gsts):

    cache = [{}, {}, {}, 0]

    indices = list(range(order.len()))
    random.shuffle(indices)

    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y)
        random.shuffle(y_parents)

        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)

            if len(history) > 0 and not covered:
                continue

            j = order.index(x)

            for k in range(j, i + 1):
                z = order.get(k)
                cache[0][k] = z
                cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()

            tuck(i, j, order)
            edge_bump, score_bump = update(i, j, order, gsts)

            # because things that should be zero sometimes are not
            if score_bump > 1e-6:
                order.bump_edges(edge_bump)
                return True

            # ibid
            if score_bump > -1e-6:
                flipped = flipped ^ set(
                    [
                        tuple(sorted([x, z]))
                        for z in order.get_parents(x)
                        if order.index(z) < i
                    ]
                )

                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]

            for k in range(j, i + 1):
                z = cache[0][k]
                order.set(k, z)
                order.set_parents(z, cache[1][k])
                order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])

    return False


# updates the parents and scores after a tuck
def update(i: int, j: int, order, gsts):

    edge_bump = 0
    old_score = 0
    new_score = 0

    for k in range(j, i + 1):
        z = order.get(k)
        z_parents = order.get_parents(z)

        edge_bump -= len(z_parents)
        old_score += order.get_local_score(z)

        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]
        local_score = gsts[z].trace(candidates, z_parents)
        order.set_local_score(z, local_score)

        edge_bump += len(z_parents)
        new_score += local_score

    return edge_bump, new_score - old_score


# tucks the node at position i into position j
def tuck(i: int, j: int, order):

    ancestors = []
    get_ancestors(order.get(i), ancestors, order)

    shift = 0

    for k in range(j + 1, i + 1):
        if order.get(k) in ancestors:
            order.insert(j + shift, order.pop(k))
            shift += 1


# returns ancestors of y
def get_ancestors(y: int, ancestors: List[int], order):

    ancestors.append(y)

    for x in order.get_parents(y):
        if x not in ancestors:
            get_ancestors(x, ancestors, order)
