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
from causallearn.search.PermutationBased.gst import GST;
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


def boss(
    X: np.ndarray,
    score_func: str = "local_score_BIC_from_cov",
    parameters: Optional[Dict[str, Any]] = None,
    verbose: Optional[bool] = True,
    node_names: Optional[List[str]] = None,
) -> GeneralGraph:
    """
    Perform a best order score search (BOSS) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BIC_from_cov', 'local_score_BDeu')).
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
        # % k-fold negative cross validated likelihood based on regression in RKHS
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
        
    order = [v for v in range(p)]

    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}
    
    variables = [v for v in order]
    while True:
        improved = False
        random.shuffle(variables)
        if verbose:
            for i, v in enumerate(order):
                parents[v].clear()
                gsts[v].trace(order[:i], parents[v])
            sys.stdout.write("\rBOSS edge count: %i    " % np.sum([len(parents[v]) for v in range(p)]))
            sys.stdout.flush()

        for v in variables:
            improved |= better_mutation(v, order, gsts)
        if not improved: break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    runtime = time.perf_counter() - runtime
    
    if verbose:
        sys.stdout.write("\nBOSS completed in: %.2fs \n" % runtime)
        sys.stdout.flush()

    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)

    return G


def reversed_enumerate(iter, j):
    for w in reversed(iter):
        yield j, w
        j -= 1


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]: best = j
        
    if scores[i] + 1e-6 > scores[best]: return False
    order.remove(v)
    order.insert(best - int(best > i), v)

    return True
