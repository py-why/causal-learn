"""
Determinism-aware Greedy Equivalence Search (DGES)

Based on: "On Causal Discovery in the Presence of Deterministic Relations"
Li et al., NeurIPS 2024.

DGES handles causal discovery when some variables have deterministic
relationships (zero noise variance). Following Algorithm 1 from
"On Causal Discovery in the Presence of Deterministic Relations"
(Li et al., NeurIPS 2024), it comprises three phases:
  1. Detect MinDCs (minimal deterministic clusters) from data
  2. Run modified GES with DC-aware constraints:
     - Forward: force add edge i->j when PA_j determines V_i
     - Backward: protect edges within MinDC sets
  3. Exact search on DC + neighbors to refine the graph

Key modifications over standard GES:
  - Modified BIC score: log(sigma + epsilon) instead of log(sigma),
    to avoid -inf scores for deterministic relations
  - Supports nonlinear score functions (e.g. local_score_CV_general)
    with kernel-regression-based MinDC detection
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.search.ScoreBased import ExactSearch as _ExactSearchModule
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import (
    local_score_cv_general,
    local_score_marginal_general,
    local_score_cv_multi,
    local_score_marginal_multi,
    local_score_BIC_from_cov,
)
from causallearn.utils.GESUtils import (
    Combinatorial,
    find_subset_include,
    score_g,
    feval,
    precompute_graph_info,
    check_clique_fast,
    insert_vc2_fast,
    insert_changed_score_fast,
    delete_changed_score_fast,
    insert,
    delete,
)

# Score functions that assume linearity (covariance-based)
_LINEAR_SCORES = {
    "local_score_BIC",
    "local_score_BIC_from_cov",
    "local_score_BIC_from_cov_deterministic",
    "local_score_BDeu",
}

# Mapping from score name to function for nonlinear scores
_NONLINEAR_SCORE_MAP = {
    "local_score_CV_general": local_score_cv_general,
    "local_score_marginal_general": local_score_marginal_general,
    "local_score_CV_multi": local_score_cv_multi,
    "local_score_marginal_multi": local_score_marginal_multi,
}


# =====================================================================
# Modified BIC score for deterministic relations
# =====================================================================

def local_score_BIC_from_cov_deterministic(Data, i, PAi, parameters=None):
    """
    BIC score modified for deterministic relations.

    Change from standard BIC: log(sigma) -> log(sigma + epsilon)
    where epsilon is a small constant (default 0.01), so that
    deterministic relations (sigma=0) get a finite score instead of -inf.

    Parameters
    ----------
    Data : tuple (cov, n)
        Covariance matrix and number of samples.
    i : int
        Target variable index.
    PAi : list of int
        Parent variable indices.
    parameters : dict, optional
        'lambda_value': penalty coefficient (default 0.5 = standard BIC)
        'det_epsilon': epsilon for deterministic score (default 0.01)
    """
    cov, n = Data

    if parameters is None:
        parameters = {}
    lambda_value = parameters.get("lambda_value", 0.5)
    det_epsilon = parameters.get("det_epsilon", 0.01)

    sigma = cov[i, i]
    if len(PAi) > 0:
        yX = cov[np.ix_([i], PAi)]
        XX = cov[np.ix_(PAi, PAi)]
        try:
            XX_inv = np.linalg.inv(XX)
        except np.linalg.LinAlgError:
            XX_inv = np.linalg.pinv(XX)
        sigma = float(cov[i, i] - yX @ XX_inv @ yX.T)
    if sigma <= 0:
        sigma = 0.0

    likelihood = -0.5 * n * (1 + np.log(sigma + det_epsilon))
    penalty = lambda_value * (len(PAi) + 1) * np.log(n)
    return likelihood - penalty


# =====================================================================
# ExactSearch BIC score for deterministic relations
# =====================================================================

def _bic_score_node_deterministic(X, i, structure, det_epsilon=0.01):
    """BIC score for exact search, modified for deterministic relations.

    Uses log(sigma + det_epsilon) instead of log(sigma) to
    avoid -inf for deterministic relations where sigma ≈ 0.
    sigma is the residual variance (consistent with covariance-based BIC).
    """
    structure = list(structure)
    n, d = X.shape
    if len(structure) == 0:
        sigma = float(np.var(X[:, i]))
    else:
        a = X[:, structure]
        b = X[:, i]
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        sigma = float(np.var(b - a @ coef))

    bic = n * np.log(sigma + det_epsilon) + len(structure) * np.log(n)
    return float(bic)


# =====================================================================
# MinDC detection
# =====================================================================

def detect_deterministic_clusters(X, adj_matrix, threshold=1e-5,
                                  nonlinear=False):
    """
    Detect minimal deterministic clusters from data and a DAG/CPDAG.

    For each variable with parents in the estimated graph, check if
    the residual variance is near zero (deterministic relation).

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data matrix.
    adj_matrix : ndarray, shape (d, d)
        Adjacency matrix where adj_matrix[i,j]=1 means i->j or i-j.
    threshold : float
        For linear: absolute residual variance threshold (default 1e-5).
        For nonlinear: relative threshold — a relation is deterministic
        if residual_var / total_var < threshold, i.e. R^2 > 1-threshold.
        Default 1e-5 works for linear; for nonlinear, DGES auto-adjusts
        to 0.01 (R^2 > 0.99) if threshold < 0.001.
    nonlinear : bool
        If True, use kernel ridge regression instead of linear regression
        to compute residuals, enabling detection of nonlinear deterministic
        relations (e.g. X2 = X0^2 + sin(X1)).

    Returns
    -------
    det_clusters : list of ndarray
        Each element is an array of variable indices forming a
        minimal deterministic cluster (the deterministic node + its parents).
    """
    d = X.shape[1]
    X_centered = X - np.mean(X, axis=0)
    det_clusters = []

    # For nonlinear, cross-validated kernel regression residuals are never
    # exactly zero. Use a relative threshold (R^2-based) instead.
    if nonlinear and threshold < 1e-3:
        nl_threshold = 0.01  # R^2 > 0.99
    else:
        nl_threshold = threshold

    for i in range(d):
        parents = np.where(adj_matrix[:, i] != 0)[0]
        if len(parents) > 0:
            y_i = X_centered[:, i]
            x_i = X_centered[:, parents]

            if nonlinear:
                sigma = _nonlinear_residual_variance(x_i, y_i)
                var_y = np.var(y_i)
                # Use relative criterion: sigma/var(y) < threshold
                is_det = (var_y > 0) and (sigma / var_y < nl_threshold)
            else:
                coef = np.linalg.lstsq(x_i, y_i, rcond=None)[0]
                sigma = np.var(y_i - x_i @ coef)
                is_det = sigma < threshold

            if is_det:
                if nonlinear:
                    # For nonlinear, all parents contribute
                    det_cluster = np.concatenate(([i], parents))
                else:
                    det_cluster_idx = np.where(np.abs(coef) > threshold)[0]
                    det_cluster = np.concatenate(([i], parents[det_cluster_idx]))
                det_clusters.append(det_cluster)

    return det_clusters


def _nonlinear_residual_variance(X_parents, y):
    """
    Compute residual variance using kernel ridge regression (RBF kernel).

    Uses 5-fold cross-validation residuals to avoid overfitting, which would
    make the residual variance artificially low.

    Parameters
    ----------
    X_parents : ndarray, shape (n, p)
        Parent variable data.
    y : ndarray, shape (n,)
        Target variable data.

    Returns
    -------
    sigma : float
        Residual variance (from cross-validated predictions).
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import cross_val_predict

    n = X_parents.shape[0]
    X_p = np.atleast_2d(X_parents)
    if X_p.shape[0] == 1:
        X_p = X_p.T

    # Use median heuristic for RBF bandwidth
    from sklearn.metrics.pairwise import euclidean_distances
    # Subsample for efficiency if large
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        dists = euclidean_distances(X_p[idx])
    else:
        dists = euclidean_distances(X_p)
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

    krr = KernelRidge(alpha=1e-3, kernel='rbf', gamma=gamma)

    # Cross-validated predictions to avoid overfitting
    n_folds = min(5, n)
    y_pred = cross_val_predict(krr, X_p, y, cv=n_folds)
    sigma = np.var(y - y_pred)

    return sigma


def detect_mindc_from_data(X, threshold=1e-5, nonlinear=False):
    """
    Detect minimal deterministic clusters from data alone (Phase 1 of Algorithm 1).

    Unlike detect_deterministic_clusters(), this does NOT require a pre-estimated
    graph. For each variable, it regresses on ALL other variables and checks if
    the residual variance is near zero. If so, it uses backward elimination to
    find the minimal parent set.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data matrix.
    threshold : float
        For linear: absolute residual variance threshold.
        For nonlinear: relative threshold (R^2-based).
    nonlinear : bool
        If True, use kernel ridge regression.

    Returns
    -------
    mindcs : list of (int, frozenset)
        Each element is (det_node, parent_set) where det_node is the
        deterministic variable and parent_set is the minimal set of
        variables that determine it.
    """
    n, d = X.shape
    X_centered = X - np.mean(X, axis=0)
    mindcs = []

    if nonlinear and threshold < 1e-3:
        rel_threshold = 0.01  # R^2 > 0.99
    else:
        rel_threshold = threshold

    for i in range(d):
        others = [j for j in range(d) if j != i]
        y = X_centered[:, i]
        x_all = X_centered[:, others]

        if nonlinear:
            sigma = _nonlinear_residual_variance(x_all, y)
            var_y = np.var(y)
            is_det = (var_y > 0) and (sigma / var_y < rel_threshold)
        else:
            coef = np.linalg.lstsq(x_all, y, rcond=None)[0]
            sigma = np.var(y - x_all @ coef)
            is_det = sigma < threshold

        if is_det:
            # Backward elimination to find minimal parent set
            parent_set = set(others)
            changed = True
            while changed:
                changed = False
                for j in sorted(parent_set):
                    reduced = parent_set - {j}
                    if len(reduced) == 0:
                        continue
                    reduced_list = sorted(reduced)
                    x_red = X_centered[:, reduced_list]
                    if nonlinear:
                        sig = _nonlinear_residual_variance(x_red, y)
                        still_det = (var_y > 0) and (sig / var_y < rel_threshold)
                    else:
                        c = np.linalg.lstsq(x_red, y, rcond=None)[0]
                        sig = np.var(y - x_red @ c)
                        still_det = sig < threshold
                    if still_det:
                        parent_set = reduced
                        changed = True
                        break
            mindcs.append((i, frozenset(parent_set)))

    return mindcs


def _pa_determines(pa_set, node, mindcs):
    """
    Check if pa_set determines node according to pre-computed MinDC info.

    Parameters
    ----------
    pa_set : set
        Current parent set of some variable in the graph.
    node : int
        The node to check.
    mindcs : list of (int, frozenset)
        Pre-computed MinDCs from detect_mindc_from_data().

    Returns
    -------
    bool
        True if pa_set contains all parents needed to determine node.
    """
    for det_node, parents in mindcs:
        if det_node == node and parents.issubset(pa_set):
            return True
    return False


def _in_same_mindc_set(i, j, mindc_sets):
    """
    Check if i and j are in the same MinDC set.

    Parameters
    ----------
    i, j : int
        Node indices.
    mindc_sets : list of frozenset
        MinDC sets (each is a frozenset of variable indices).

    Returns
    -------
    bool
    """
    for s in mindc_sets:
        if i in s and j in s:
            return True
    return False


def find_cluster_neighbors(det_clusters, adj_matrix, d):
    """
    Find all neighbors of deterministic cluster nodes in the graph.

    Returns
    -------
    all_neighbors : list of int
        Sorted list of all nodes in deterministic clusters and their neighbors.
    """
    dc_nodes = set()
    for cluster in det_clusters:
        dc_nodes.update(cluster)

    all_neighbors = set(dc_nodes)
    sym_adj = adj_matrix + adj_matrix.T
    for i in range(d):
        if i in dc_nodes:
            neighbors = np.where(sym_adj[:, i] != 0)[0]
            all_neighbors.update(neighbors)

    return sorted(list(all_neighbors))


# =====================================================================
# Main DGES function
# =====================================================================

# =====================================================================
# Score function setup helper
# =====================================================================

def _setup_score_func(X, score_func, maxP, parameters):
    """Set up LocalScoreClass for the given score function name.

    Returns (localScoreClass, maxP, parameters) with defaults filled in.
    Defaults are aligned with GES (lambda_value=0.5, maxP=N).
    """
    N = X.shape[1]

    if score_func == "local_score_CV_general":
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01}
        if maxP is None:
            maxP = N
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general,
            parameters=parameters,
        )
    elif score_func == "local_score_marginal_general":
        parameters = {}
        if maxP is None:
            maxP = N
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general,
            parameters=parameters,
        )
    elif score_func == "local_score_CV_multi":
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01, "dlabel": {}}
            for i in range(N):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi,
            parameters=parameters,
        )
    elif score_func == "local_score_marginal_multi":
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(N):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi,
            parameters=parameters,
        )
    elif score_func in ("local_score_BIC", "local_score_BIC_from_cov"):
        if maxP is None:
            maxP = N
        if parameters is None:
            parameters = {}
        if "lambda_value" not in parameters:
            parameters["lambda_value"] = 0.5
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov,
            parameters=parameters,
        )
    elif score_func == "local_score_BIC_from_cov_deterministic":
        if maxP is None:
            maxP = N
        if parameters is None:
            parameters = {}
        if "lambda_value" not in parameters:
            parameters["lambda_value"] = 0.5
        if "det_epsilon" not in parameters:
            parameters["det_epsilon"] = 0.01
        lsc = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov_deterministic,
            parameters=parameters,
        )
    else:
        raise ValueError(f"Unknown score function: {score_func}")

    return lsc, maxP, parameters


# =====================================================================
# DGES main entry point (Algorithm 1, paper-faithful)
# =====================================================================

def dges(
    X: np.ndarray,
    score_func: str = "local_score_BIC_from_cov_deterministic",
    maxP: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
    node_names: Union[List[str], None] = None,
    det_threshold: float = 1e-5,
    det_epsilon: float = 0.01,
    exact_search_method: str = "astar",
    skip_exact_search: bool = True,
) -> Dict[str, Any]:
    """
    Determinism-aware Greedy Equivalence Search (DGES).

    Three phases following Algorithm 1 from "On Causal Discovery in the
    Presence of Deterministic Relations" (Li et al., NeurIPS 2024):

      Phase 1: Detect MinDCs from data (before running GES)
      Phase 2: Run modified GES with DC-aware forward/backward constraints
        - Forward: when PA_j determines V_i, force add edge i->j
          (V_i is redundant given PA_j, causing false independence)
        - Backward: when i and j are in the same MinDC, keep edge
          (trust dependency, don't delete)
      Phase 3: Exact search on DC + neighbors (optional, disabled by default)

    By default only Phase 1 + 2 are executed (fast). Set skip_exact_search=False
    to also run Phase 3, which can be slow for large graphs.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input data.
    score_func : str
        Score function name.
    maxP : float, optional
        Maximum number of parents allowed.
    parameters : dict, optional
        Score function parameters.
    node_names : list of str, optional
        Names for the variables.
    det_threshold : float
        Threshold for detecting deterministic relations.
    det_epsilon : float
        Epsilon for modified BIC: log(sigma + epsilon).
    exact_search_method : str
        'astar' or 'dp'. Only used when skip_exact_search=False.
    skip_exact_search : bool
        If True (default), skip Phase 3 (exact search) for faster execution.
        Set to False to run the full 3-phase algorithm.

    Returns
    -------
    Record : dict
        'G': learned causal graph (GeneralGraph)
        'update1': forward step updates
        'update2': backward step updates
        'G_step1': graphs at each forward step
        'G_step2': graphs at each backward step
        'score': final score
        'det_clusters': detected deterministic clusters
        'mindcs': raw MinDC info as list of (det_node, frozenset(parents))
        'exact_search_nodes': nodes used in exact search
    """
    import warnings

    N = X.shape[1]

    if parameters is None:
        parameters = {}
    if "det_epsilon" not in parameters:
        parameters["det_epsilon"] = det_epsilon

    is_nonlinear = score_func not in _LINEAR_SCORES

    # --- Phase 1: Detect MinDCs from data ---
    mindcs = detect_mindc_from_data(
        X, threshold=det_threshold, nonlinear=is_nonlinear,
    )
    det_clusters = [
        np.array([det_node] + sorted(parents))
        for det_node, parents in mindcs
    ]

    # Build MinDC sets (de-duplicated frozensets of all members)
    mindc_sets = []
    for det_node, parents in mindcs:
        s = frozenset({det_node} | parents)
        if s not in mindc_sets:
            mindc_sets.append(s)

    # --- Phase 2: Modified GES with DC-aware constraints ---
    score_func_obj, maxP, parameters = _setup_score_func(
        X, score_func, maxP, parameters,
    )

    if node_names is None:
        node_names = [("X%d" % (i + 1)) for i in range(N)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)
    score = score_g(X, G, score_func_obj, parameters)
    G = pdag2dag(G)
    G = dag2cpdag(G)

    # --- Forward phase (with DC-aware edge forcing) ---
    # Modification 1 (forward): when PA_j determines V_i, force edge i->j.
    # This handles the case where V_i is redundant given PA_j (all of V_i's
    # MinDC parents are in PA_j), causing a false independence.
    # We do NOT force when PA_j determines V_j, as that would force ALL
    # edges to j once j is fully determined.
    record_local_score = {}
    score_new = score
    update1 = []
    G_step1 = []

    while True:
        score = score_new
        max_chscore = -1e7
        max_desc = []
        force_add = False

        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == 0
                    and G.graph[j, i] == 0
                    and i != j
                    and len(_pa[j]) <= maxP
                ):
                    # Case B only: PA_j determines V_i
                    # (V_i is a det function of PA_j → false independence)
                    dc_forced = _pa_determines(_pa[j], i, mindcs)

                    NA = _nbrs[j] & _adj[i]
                    T0 = sorted(_nbrs[j] - _adj[i])
                    sub = Combinatorial(T0)
                    S = np.zeros(len(sub))
                    for k in range(len(sub)):
                        if S[k] < 2:
                            T_set = set(sub[k])
                            NAT = NA | T_set
                            V1 = check_clique_fast(G, NAT)
                            if V1:
                                if not S[k]:
                                    V2 = insert_vc2_fast(j, i, NAT, _semi)
                                else:
                                    V2 = 1
                                if V2:
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = (
                                        insert_changed_score_fast(
                                            X, i, j, sub[k],
                                            NA, _pa[j],
                                            record_local_score,
                                            score_func_obj,
                                            parameters,
                                        )
                                    )

                                    if dc_forced:
                                        # Force: ensure this edge is treated
                                        # as an improvement (positive chscore)
                                        chscore = max(chscore, 1.0)

                                    if chscore > max_chscore:
                                        max_chscore = chscore
                                        max_desc = desc
                                        force_add = dc_forced
                            else:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0 and not force_add:
                break
            G = insert(G, max_desc[0], max_desc[1], max_desc[2])
            update1.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)
        else:
            score_new = score
            break

    # --- Backward phase (with DC-aware edge protection) ---
    # Modification 1 (backward): when i and j are in the same MinDC set,
    # don't delete the edge (trust the dependency).
    score_new = score
    update2 = []
    G_step2 = []

    while True:
        score = score_new
        max_chscore = -1e7
        max_desc = []

        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (j in _nbrs[i]) or (i in _pa[j]):
                    # Protect edges within MinDC sets
                    if _in_same_mindc_set(i, j, mindc_sets):
                        continue

                    NA = _nbrs[j] & _adj[i]
                    H0 = sorted(NA)
                    sub = Combinatorial(H0)
                    S = np.ones(len(sub))
                    for k in range(len(sub)):
                        if S[k] == 1:
                            H_set = set(sub[k])
                            V = check_clique_fast(G, NA - H_set)
                            if V:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2
                        else:
                            V = 1

                        if V:
                            chscore, desc, record_local_score = (
                                delete_changed_score_fast(
                                    X, i, j, sub[k],
                                    NA, _pa[j],
                                    record_local_score,
                                    score_func_obj,
                                    parameters,
                                )
                            )
                            if chscore > max_chscore:
                                max_chscore = chscore
                                max_desc = desc

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0:
                break
            G = delete(G, max_desc[0], max_desc[1], max_desc[2])
            update2.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    # --- Phase 3: Exact search on DC + neighbors ---
    exact_search_nodes = []

    if not skip_exact_search and len(det_clusters) > 0:
        # Build adjacency from current graph for neighbor finding
        raw = G.graph
        adj_for_detection = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if raw[i, j] == -1 and raw[j, i] == 1:
                    adj_for_detection[i, j] = 1
                elif raw[i, j] == -1 and raw[j, i] == -1:
                    adj_for_detection[i, j] = 1
                    adj_for_detection[j, i] = 1

        exact_search_nodes = find_cluster_neighbors(
            det_clusters, adj_for_detection, N,
        )

        if len(exact_search_nodes) > 0:
            _orig_bic_score_node = _ExactSearchModule.bic_score_node

            if is_nonlinear:
                X_sub = X[:, exact_search_nodes]
                score_fun = _NONLINEAR_SCORE_MAP.get(score_func)
                if score_fun is None:
                    warnings.warn(
                        f"Nonlinear exact search not supported for {score_func}. "
                        "Skipping exact search."
                    )
                else:
                    sub_params = dict(parameters)
                    sub_score_class = LocalScoreClass(
                        data=X_sub,
                        local_score_fun=score_fun,
                        parameters=sub_params,
                    )

                    def _nonlinear_score_node(X_data, i, structure,
                                              _sc=sub_score_class):
                        return _sc.score(i, list(structure))

                    _ExactSearchModule.bic_score_node = _nonlinear_score_node
            else:
                _eps = parameters.get("det_epsilon", det_epsilon)
                X_centered = X - np.mean(X, axis=0)
                X_sub = X_centered[:, exact_search_nodes]
                _ExactSearchModule.bic_score_node = \
                    lambda X, i, structure, _e=_eps: \
                    _bic_score_node_deterministic(X, i, structure,
                                                 det_epsilon=_e)

            try:
                dag_sub, _ = bic_exact_search(
                    X_sub,
                    super_graph=None,
                    search_method=exact_search_method,
                )

                # Combine: keep GES result for non-DC nodes,
                # exact search result for DC+neighbor nodes
                non_dc_nodes = set(range(N)) - set(exact_search_nodes)

                combined_adj = np.zeros((N, N))
                for ii in non_dc_nodes:
                    for jj in range(N):
                        if adj_for_detection[ii, jj] != 0:
                            combined_adj[ii, jj] = adj_for_detection[ii, jj]
                        if adj_for_detection[jj, ii] != 0:
                            combined_adj[jj, ii] = adj_for_detection[jj, ii]

                for si in range(len(exact_search_nodes)):
                    for sj in range(len(exact_search_nodes)):
                        if dag_sub[si, sj] == 1:
                            gi = exact_search_nodes[si]
                            gj = exact_search_nodes[sj]
                            combined_adj[gi, gj] = 1

                new_nodes = [GraphNode(name) for name in node_names]
                G = GeneralGraph(new_nodes)

                for ii in range(N):
                    for jj in range(N):
                        if combined_adj[ii, jj] == 1 and combined_adj[jj, ii] == 0:
                            G.add_edge(Edge(
                                new_nodes[ii], new_nodes[jj],
                                Endpoint.TAIL, Endpoint.ARROW,
                            ))
                        elif (combined_adj[ii, jj] == 1
                              and combined_adj[jj, ii] == 1
                              and ii < jj):
                            G.add_edge(Edge(
                                new_nodes[ii], new_nodes[jj],
                                Endpoint.TAIL, Endpoint.TAIL,
                            ))

                G = pdag2dag(G)
                G = dag2cpdag(G)

            except Exception as e:
                warnings.warn(
                    f"Exact search failed: {e}. "
                    "Returning modified GES result."
                )
            finally:
                _ExactSearchModule.bic_score_node = _orig_bic_score_node

    Record = {
        "update1": update1,
        "update2": update2,
        "G_step1": G_step1,
        "G_step2": G_step2,
        "G": G,
        "score": score,
        "det_clusters": det_clusters,
        "mindcs": mindcs,
        "mindc_sets": mindc_sets,
        "exact_search_nodes": exact_search_nodes,
    }
    return Record
