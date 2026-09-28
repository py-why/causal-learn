"""
XGES -- Extremely Greedy Equivalence Search.

Reference
---------
Achille Nazaret and David Blei, "Extremely Greedy Equivalence Search",
Uncertainty in Artificial Intelligence (UAI) 2024. arXiv:2502.19551.

This is a **clean-room** implementation written *only* from the paper
(arXiv:2502.19551).  No line of the reference implementation
(``github.com/ANazaret/XGES`` C++/Python, or the ``xges`` PyPI package) was
read, copied, adapted, or paraphrased into this file; the ``xges`` PyPI
package is used at most as an output-only numerical parity oracle in the
tests.  This mirrors house practice: causal-learn's ``GES`` is likewise a
from-algorithm port, not vendored source.

What it does
------------
XGES learns a CPDAG (a Markov equivalence class) from observational data by
greedily applying the single best-scoring graph edit across three operators
-- Insert, Delete and Reverse -- maintained together and re-evaluated after
each applied edit.  This differs from GES's phase-separated forward/backward
passes:

* ``Insert(x, y, T)``  -- add ``x -> y`` and orient ``t - y`` as ``t -> y``
  for ``t in T``  (paper Eqs. 6-9, 12).
* ``Delete(x, y, H)``  -- remove ``x - y`` / ``x -> y`` and orient the
  members of ``H``  (paper Table 2).
* ``Reverse(x, y, T)`` -- turn an existing ``y -> x`` into ``x -> y`` (and
  orient ``t - y`` as ``t -> y`` for ``t in T``).  This operator is *not*
  present in GES (paper Table 2, following Hauser & Buehlmann 2012).

``XGES-0`` is the pure three-queue greedy loop (Algorithm 2); the full
``XGES`` adds a local-optima escape (Algorithm 3) that forces each valid
deletion and re-runs the greedy search (forbidding re-insertion of the
deleted edge), keeping the result only when it strictly improves the score.
The two are selectable via ``extended_search``.

Score substrate
---------------
The BIC local score, PDAG<->DAG conversion and ``GeneralGraph`` return shape
are *reused* from causal-learn (no new hard dependency beyond numpy/scipy).
The paper's BIC penalty is ``S = log p_hat(D) - (alpha / 2) * log(n) * |theta|``
with ``alpha = 2`` in the paper's experiments (``alpha = 1`` is the textbook
BIC).  causal-learn's BIC penalises ``lambda_value * (|Pa| + 1) * log(n)`` per
node, i.e. ``lambda_value * log(n)`` per free parameter.  Matching the
per-parameter penalty gives the mapping

    lambda_value = alpha / 2

so the paper default ``alpha = 2`` corresponds to ``lambda_value = 1.0`` and
the textbook ``alpha = 1`` to causal-learn's default ``lambda_value = 0.5``.
(The extra ``+1`` variance parameter per node is a constant that cancels in
every operator's score-delta, so it does not affect the search.)

GeneralGraph edge convention (verified against this checkout's GES.py):
``graph[j, i] = 1`` and ``graph[i, j] = -1`` means ``i -> j``; both ``-1``
means the undirected edge ``i - j``.
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import (
    Combinatorial,
    check_clique_fast,
    delete,
    insert,
    insert_vc2_fast,
    precompute_graph_info,
    score_g,
)
from causallearn.utils.PDAG2DAG import pdag2dag

TAIL = Endpoint.TAIL.value    # -1
ARROW = Endpoint.ARROW.value  # 1
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Operator score-deltas (paper Eqs. 12 and Table 2).
#
# All three are local: for a *valid* operator the change-score equals the
# change in the total decomposable score.  They are exposed publicly so that
# they can be checked against a brute-force full recompute (see TestXGES.py) --
# a valid operator's delta MUST equal score_g(after) - score_g(before).
# ---------------------------------------------------------------------------

def _ins_delta(sf, nbrs, adj, pa, x, y, Tset):
    NA = nbrs[y] & adj[x]
    base = (NA | Tset) | pa[y]
    return sf.score(y, sorted(base | {x})) - sf.score(y, sorted(base))


def _del_delta(sf, nbrs, adj, pa, x, y, Hset):
    NA = nbrs[y] & adj[x]
    base = (NA - Hset) | pa[y]           # (Ne(y) & Ad(x)) \ H  union  Pa(y)
    return sf.score(y, sorted(base - {x})) - sf.score(y, sorted(base | {x}))


def _rev_delta(sf, nbrs, adj, pa, x, y, Tset):
    # existing edge y -> x becomes x -> y ; orient t -> y for t in T.
    NA = nbrs[y] & adj[x]
    base = (NA | Tset) | pa[y]
    d_y = sf.score(y, sorted(base | {x})) - sf.score(y, sorted(base))
    d_x = sf.score(x, sorted(pa[x] - {y})) - sf.score(x, sorted(pa[x]))
    return d_y + d_x


def insert_change_score(score_func: LocalScoreClass, G: GeneralGraph, x: int, y: int, T) -> float:
    """Score change of Insert(x, y, T) on CPDAG ``G`` (paper Eq. 12)."""
    nbrs, adj, pa, _ = precompute_graph_info(G, G.num_vars)
    return _ins_delta(score_func, nbrs, adj, pa, x, y, set(T))


def delete_change_score(score_func: LocalScoreClass, G: GeneralGraph, x: int, y: int, H) -> float:
    """Score change of Delete(x, y, H) on CPDAG ``G`` (paper Table 2)."""
    nbrs, adj, pa, _ = precompute_graph_info(G, G.num_vars)
    return _del_delta(score_func, nbrs, adj, pa, x, y, set(H))


def reverse_change_score(score_func: LocalScoreClass, G: GeneralGraph, x: int, y: int, T) -> float:
    """Score change of Reverse(x, y, T) on CPDAG ``G`` (paper Table 2)."""
    nbrs, adj, pa, _ = precompute_graph_info(G, G.num_vars)
    return _rev_delta(score_func, nbrs, adj, pa, x, y, set(T))


# ---------------------------------------------------------------------------
# Validity helpers.
# ---------------------------------------------------------------------------

def _reverse_path_ok(y, x, block, semi, cap=20000):
    """Reverse validity (paper Table 2): every semi-directed path from ``y`` to
    ``x`` *other than the direct edge* ``y -> x`` must be blocked by ``block``
    (= (Ne(y) & Ad(x)) | T | Ne(x)).

    Enumerated by DFS over simple semi-directed paths (children + undirected
    neighbours).  If the number of paths exceeds ``cap`` we conservatively
    return ``False`` (skip the reverse) rather than risk an unchecked edit --
    the greedy loop's score guard would reject it anyway.
    """
    state = {"ok": True, "count": 0}
    path = [y]
    visited = {y}

    def dfs(u):
        if not state["ok"]:
            return
        if u == x:
            state["count"] += 1
            if state["count"] > cap:
                state["ok"] = False
                return
            if len(path) > 2 and not any(v in block for v in path[1:-1]):
                state["ok"] = False   # an unblocked non-trivial path -> invalid
            return
        for w in semi.get(u, ()):
            if w in visited:
                continue
            visited.add(w)
            path.append(w)
            dfs(w)
            path.pop()
            visited.discard(w)
            if not state["ok"]:
                return

    dfs(y)
    return state["ok"]


def _is_extendable(G: GeneralGraph) -> bool:
    """Return True iff the PDAG ``G`` admits a consistent DAG extension.

    Dor-Tarsi (1992) sink-elimination -- terminates in at most N steps and
    returns False (never hangs) on a non-extendable PDAG.  Used as a safety
    guard before handing an edited graph to ``pdag2dag``.
    """
    A = G.graph
    N = A.shape[0]
    removed = [False] * N
    remaining = N
    while remaining > 0:
        found = -1
        for v in range(N):
            if removed[v]:
                continue
            # sink: no outgoing directed edge v -> k among remaining nodes
            has_out = any(
                (not removed[k]) and k != v and A[k, v] == ARROW and A[v, k] == TAIL
                for k in range(N)
            )
            if has_out:
                continue
            undirected = [k for k in range(N)
                          if not removed[k] and k != v and A[v, k] == TAIL and A[k, v] == TAIL]
            adjacent = [k for k in range(N)
                        if not removed[k] and k != v and (A[v, k] != 0 or A[k, v] != 0)]
            ok = True
            for u in undirected:
                for w in adjacent:
                    if w != u and A[u, w] == 0 and A[w, u] == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                found = v
                break
        if found == -1:
            return False
        removed[found] = True
        remaining -= 1
    return True


# ---------------------------------------------------------------------------
# Applying an operator and (re)canonicalising to a CPDAG.
# ---------------------------------------------------------------------------

def _apply_reverse(G: GeneralGraph, nodes, x, y, T):
    """Turn existing ``y -> x`` into ``x -> y`` and orient ``t -> y`` (t in T)."""
    e = G.get_edge(nodes[y], nodes[x])
    if e is not None:
        G.remove_edge(e)
    G.add_edge(Edge(nodes[x], nodes[y], Endpoint.TAIL, Endpoint.ARROW))  # x -> y
    for t in T:
        et = G.get_edge(nodes[t], nodes[y])
        if et is not None:
            G.remove_edge(et)
        G.add_edge(Edge(nodes[t], nodes[y], Endpoint.TAIL, Endpoint.ARROW))  # t -> y
    return G


def _apply_op(G: GeneralGraph, nodes, kind, params) -> Optional[GeneralGraph]:
    """Apply an operator to a *copy* of ``G`` and return the canonical CPDAG,
    or ``None`` if the resulting PDAG is not DAG-extendable."""
    Gc = deepcopy(G)
    x, y, S = params
    if kind == "insert":
        insert(Gc, x, y, list(S))
    elif kind == "delete":
        delete(Gc, x, y, list(S))
    else:  # reverse
        _apply_reverse(Gc, nodes, x, y, list(S))
    if not _is_extendable(Gc):
        return None
    return dag2cpdag(pdag2dag(Gc))


def _score_cpdag(X, G: GeneralGraph, sf, parameters) -> float:
    """Total decomposable score of the CPDAG ``G`` (scored on a DAG member)."""
    return score_g(X, pdag2dag(deepcopy(G)), sf, parameters)


# ---------------------------------------------------------------------------
# Candidate enumeration and priority ordering.
# ---------------------------------------------------------------------------

def _enumerate_candidates(sf, G, N, maxP, forbidden):
    nbrs, adj, pa, semi = precompute_graph_info(G, N)
    out = []

    # -- Insert(x, y, T): x not adjacent to y (paper I1-I4) ------------------
    for y in range(N):
        if maxP is not None and len(pa[y]) >= maxP:
            continue
        for x in range(N):
            if x == y or x in adj[y]:
                continue
            if frozenset((x, y)) in forbidden:
                continue
            NA = nbrs[y] & adj[x]
            T0 = sorted(nbrs[y] - adj[x])
            for T in Combinatorial(T0):
                C = NA | set(T)
                if not check_clique_fast(G, C):
                    continue
                if not insert_vc2_fast(y, x, C, semi):
                    continue
                out.append((_ins_delta(sf, nbrs, adj, pa, x, y, set(T)),
                            "insert", (x, y, list(T))))

    # -- Delete(x, y, H): x - y or x -> y (paper Table 2) -------------------
    for y in range(N):
        for x in range(N):
            if x == y:
                continue
            if not ((y in nbrs[x]) or (x in pa[y])):
                continue
            NA = nbrs[y] & adj[x]
            for H in Combinatorial(sorted(NA)):
                if not check_clique_fast(G, NA - set(H)):
                    continue
                out.append((_del_delta(sf, nbrs, adj, pa, x, y, set(H)),
                            "delete", (x, y, list(H))))

    # -- Reverse(x, y, T): existing y -> x (paper Table 2) ------------------
    for x in range(N):
        for y in list(pa[x]):
            if maxP is not None and len(pa[y]) >= maxP:
                continue
            NA = nbrs[y] & adj[x]
            T0 = sorted(nbrs[y] - adj[x])
            for T in Combinatorial(T0):
                C = NA | set(T)
                if not check_clique_fast(G, C):
                    continue
                if not _reverse_path_ok(y, x, C | nbrs[x], semi):
                    continue
                out.append((_rev_delta(sf, nbrs, adj, pa, x, y, set(T)),
                            "reverse", (x, y, list(T))))
    return out


def _ordered(cands):
    """Priority order (paper Algorithm 2): Delete (delta >= 0) first, then
    Reverse (delta > 0), then Insert (delta > 0); within each kind, largest
    delta first."""
    dels = sorted([c for c in cands if c[1] == "delete" and c[0] >= -1e-9], key=lambda c: -c[0])
    revs = sorted([c for c in cands if c[1] == "reverse" and c[0] > 1e-9], key=lambda c: -c[0])
    inss = sorted([c for c in cands if c[1] == "insert" and c[0] > 1e-9], key=lambda c: -c[0])
    return dels + revs + inss


# ---------------------------------------------------------------------------
# XGES-0 greedy core and the full-XGES extended search.
# ---------------------------------------------------------------------------

def _run_xges0(X, sf, parameters, N, nodes, maxP, G, forbidden=frozenset()):
    """XGES-0 greedy loop (paper Algorithm 2).  Every accepted move is verified
    against a full recompute of the total score, so the engine never commits a
    score-decreasing edit -- if a candidate's validity were mis-derived it is
    silently skipped, degrading gracefully rather than returning a wrong CPDAG.
    """
    G = deepcopy(G)
    cur = _score_cpdag(X, G, sf, parameters)
    max_iter = 10 * N * N + 50
    for _ in range(max_iter):
        applied = False
        for delta, kind, params in _ordered(_enumerate_candidates(sf, G, N, maxP, forbidden)):
            Gnew = _apply_op(G, nodes, kind, params)
            if Gnew is None:
                continue
            new = _score_cpdag(X, Gnew, sf, parameters)
            ok = (new >= cur - _EPS) if kind == "delete" else (new > cur + _EPS)
            if ok:
                G, cur = Gnew, new
                applied = True
                break
        if not applied:
            break
    return G, cur


def _extended_search(X, sf, parameters, N, nodes, maxP, G, score):
    """Full XGES local-optima escape (paper Algorithm 3): force each valid
    deletion, re-run XGES-0 forbidding re-insertion of the deleted pair, and
    keep the result only if it strictly improves the total score."""
    improved = True
    while improved:
        improved = False
        del_cands = [c for c in _enumerate_candidates(sf, G, N, maxP, frozenset())
                     if c[1] == "delete"]
        del_cands.sort(key=lambda c: -c[0])
        for _, _kind, params in del_cands:
            x, y, _H = params
            Gd = _apply_op(G, nodes, "delete", params)
            if Gd is None:
                continue
            forbidden = frozenset([frozenset((x, y))])
            Gp, sp = _run_xges0(X, sf, parameters, N, nodes, maxP, Gd, forbidden)
            if sp > score + _EPS:
                G, score, improved = Gp, sp, True
                break
    return G, score


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def xges(
    X: ndarray = None,
    score_func: str = "local_score_BIC",
    maxP: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
    node_names: Union[List[str], None] = None,
    cov: Optional[ndarray] = None,
    n: Optional[int] = None,
    lambda_value: Optional[float] = None,
    alpha: Optional[float] = None,
    extended_search: bool = True,
) -> Dict[str, Any]:
    """Perform XGES (Extremely Greedy Equivalence Search).

    Clean-room implementation from Nazaret & Blei, UAI 2024 (arXiv:2502.19551).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features), optional
        Input data.  May be ``None`` if ``cov`` and ``n`` are given (BIC only).
    score_func : str
        Score function name -- same set as ``ges`` (``'local_score_BIC'``,
        ``'local_score_BIC_from_cov'``, ``'local_score_BDeu'``,
        ``'local_score_CV_general'``, ``'local_score_marginal_general'``,
        ``'local_score_CV_multi'``, ``'local_score_marginal_multi'``).
    maxP : float, optional
        Maximum number of parents allowed while searching.
    parameters : dict, optional
        Score-function parameters (e.g. ``{'lambda_value': ...}`` for BIC).
    node_names : list of str, optional
        Variable names; defaults to ``X1 ... Xn``.
    cov, n : ndarray / int, optional
        Covariance matrix and sample size, as an alternative to ``X`` for BIC.
    lambda_value : float, optional
        BIC penalty coefficient (``lambda_value * (|Pa| + 1) * log(n)`` per
        node).  Default 0.5 (textbook BIC).
    alpha : float, optional
        The paper's BIC penalty (``S = ll - (alpha / 2) * log(n) * |theta|``).
        When given (and ``lambda_value`` is not), it is mapped to
        ``lambda_value = alpha / 2`` -- so the paper default ``alpha = 2`` maps
        to ``lambda_value = 1.0``.  Provided so results can be lined up against
        the ``xges`` PyPI parity oracle.
    extended_search : bool
        ``True`` (default) runs full XGES (Algorithm 3, with the local-optima
        escape); ``False`` runs only the XGES-0 greedy core (Algorithm 2).

    Returns
    -------
    Record : dict
        ``Record['G']`` -- learned CPDAG as a ``GeneralGraph`` (same edge
        convention as ``ges``: ``graph[j, i] = 1, graph[i, j] = -1`` means
        ``i -> j``; both ``-1`` means ``i - j``).
        ``Record['score']`` -- total score of the learned CPDAG.
    """
    # --- alpha <-> lambda_value calibration (see module docstring) ----------
    if alpha is not None and lambda_value is None:
        lambda_value = alpha / 2.0

    # --- input handling (mirrors GES.py) -----------------------------------
    if cov is not None and n is not None:
        if X is not None:
            warnings.warn("Both X and cov/n provided. Using cov and n, ignoring X.")
        X = None
    elif X is None:
        raise ValueError("Either X or (cov, n) must be provided.")

    if X is not None and X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    n_features = cov.shape[0] if cov is not None else X.shape[1]

    if lambda_value is not None:
        if parameters is None:
            parameters = {}
        parameters["lambda_value"] = lambda_value

    # --- score-function set-up (same options as GES.py) --------------------
    if score_func == "local_score_CV_general":
        if X is None:
            raise ValueError("local_score_CV_general requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01}
        maxP = n_features if maxP is None else maxP
        N = n_features
        local_score_class = LocalScoreClass(data=X, local_score_fun=local_score_cv_general, parameters=parameters)
    elif score_func == "local_score_marginal_general":
        if X is None:
            raise ValueError("local_score_marginal_general requires raw data X, not cov/n.")
        parameters = {}
        maxP = n_features if maxP is None else maxP
        N = n_features
        local_score_class = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)
    elif score_func == "local_score_CV_multi":
        if X is None:
            raise ValueError("local_score_CV_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01, "dlabel": {i: i for i in range(n_features)}}
        maxP = len(parameters["dlabel"]) if maxP is None else maxP
        N = len(parameters["dlabel"])
        local_score_class = LocalScoreClass(data=X, local_score_fun=local_score_cv_multi, parameters=parameters)
    elif score_func == "local_score_marginal_multi":
        if X is None:
            raise ValueError("local_score_marginal_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"dlabel": {i: i for i in range(n_features)}}
        maxP = len(parameters["dlabel"]) if maxP is None else maxP
        N = len(parameters["dlabel"])
        local_score_class = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)
    elif score_func in ("local_score_BIC", "local_score_BIC_from_cov"):
        maxP = n_features if maxP is None else maxP
        N = n_features
        if parameters is None:
            parameters = {}
        parameters.setdefault("lambda_value", 0.5)
        if cov is not None:
            local_score_class = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters, cov=cov, n=n)
        else:
            local_score_class = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)
    elif score_func == "local_score_BDeu":
        if X is None:
            raise ValueError("local_score_BDeu requires raw data X, not cov/n.")
        maxP = n_features if maxP is None else maxP
        N = n_features
        local_score_class = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    else:
        raise Exception("Unknown function!")

    sf = local_score_class

    if node_names is None:
        node_names = ["X%d" % (i + 1) for i in range(N)]
    nodes = [GraphNode(name) for name in node_names]

    # --- initial (empty) CPDAG ---------------------------------------------
    G = GeneralGraph(nodes)
    G = dag2cpdag(pdag2dag(G))

    # --- XGES-0 greedy core, then (optionally) the extended search ---------
    G, score = _run_xges0(X, sf, parameters, N, nodes, maxP, G)
    if extended_search:
        G, score = _extended_search(X, sf, parameters, N, nodes, maxP, G, score)

    return {"G": G, "score": score}
