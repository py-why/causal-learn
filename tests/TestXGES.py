import sys
import unittest
from copy import deepcopy

import numpy as np

sys.path.append("")

# --- imports from PRE-EXISTING causal-learn modules (not just the new code) ---
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import score_g
from causallearn.utils.PDAG2DAG import pdag2dag

# --- the module under test ---
from causallearn.search.ScoreBased.XGES import (
    xges,
    insert_change_score,
    delete_change_score,
    reverse_change_score,
    _enumerate_candidates,
    _apply_op,
)


# ------------------------------------------------------------------------- #
# A small, known linear-Gaussian ground truth, deterministic across runs.
#   True DAG edges (indices):
#       0->1, 0->3, 1->2, 1->3, 2->3, 2->4, 3->4
#   The corresponding CPDAG is built from the DAG with dag2cpdag (a
#   pre-existing causal-learn utility), so the "truth" is defined by the
#   library itself rather than hand-transcribed.
# ------------------------------------------------------------------------- #
NUM_NODES = 5
TRUTH_DAG_EDGES = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def _build_truth():
    nodes = [GraphNode(f"X{i + 1}") for i in range(NUM_NODES)]
    dag = GeneralGraph(nodes)
    W = np.zeros((NUM_NODES, NUM_NODES))
    rng = np.random.RandomState(0)
    for (a, b) in TRUTH_DAG_EDGES:
        dag.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a -> b
        # weight matrix entry W[b, a]: b = a * weight + ...  (row = child)
        W[b, a] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
    truth_cpdag = dag2cpdag(dag)
    return truth_cpdag, W


def _simulate_linear_gaussian(W, n_samples, seed):
    """X = W X + E  =>  X = (I - W)^{-1} E, standard-normal exogenous noise."""
    N = W.shape[0]
    rng = np.random.RandomState(seed)
    E = rng.randn(n_samples, N)
    return E @ np.linalg.inv(np.eye(N) - W).T


class TestXGES(unittest.TestCase):

    # -- CI gate: recover a known small equivalence class (the correctness floor) --
    def test_xges_simulate_linear_gaussian_shd(self):
        print("Now start test_xges_simulate_linear_gaussian_shd ...")
        truth_cpdag, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=5000, seed=42)

        for extended in (False, True):
            rec = xges(data, score_func="local_score_BIC", extended_search=extended)
            shd = SHD(truth_cpdag, rec["G"]).get_shd()
            print(f"    xges(extended_search={extended})\tSHD: {shd} of {len(TRUTH_DAG_EDGES)}")
            self.assertLessEqual(
                shd, 1,
                f"XGES (extended_search={extended}) SHD={shd} > 1; the learned CPDAG "
                f"differs from the ground-truth equivalence class.",
            )
        print("test_xges_simulate_linear_gaussian_shd passed!\n")

    # -- XGES must land in the SAME equivalence class as GES on the same data --
    def test_xges_matches_ges(self):
        print("Now start test_xges_matches_ges ...")
        _, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=5000, seed=7)

        rec = xges(data, score_func="local_score_BIC")
        g = ges(data, score_func="local_score_BIC")
        shd_between = SHD(g["G"], rec["G"]).get_shd()
        print(f"    SHD(GES, XGES) = {shd_between}")
        self.assertEqual(
            shd_between, 0,
            "XGES and GES returned different CPDAGs on the same data; they should "
            "recover the same Markov equivalence class.",
        )
        print("test_xges_matches_ges passed!\n")

    # -- Honesty check: every operator's local change-score MUST equal the
    #    change in the total decomposable score (a full brute-force recompute). --
    def test_operator_change_scores_match_bruteforce(self):
        print("Now start test_operator_change_scores_match_bruteforce ...")
        truth_cpdag, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=4000, seed=13)
        parameters = {"lambda_value": 0.5}
        sf = LocalScoreClass(
            data=data, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )

        # Use the ground-truth CPDAG as the working graph: it has real directed
        # edges, so the Reverse operator (which needs an existing directed edge)
        # is exercised regardless of what the search happens to recover.
        G = deepcopy(truth_cpdag)
        nodes = G.get_nodes()
        N = G.num_vars
        base = score_g(data, pdag2dag(deepcopy(G)), sf, parameters)

        counts = {"insert": 0, "delete": 0, "reverse": 0}
        for delta, kind, params in _enumerate_candidates(sf, G, N, N, frozenset()):
            x, y, S = params
            # Recompute the operator's advertised delta through its public API,
            # so we also exercise insert/delete/reverse_change_score directly.
            if kind == "insert":
                api_delta = insert_change_score(sf, G, x, y, S)
            elif kind == "delete":
                api_delta = delete_change_score(sf, G, x, y, S)
            else:
                api_delta = reverse_change_score(sf, G, x, y, S)
            self.assertAlmostEqual(delta, api_delta, places=9)

            Gnew = _apply_op(G, nodes, kind, params)
            if Gnew is None:
                continue
            actual = score_g(data, pdag2dag(deepcopy(Gnew)), sf, parameters) - base
            self.assertAlmostEqual(
                delta, actual, places=6,
                msg=f"{kind}{params}: predicted delta {delta} != brute-force {actual}",
            )
            counts[kind] += 1

        print(f"    verified change-scores against brute force: {counts}")
        # The Reverse operator is the XGES-specific one; make sure it was tested.
        self.assertGreater(counts["reverse"], 0, "No valid Reverse operator was exercised.")
        self.assertGreater(counts["insert"], 0)
        self.assertGreater(counts["delete"], 0)
        print("test_operator_change_scores_match_bruteforce passed!\n")

    # -- Parity oracle against the `xges` PyPI package (see VALIDATION.md).
    #    Skipped when `xges` is not installed (dependency installs are blocked
    #    in CI here); a human must run this on Colab before upstreaming. --
    def test_xges_pypi_parity_oracle(self):
        print("Now start test_xges_pypi_parity_oracle ...")
        try:
            import xges as xges_pkg  # noqa: F401
        except Exception:
            self.skipTest(
                "The `xges` PyPI package is not installed in this environment; "
                "run this parity oracle on Colab per VALIDATION.md."
            )

        truth_cpdag, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=5000, seed=42)

        # Paper BIC uses alpha (default 2.0); causal-learn BIC uses
        # lambda_value; the calibrated mapping is lambda_value = alpha / 2.
        alpha = 2.0
        ours = xges(data, score_func="local_score_BIC", alpha=alpha)["G"]

        # Run the reference solver on the SAME data with the SAME penalty.
        # (The public API of the `xges` package is resolved at runtime; the
        # helper below adapts its output to a causal-learn GeneralGraph.)
        ref_adj = _run_reference_xges(data, alpha)
        ref_G = _adjacency_to_cpdag(ref_adj)

        shd = SHD(ref_G, ours).get_shd()
        print(f"    SHD(xges-pypi, causal-learn xges) = {shd}")
        self.assertEqual(
            shd, 0,
            "causal-learn XGES and the `xges` PyPI reference returned different "
            "CPDAGs on the same data with the alpha<->lambda-mapped penalty.",
        )
        print("test_xges_pypi_parity_oracle passed!\n")


# --------------------------------------------------------------------------- #
# Helpers for the parity oracle. Kept out of the module-under-test so the
# reference package is only ever touched as an output oracle.
# --------------------------------------------------------------------------- #
def _run_reference_xges(data, alpha):
    """Run the `xges` PyPI solver and return its CPDAG adjacency matrix, where
    ``A[i, j] == 1`` means an oriented ``i -> j`` and an undirected edge appears
    as ``A[i, j] == A[j, i] == 1`` (the convention `_adjacency_to_cpdag`
    expects). Uses the `xges` 0.1.x API: ``XGES(alpha).fit(X, ...)`` populates
    the solver, and ``get_adjacency_matrix()`` returns the learned CPDAG.
    """
    import xges as xges_pkg

    solver = xges_pkg.XGES(alpha=alpha)
    solver.fit(data, extended_search=True, verbose=0)
    return np.asarray(solver.get_adjacency_matrix())


def _adjacency_to_cpdag(adj):
    """Convert a {0,1} CPDAG adjacency (i->j is adj[i,j]=1; undirected is
    symmetric) into a causal-learn GeneralGraph using its edge convention."""
    N = adj.shape[0]
    nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    G = GeneralGraph(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            aij, aji = adj[i, j], adj[j, i]
            if aij and aji:
                G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
            elif aij:
                G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            elif aji:
                G.add_edge(Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
    return G


if __name__ == "__main__":
    unittest.main()
