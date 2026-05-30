import unittest

import numpy as np

from causallearn.search.PermutationBased.gst import GST


# Deterministic local scores for vertex 0 over candidate parents {1, 2, 3},
# taken from py-why/causal-learn#264. BOSS uses a higher-is-better score.
# S({1}) and S({2}) use the issue's reported trace values; the remaining
# entries use the issue's local-score table.
_LOCAL_SCORES = {
    frozenset():            0.0,
    frozenset({1}):         2.0837,
    frozenset({2}):         4.3568,
    frozenset({3}):         1.53,
    frozenset({1, 2}):     -0.03,
    frozenset({1, 3}):      3.29,
    frozenset({2, 3}):      2.33,
    frozenset({1, 2, 3}):  -0.25,
}


class _MockScore:
    """Minimal score object exposing only the interface that GST relies on."""

    def __init__(self, n_vars, scores):
        # GST reads data.shape[1] to enumerate the candidate parents.
        self.data = np.zeros((1, n_vars))
        self._scores = scores

    def score(self, i, PAi):
        return self._scores[frozenset(PAi)]

    def score_nocache(self, i, PAi):
        return self._scores[frozenset(PAi)]


class TestGST(unittest.TestCase):
    """Regression test for GST grow-branch ordering (py-why/causal-learn#264).

    BOSS uses a higher-is-better score, and TETRAD's GrowShrinkTree visits grow
    branches in descending growScore order. The trace path -- not just the
    presentation order -- depends on this direction, so the highest-scoring grow
    branch reachable within the prefix must be visited first.
    """

    def test_grow_branches_sorted_descending(self):
        gst = GST(0, _MockScore(4, _LOCAL_SCORES))
        # Force the root to grow over all candidates {1, 2, 3}.
        gst.trace([1, 2, 3])
        grow_scores = [branch.grow_score for branch in gst.root.branches]
        self.assertEqual(grow_scores, sorted(grow_scores, reverse=True))
        # 2 (4.3568) > 1 (2.0837) > 3 (1.53)
        self.assertEqual([branch.add for branch in gst.root.branches], [2, 1, 3])

    def test_trace_follows_highest_scoring_branch(self):
        # With prefix [1, 2], both {1} and {2} are reachable grow branches.
        # Descending order (TETRAD-style) visits {2} first    -> parents {2}.
        # Ascending order (the pre-fix behavior) would visit {1} -> parents {1}.
        gst = GST(0, _MockScore(4, _LOCAL_SCORES))
        parents = []
        score = gst.trace([1, 2], parents)
        self.assertEqual(parents, [2])
        self.assertAlmostEqual(score, 4.3568, places=6)


if __name__ == "__main__":
    unittest.main()
