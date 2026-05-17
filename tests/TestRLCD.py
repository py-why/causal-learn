import unittest

import numpy as np

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.HiddenCausal.RLCD import Chi2RankTest, RLCD
from causallearn.utils.GraphUtils import GraphUtils


class TestRLCD(unittest.TestCase):
    def test_rlcd_recovers_linear_gaussian_hidden_structure(self):
        rng = np.random.default_rng(1)
        sample_size = 2000
        L1 = rng.normal(size=sample_size)
        L2 = 0.8 * L1 + rng.normal(size=sample_size)
        X1 = 1.2 * L1 + 0.05 * rng.normal(size=sample_size)
        X2 = 1.4 * L1 + 0.05 * rng.normal(size=sample_size)
        X3 = 1.6 * L1 + 0.05 * rng.normal(size=sample_size)
        X4 = 1.1 * L2 + 0.05 * rng.normal(size=sample_size)
        X5 = 1.3 * L2 + 0.05 * rng.normal(size=sample_size)
        X6 = 1.5 * L2 + 0.05 * rng.normal(size=sample_size)
        data = np.column_stack([X1, X2, X3, X4, X5, X6])
        data = (data - data.mean(axis=0)) / data.std(axis=0)

        cg = RLCD(
            data,
            ranktest_method=Chi2RankTest(data),
            stage1_method="all",
            alpha_dict={0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
            maxk=2,
        )

        self.assertIsInstance(cg, CausalGraph)
        self.assertIsInstance(cg.stage1_cg, CausalGraph)
        self.assertEqual(cg.all_vars[:6], ["X1", "X2", "X3", "X4", "X5", "X6"])
        self.assertEqual(len(cg.all_vars), 8)

        graph = cg.G.graph

        def get_latent_parent(children):
            for latent_idx in range(6, 8):
                if np.all(graph[children, latent_idx] == 1) and np.all(graph[latent_idx, children] == -1):
                    return latent_idx
            return None

        l1_parent = get_latent_parent([0, 1, 2])
        l2_parent = get_latent_parent([3, 4, 5])
        self.assertIsNotNone(l1_parent)
        self.assertIsNotNone(l2_parent)
        self.assertNotEqual(l1_parent, l2_parent)
        self.assertIn((graph[l2_parent, l1_parent], graph[l1_parent, l2_parent]), [(1, -1), (-1, 1)])
        self.assertIsNotNone(GraphUtils.to_pydot(cg.G))


if __name__ == "__main__":
    unittest.main()
