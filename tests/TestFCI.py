import sys

sys.path.append("")
import pandas as pd
import unittest
import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, kci


def gen_coef():
    return np.random.uniform(1, 3)


class TestFCI(unittest.TestCase):

    def test_simple_test(self):
        np.random.seed(0)
        sample_size, loc, scale = 200, 0.0, 1.0
        X1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        X2 = X1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X3 = X1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X4 = X2 * gen_coef() + X3 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        data = np.array([X1, X2, X3, X4]).T
        G = fci(data, fisherz, 0.05, verbose=True)

        nodes = G.get_nodes()
        assert G.is_adjacent_to(nodes[0], nodes[1])

        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[0], nodes[1]).add_forbidden_by_node(nodes[1], nodes[0])
        G_with_background_knowledge = fci(data, fisherz, 0.05, verbose=True, knowledge=bk)
        assert not G_with_background_knowledge.is_adjacent_to(nodes[0], nodes[1])


    def test_simple_test2(self):
        np.random.seed(0)
        sample_size, loc, scale = 2000, 0.0, 1.0
        T1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        T2 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        C = np.random.normal(loc=loc, scale=scale, size=sample_size)
        F = C * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        H = C * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        B = F * gen_coef() + T1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        D = H * gen_coef() + T2 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        A = D * gen_coef() + T1 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        E = B * gen_coef() + T2 * gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
        data = np.array([A, B, C, D, E, F, H]).T
        G = fci(data, fisherz, 0.05, verbose=True)
        print(G)

    def test_fritl(self):
        np.random.seed(0)
        sample_size, loc, scale = 1000, 0.0, 1.0
        L1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        L2 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        L3 = np.random.normal(loc=loc, scale=scale, size=sample_size)
        X1 = gen_coef() * L1 + gen_coef() * L2 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X2 = gen_coef() * X1 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X3 = gen_coef() * X2 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X4 = gen_coef() * X1 + gen_coef() * L3 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X5 = gen_coef() * X3 + gen_coef() * L3 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X6 = gen_coef() * L1 + np.random.normal(loc=loc, scale=scale, size=sample_size)
        X7 = gen_coef() * L2 + gen_coef() * L3 + gen_coef() * X6 + np.random.normal(loc=loc, scale=scale,
                                                                                    size=sample_size)
        data = np.array([X1, X2, X3, X4, X5, X6, X7]).T

        G = fci(data, fisherz, 0.05, verbose=True)
        print(G)
